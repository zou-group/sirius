import os
import json
import time
import datetime
from openai import OpenAI
from agent import Agent

client = OpenAI()
model = "gpt-3.5-turbo"

# Define agents
agents = [
    Agent(name="feedback", model=model, next_agent=None, pre_agent=None, api_type=client),
    Agent(name="judge", model=model, next_agent=None, pre_agent=None, api_type=client),
    Agent(name="actor", model=model, next_agent=None, pre_agent=None, api_type=client)
]


def fine_tune(model_name: str):
    job_ids = {}
    dict_path = f"logs/actor_critic/generate/{model_name}_{model_name}"
    os.makedirs(dict_path, exist_ok=True)
    id_file_path = f"{dict_path}/finetuning_ids.jsonl"

    for agent in agents:
        file_path = f"{dict_path}/finetune_{agent.name}.jsonl"
        print(f"Submitting fine-tuning for {agent.name}: {file_path}")
        job = agent.fine_tune(file_path)
        job_ids[agent.name] = job.id

        with open(id_file_path, "a") as f:
            f.write(json.dumps({"agent": agent.name, "id": job.id}) + "\n")


    while True:
        all_done = True
        for agent_name, job_id in job_ids.items():
            job = client.fine_tuning.jobs.retrieve(job_id)
            if job.fine_tuned_model is None:
                print(f"{agent_name}'s fine-tuning still in progress...")
                all_done = False
        if all_done:
            break
        time.sleep(60)

    for agent in agents:
        job_id = job_ids[agent.name]
        job = client.fine_tuning.jobs.retrieve(job_id)
        agent.model = job.fine_tuned_model
        print(f"{agent.name} fine-tuned model: {agent.model}")

        with open(id_file_path, "a") as f:
            f.write(json.dumps({"agent": agent.name, "model": agent.model}) + "\n")


if __name__ == '__main__':
    start = datetime.datetime.now()
    fine_tune(model)
    end = datetime.datetime.now()
    print("Time cost:", round((end - start).total_seconds() / 60, 2), "mins")
