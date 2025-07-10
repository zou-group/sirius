import os
import json
import time
import datetime
from openai import OpenAI
from agent import Agent

client = OpenAI()
model = "gpt-3.5-turbo"

# Define agents
solver = Agent(name="solver", model=model, next_agent=None, pre_agent=None, api_type=client)
analyst = Agent(name="analyst", model=model, next_agent=solver, pre_agent=None, api_type=client)

agents = [analyst, solver]


def fine_tune(model_name: str):
    log_dir = f"logs/fine_tune/{model_name}"
    os.makedirs(log_dir, exist_ok=True)

    id_file = os.path.join(log_dir, "finetuning_ids.jsonl")
    job_ids = {}
    id_records = []
    model_records = []

    for agent in agents:
        file_path = os.path.join(log_dir, f"finetune_{agent.name}.jsonl")
        print(f"Submitting fine-tuning for {agent.name}: {file_path}")

        fine_tuned_object = agent.fine_tune(file_path)
        job_id = fine_tuned_object.id
        job_ids[agent.name] = job_id
        print(f"{agent.name}'s job id: {job_id}")
        id_records.append({"agent": agent.name, "id": job_id})

    while True:
        all_done = True
        for agent_name, job_id in job_ids.items():
            job = client.fine_tuning.jobs.retrieve(job_id)
            if job.fine_tuned_model is None:
                print(f"{agent_name}'s job still running...")
                all_done = False
        if all_done:
            break
        time.sleep(60)

    for agent in agents:
        job_id = job_ids[agent.name]
        job = client.fine_tuning.jobs.retrieve(job_id)
        agent.model = job.fine_tuned_model
        print(f"{agent.name} fine-tuned to: {agent.model}")
        model_records.append({"agent": agent.name, "model": agent.model})

    # Step 4: Write all records
    with open(id_file, 'a', encoding='utf-8') as f:
        for record in id_records + model_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    fine_tune(model)
    end_time = datetime.datetime.now()
    print('Time cost:', round((end_time - start_time).total_seconds() / 60, 2), 'mins')
