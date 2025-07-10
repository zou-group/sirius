import os
import json
import time
import datetime
from openai import OpenAI

from args import parse_args
from Problem_solving.PhyChem.agent import physicist, mathematician, summerizer

client = OpenAI()


def fine_tune(args, agents):
    log_dir = f"logs/{args.subject}/{args.prompt_type}/{args.model}/"
    os.makedirs(log_dir, exist_ok=True)

    id_file = os.path.join(log_dir, "finetuning_ids.jsonl")
    id_records = []
    model_records = []

    job_ids = {}

    for agent in agents:
        file_path = os.path.join(log_dir, f"finetune_{agent.name}.jsonl")
        print(f"Submitting fine-tuning for {agent.name} using file: {file_path}")

        fine_tuned_object = agent.fine_tune(file_path)
        job_id = fine_tuned_object.id
        job_ids[agent.name] = job_id

        print(f"{agent.name}'s fine-tuning job id: {job_id}")
        id_records.append({"agent": agent.name, "id": job_id})

    all_completed = False
    while not all_completed:
        all_completed = True
        for agent_name, job_id in job_ids.items():
            job = client.fine_tuning.jobs.retrieve(job_id)
            if job.fine_tuned_model is None:
                print(f"{agent_name}'s fine-tuning still in progress...")
                all_completed = False
        if not all_completed:
            time.sleep(60)

    for agent in agents:
        job_id = job_ids[agent.name]
        job = client.fine_tuning.jobs.retrieve(job_id)
        agent.model = job.fine_tuned_model
        print(f"{agent.name}'s fine-tuning completed. New model: {agent.model}")
        model_records.append({"agent": agent.name, "model": agent.model})

    with open(id_file, "a", encoding="utf-8") as f:
        for rec in id_records + model_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    args = parse_args()
    print(args)

    agents = [physicist, mathematician, summerizer]
    start = datetime.datetime.now()
    fine_tune(args, agents)
    end = datetime.datetime.now()

    print("Time cost:", round((end - start).total_seconds() / 60, 2), "mins")
