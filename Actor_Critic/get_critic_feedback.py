import os
import datetime
import json
import re
from tqdm import tqdm
import multiprocessing

from args import parse_args
from prompt import *
from prompt_critic import *
from libs.data_loader import load_jsonl_objects
from Actor_Critic.agent import Agent, critic_agent

DATA_BATCH_SIZE = 1


def get_feedback(batched_input_data, rank, log_dir):
    for item in tqdm(batched_input_data, desc=str(rank), position=rank):
        item = item[0]
        index = item['index']
        question = item['question']
        original_response = item['single_log']['messages'][2]['content']
        context = item['context']
        os.makedirs(f'{log_dir}/feedback', exist_ok=True)
        feedback_path = f'{log_dir}/feedback/{index}_feedback.jsonl'

        if os.path.exists(feedback_path):
            print(f"Problem {index} feedback exists")
            continue

        print(index)
        user_prompt = critic_users_prompt.format(
            question=question,
            context=context,
            original_response=original_response
        )

        feedback = item.copy()
        feedback['feedback'] = {}
        feedback_agent_log = critic_agent.call_agent(
            sys_prompt=critic_sys_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=4096,
            stop=None,
            n=1
        )
        feedback['feedback'] = feedback_agent_log['messages'][2]['content']
        feedback['feedback_agent_log'] = feedback_agent_log

        with open(feedback_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(feedback, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    args = parse_args()
    print(args)
    print(critic_agent.model)

    inputfile = args.input_file
    print(inputfile)

    log_dir = inputfile.replace('False.jsonl', f'feedback-{critic_agent.model}')
    os.makedirs(log_dir, exist_ok=True)
    num_processes =32
    input_datas = load_jsonl_objects(inputfile)
    batched_dataset = [
        input_datas[i: i + DATA_BATCH_SIZE]
        for i in range(0, len(input_datas), DATA_BATCH_SIZE)
    ]

    print(len(batched_dataset))
    start_time = datetime.datetime.now()

    processes = []
    for i in range(num_processes):
        p = multiprocessing.Process(
            target=get_feedback,
            args=(batched_dataset[i::32], i, log_dir)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end_time = datetime.datetime.now()
    print('Time cost:', round((end_time - start_time).total_seconds() / 60, 2), 'mins')
