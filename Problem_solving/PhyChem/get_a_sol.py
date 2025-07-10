import os
import json
import datetime
import multiprocessing
from tqdm import tqdm

from args import parse_args
from libs.data_loader import load_dataset, extract_answer_letter, extract_answer_number, extract_answer_yesno
from libs.utils import compare_answer_with_groundtruth
from prompt import *
from Problem_solving.PhyChem.agent import summerizer, mathematician, physicist

# Agents
agent_1 = physicist
agent_2 = mathematician
agent_3 = summerizer

def get_solve(subject, temperature, max_tokens, rank, batch, log_dir, correct_count):
    correct = 0
    os.makedirs(f'{log_dir}/sol', exist_ok=True)
    os.makedirs(f'{log_dir}/correct', exist_ok=True)
    os.makedirs(f'{log_dir}/wrong', exist_ok=True)

    for item in tqdm(batch, desc=str(rank), position=rank):
        question = item['question']
        groundtruth = item['groundtruth']
        task = item['task']
        index = item['index']
        
        sol_path = f'{log_dir}/sol/{index}_sol.jsonl'
        correct_path = f'{log_dir}/correct/{index}_correct.jsonl'
        wrong_path = f'{log_dir}/wrong/{index}_wrong.jsonl'

        if os.path.exists(sol_path):
            print(f"Problem {index} exists, skipping.")
            continue

        # Choose formatting
        if 'MMLU' in task or 'gpqa_diamond' in task:
            format_prompt = format_prompt_letter
            extract_answer = extract_answer_letter
        elif 'theoremqa' in task:
            format_prompt = format_prompt_number
            extract_answer = extract_answer_number
        
        

        if subject == 'phy':
            sys_sol_prompt_ = sys_sol_prompt_phy
            a1_prompt = phy_sol_prompt_phy
            a2_prompt = math_sol_prompt_phy
            a3_prompt = sum_sol_prompt_phy
        elif subject == 'chem':
            sys_sol_prompt_ = sys_sol_prompt_chem
            a1_prompt = chem_sol_prompt_chem
            a2_prompt = math_sol_prompt_chem
            a3_prompt = sum_sol_prompt_chem
        else:
            raise ValueError(f"Unknown subject: {subject}")

        # Multi-agent solve
        a1_log = agent_1.call_agent(
            sys_prompt=sys_sol_prompt_,
            user_prompt=a1_prompt.format(question=question),
            temperature=temperature,
            max_tokens=max_tokens
        )
        a1_resp = a1_log['messages'][2]['content']

        a2_log = agent_2.call_agent(
            sys_prompt=sys_sol_prompt_,
            user_prompt=a2_prompt.format(question=question, agent_1_response=a1_resp),
            temperature=temperature,
            max_tokens=max_tokens
        )
        a2_resp = a2_log['messages'][2]['content']

        a3_log = agent_3.call_agent(
            sys_prompt=sys_sol_prompt_,
            user_prompt=a3_prompt.format(
                question=question,
                agent_1_response=a1_resp,
                agent_2_response=a2_resp,
                format_prompt=format_prompt
            ),
            temperature=temperature,
            max_tokens=max_tokens
        )
        a3_resp = a3_log['messages'][2]['content']
        answer = extract_answer(a3_resp[-15:])  # still fragile, consider better extraction

        if isinstance(groundtruth, str):
            groundtruth = [groundtruth]

        info = item.copy()
        info.update({
            'score': compare_answer_with_groundtruth(answer, *groundtruth),
            'answer': answer,
            'agent_1_log': a1_log,
            'agent_2_log': a2_log,
            'agent_3_log': a3_log
        })

        with open(sol_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(info, ensure_ascii=False) + '\n')

        out_path = correct_path if info['score'] else wrong_path
        if not info['score']:
            print("Wrong:", index)

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(info, ensure_ascii=False) + '\n')

        if info['score']:
            correct += 1

    correct_count[rank] = correct


if __name__ == '__main__':
    args = parse_args()
    print(args)

    start = datetime.datetime.now()
    input_data = load_dataset(args)

    num_processes = min(64, len(input_data))
    batches = [input_data[i::num_processes] for i in range(num_processes)]

   
    log_dir = f"logs/solve_{args.subject}_{args.model}"  
    os.makedirs(log_dir, exist_ok=True)

    manager = multiprocessing.Manager()
    correct_count = manager.dict()

    processes = []
    for i in range(num_processes):
        p = multiprocessing.Process(target=get_solve, args=(
            args.subject, args.temperature, args.max_tokens,
            i, batches[i], log_dir, correct_count))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_correct = sum(correct_count.values())
    elapsed = round((datetime.datetime.now() - start).total_seconds() / 60, 2)

    print(f"Total correct: {total_correct}")
    print(f"Time cost: {elapsed} mins")
