import os
import json
import datetime
from tqdm import tqdm
import multiprocessing

from args import parse_args
from libs.data_loader import load_dataset, extract_answer_yesno
from prompt import *
from libs.utils import compare_answer_with_groundtruth
from Problem_solving.PubMed.agent import agent_1, agent_2

DATA_BATCH_SIZE = 1

def get_solve( temperature, max_tokens, rank, batched_input_data, log_dir, correct_count):
    correct = 0

    os.makedirs(os.path.join(log_dir, 'sol'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'correct'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'wrong'), exist_ok=True)

    for batch in tqdm(batched_input_data, desc=str(rank), position=rank):
        item = batch[0]
        question = item['question']
        groundtruth = item['groundtruth']
        context = item['context']
        index = item['index']

        sol_path = os.path.join(log_dir, 'sol', f'{index}_sol.jsonl')
        correct_path = os.path.join(log_dir, 'correct', f'{index}_correct.jsonl')
        wrong_path = os.path.join(log_dir, 'wrong', f'{index}_wrong.jsonl')

        if os.path.exists(sol_path):
            print(f"Problem {index} already solved.")
            continue

        extract_answer = extract_answer_yesno
        format_prompt = format_prompt_yesno
        sys_prompt = sys_sol_prompt_pubmed

        agent_1_prompt = context_analyst_sol_prompt_pubmed.format(context=context)
        agent_1_log = agent_1.call_agent(sys_prompt=sys_prompt, user_prompt=agent_1_prompt, temperature=temperature, max_tokens=max_tokens)
        agent_1_response = agent_1_log['messages'][2]['content']

        agent_2_prompt = problem_solver_sol_prompt_pubmed.format(
            question=question,
            agent_1_response=agent_1_response,
            format_prompt=format_prompt
        )
        agent_2_log = agent_2.call_agent(sys_prompt=sys_prompt, user_prompt=agent_2_prompt, temperature=temperature, max_tokens=max_tokens)
        agent_2_response = agent_2_log['messages'][2]['content']

        answer= extract_answer(agent_2_response[-15:])

        info = item.copy()
        info['answer'] = answer
        info['agent_1_log'] = agent_1_log
        info['agent_2_log'] = agent_2_log
        info['score'] = compare_answer_with_groundtruth(answer, *([groundtruth] if isinstance(groundtruth, str) else groundtruth))

        # Write outputs
        json_line = json.dumps(info, ensure_ascii=False)
        with open(sol_path, 'w', encoding='utf-8') as f_sol:
            f_sol.write(json_line + '\n')

        if info['score']:
            correct += 1
            with open(correct_path, 'w', encoding='utf-8') as f_correct:
                f_correct.write(json_line + '\n')
        else:
            print(f"Wrong: {index}")
            with open(wrong_path, 'w', encoding='utf-8') as f_wrong:
                f_wrong.write(json_line + '\n')

    correct_count[rank] = correct


if __name__ == '__main__':
    args = parse_args()
    print(args)

    start_time = datetime.datetime.now()

    if args.sol_round == 0:
        input_data = load_dataset(args)
    else:
        raise NotImplementedError("Only sol_round == 0 is currently supported.")

    batched_data = [input_data[i:i + DATA_BATCH_SIZE] for i in range(0, len(input_data), DATA_BATCH_SIZE)]
    log_dir = f'Problem_solving/PubMed/logs/{args.prompt_type}/{agent_1.model}_{agent_2.model}'
    os.makedirs(log_dir, exist_ok=True)

    manager = multiprocessing.Manager()
    correct_count = manager.dict()

    processes = []
    for i in range(min(64, len(batched_data))):
        p = multiprocessing.Process(
            target=get_solve,
            args=(args.prompt_type, args.temperature, args.max_tokens, i, batched_data[i::64], log_dir, correct_count)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_correct = sum(correct_count.values())
    end_time = datetime.datetime.now()

    print(f"Total correct: {total_correct}")
    print(f"Time cost: {round((end_time - start_time).total_seconds() / 60, 2)} mins")
