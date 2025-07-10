import os
import datetime
import json
import re
import multiprocessing
from tqdm import tqdm

from args import parse_args
from libs.data_loader import load_jsonl_objects
from prompt_critic import sys_critic_prompt, user_critic_prompt
from Actor_Critic.agent import judge_agent

DATA_BATCH_SIZE = 1


def extract_answer_true_false(input_string):
    ANSWER_PATTERN_YESNO = r"(?i)(Decision|Opinion)\s*:\s*(True|False|true|false)"
    match = re.search(ANSWER_PATTERN_YESNO, input_string)
    extracted_answer = match.group(2) if match else input_string
    return extracted_answer


def get_judgement(batched_input_data, rank, log_dir, correct_count, pt_count, pf_count, nf_count, nt_count):
    pt = pf = nf = nt = 0

    os.makedirs(f'{log_dir}/ALL/', exist_ok=True)
    os.makedirs(f'{log_dir}/True/', exist_ok=True)
    os.makedirs(f'{log_dir}/False/', exist_ok=True)

    for item in tqdm(batched_input_data, desc=str(rank), position=rank):
        item = item[0]
        index = item['index']
        print(index)

        question = item['question']
        original_response = item['single_log']['messages'][2]['content']
        context = item['context']
        score = item['score']

        all_path = f'{log_dir}/ALL/{index}_judgement.jsonl'
        true_path = f'{log_dir}/True/{index}_True.jsonl'
        false_path = f'{log_dir}/False/{index}_False.jsonl'

        if os.path.exists(all_path):
            print(f"Problems {index} exist")
            continue

        user_prompt = user_critic_prompt.format(question=question, context=context, original_response=original_response)
        info = item.copy()
        info['judgement'] = {}
        judgement_agent_log = judge_agent.call_agent(sys_prompt=sys_critic_prompt, user_prompt=user_prompt, temperature=0., max_tokens=4096, stop=None, n=1)
        judgement_agent_response = judgement_agent_log['messages'][2]['content']
        judgement = extract_answer_true_false(judgement_agent_response)
        info['judgement'] = judgement
        info['judgement_agent_log'] = judgement_agent_log

        if score is True and judgement == 'True':
            label = "PT"
            pt += 1
        elif score is True and judgement == 'False':
            label = "PF"
            pf += 1
        elif score is False and judgement == 'False':
            label = "NT"
            nt += 1
        elif score is False and judgement == 'True':
            label = "NF"
            nf += 1
        else:
            label = f"{judgement}_{score}"

        info['label'] = label

        with open(all_path, 'a', encoding='utf-8') as f_all:
            f_all.write(json.dumps(info, ensure_ascii=False) + '\n')

        if judgement == 'True':
            with open(true_path, 'a', encoding='utf-8') as f_true:
                f_true.write(json.dumps(info, ensure_ascii=False) + '\n')
        else:
            with open(false_path, 'a', encoding='utf-8') as f_false:
                f_false.write(json.dumps(info, ensure_ascii=False) + '\n')

    correct_count[rank] = pt + nt
    pt_count[rank] = pt
    pf_count[rank] = pf
    nf_count[rank] = nf
    nt_count[rank] = nt


if __name__ == '__main__':
    args = parse_args()
    print(args)
    print(judge_agent.model)

    inputfile = args.input_file
    log_dir = inputfile.replace('sol.jsonl', f'judgement-{judge_agent.model}/')
    os.makedirs(log_dir, exist_ok=True)

    input_datas = load_jsonl_objects(inputfile)
    batched_dataset = [input_datas[i: i + DATA_BATCH_SIZE] for i in range(0, len(input_datas), DATA_BATCH_SIZE)]

    num_processes = 32
    manager = multiprocessing.Manager()
    correct_count = manager.dict()
    pt_count = manager.dict()
    pf_count = manager.dict()
    nf_count = manager.dict()
    nt_count = manager.dict()

    print(len(batched_dataset))
    start_time = datetime.datetime.now()

    processes = []
    for i in range(num_processes):
        p = multiprocessing.Process(target=get_judgement,
                                    args=(batched_dataset[i::num_processes], i, log_dir, correct_count, pt_count, pf_count, nf_count, nt_count))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_correct = sum(correct_count.values())
    print("PT: ", sum(pt_count.values()))
    print("PF: ", sum(pf_count.values()))
    print("NT: ", sum(nt_count.values()))
    print("NF: ", sum(nf_count.values()))
    print("Total Predict correct: ", total_correct)

    end_time = datetime.datetime.now()
    print('time cost:', round((end_time - start_time).total_seconds() / 60, 2), 'mins')
