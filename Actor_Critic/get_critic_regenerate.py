import os
import datetime
import json
from tqdm import tqdm
import multiprocessing

from args import parse_args
from libs.data_loader import load_jsonl_objects, extract_answer_yesno, extract_response
from libs.utils import compare_answer_with_groundtruth
from prompt import *
from Actor_Critic.agent import actor_agent

DATA_BATCH_SIZE = 1


def get_rephrase_response(agent, question, regenerate_response):
    user_prompt = rephrase_user_prompt.format(question=question, original_response=regenerate_response)
    return agent.call_agent(sys_prompt=rephrase_sys_prompt, user_prompt=user_prompt,
                            temperature=0.0, max_tokens=4096, stop=None, n=1)


def get_regenerate(batched_input_data, rank, log_dir, correct_count):
    correct = 0

    os.makedirs(f'{log_dir}/sol_re/', exist_ok=True)
    os.makedirs(f'{log_dir}/correct_re/', exist_ok=True)
    os.makedirs(f'{log_dir}/wrong_re/', exist_ok=True)

    for item in tqdm(batched_input_data, desc=str(rank), position=rank):
        item = item[0]
        index = item['index']
        question = item['question']
        context = item['context']
        groundtruth = item['groundtruth']
        feedback = item['feedback']
        original_response = item['single_log']['messages'][2]['content']

        sol_path = f'{log_dir}/sol_re/{index}_sol.jsonl'
        correct_path = f'{log_dir}/correct_re/{index}_correct.jsonl'
        wrong_path = f'{log_dir}/wrong_re/{index}_wrong.jsonl'

        if os.path.exists(sol_path):
            print(f"Problem {index} exists, skipping.")
            continue

        print(f"Processing {index}")
        extract_answer = extract_answer_yesno
        format_prompt = format_prompt_yesno

        regenerate_response = None
        while not regenerate_response:
            regenerate_user_prompt = user_single_regenerate_prompt.format(
                question=question, context=context,
                original_response=original_response, feedback=feedback,
                format_prompt=format_prompt
            )
            regenerate_log = actor_agent.call_agent(
                sys_prompt=sys_single_regenerate_prompt,
                user_prompt=regenerate_user_prompt,
                temperature=0.0,
                max_tokens=4096,
                stop=None,
                n=1
            )
            regenerate_response = extract_response(regenerate_log['messages'][2]['content'])
            if not regenerate_response:
                print("Regenerated response is None, retrying...")

        rephrased_log = get_rephrase_response(actor_agent, question, regenerate_response)
        rephrased_response = rephrased_log['messages'][2]['content']

        # Record
        info = item.copy()
        info['re_log_regenerate_raw'] = regenerate_log
        info['re_rephrased_log'] = rephrased_log
        info['re_log'] = item['single_log'].copy()
        info['re_log']['messages'][2]['content'] = rephrased_response
        info['re_answer'] = extract_answer(rephrased_response)

        if isinstance(groundtruth, str):
            groundtruth = [groundtruth]

        if compare_answer_with_groundtruth(info['re_answer'], *groundtruth):
            info['re_correct'] = True
            correct += 1
            with open(sol_path, 'a', encoding='utf-8') as f_sol, \
                 open(correct_path, 'a', encoding='utf-8') as f_correct:
                f_sol.write(json.dumps(info, ensure_ascii=False) + '\n')
                f_correct.write(json.dumps(info, ensure_ascii=False) + '\n')
        else:
            info['re_correct'] = False
            print(f"Wrong: {index}")
            with open(sol_path, 'a', encoding='utf-8') as f_sol, \
                 open(wrong_path, 'a', encoding='utf-8') as f_wrong:
                f_sol.write(json.dumps(info, ensure_ascii=False) + '\n')
                f_wrong.write(json.dumps(info, ensure_ascii=False) + '\n')

    correct_count[rank] = correct


if __name__ == '__main__':
    args = parse_args()
    print(args)

    inputfile = f""
    print("Input file:", inputfile)

    log_dir = inputfile.replace("feedback.jsonl", f"regenerate-{actor_agent.model}")
    os.makedirs(log_dir, exist_ok=True)
    print("Output dir:", log_dir)

    input_datas = load_jsonl_objects(inputfile)
    batched_dataset = [input_datas[i: i + DATA_BATCH_SIZE] for i in range(0, len(input_datas), DATA_BATCH_SIZE)]

    manager = multiprocessing.Manager()
    correct_count = manager.dict()

    print("Batch size:", len(batched_dataset))
    print("Using actor agent:", actor_agent.model)

    start_time = datetime.datetime.now()
    processes = []
    for i in range(min(32, len(batched_dataset))):
        p = multiprocessing.Process(
            target=get_regenerate,
            args=(batched_dataset[i::32], i, log_dir, correct_count)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_correct = sum(correct_count.values())
    print("Total correct:", total_correct)
    end_time = datetime.datetime.now()
    print('Time cost:', round((end_time - start_time).total_seconds() / 60, 2), 'mins')
