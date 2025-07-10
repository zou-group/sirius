import os
import json
import datetime
from tqdm import tqdm

from args import parse_args
from libs.data_loader import load_jsonl_objects, extract_answer_letter, extract_answer_number, extract_response
from prompt import *
from libs.utils import compare_answer_with_groundtruth
from Problem_solving.PhyChem.agent import physicist, mathematician, summerizer


def get_rephrase_response(agent, question, regenerate_response):
    user_prompt = rephrase_user_prompt.format(question=question, original_response=regenerate_response)
    return agent.call_agent(sys_prompt=rephrase_sys_prompt, user_prompt=user_prompt,
                            temperature=0.0, max_tokens=4096, stop=None, n=1)


def get_regenerate(args, agent_1, agent_2, agent_3, received_agent='agent_1'):
    inputfile = "" 
    regenerate_sol_file = f'logs/{args.subject}/{args.re_round}_regenerate_sol.jsonl'
    regenerate_correct_file = f'logs/{args.subject}/{args.re_round}_regenerate_correct.jsonl'
    regenerate_wrong_file = f'logs/{args.subject}/{args.re_round}_regenerate_wrong.jsonl'

    os.makedirs(f"logs/{args.subject}/", exist_ok=True)

    input_logs = load_jsonl_objects(inputfile)
    sol_logs = load_jsonl_objects(regenerate_sol_file) if os.path.exists(regenerate_sol_file) else []
    existing_indices = {log['index'] for log in sol_logs}

    correct = 0

    with open(regenerate_sol_file, 'a', encoding='utf-8') as f_sol, \
         open(regenerate_correct_file, 'a', encoding='utf-8') as f_correct, \
         open(regenerate_wrong_file, 'a', encoding='utf-8') as f_wrong:

        for input_log in tqdm(input_logs):
            if input_log['index'] in existing_indices:
                print(f"Problem {input_log['index']} already regenerated")
                continue

            print(input_log['index'])
            info = input_log.copy()
            question = info['question']
            groundtruth = info['groundtruth']
            task = info['task']

            if 'MMLU' in task or 'gpqa_diamond' in task:
                format_prompt = format_prompt_letter
                extract_answer = extract_answer_letter
            elif 'theoremqa' in task:
                format_prompt = format_prompt_number
                extract_answer = extract_answer_number

            if args.subject == 'phy':
                sys_regenerate_prompt = sys_regenerate_prompt_phy
                if received_agent == 'agent_1':
                    a1_prompt = phy_regenerate_prompt_phy
                    a2_prompt = math_sol_prompt_phy
                    a3_prompt = sum_regenerate_prompt_phy
                else:
                    a1_prompt = phy_sol_prompt_phy
                    a2_prompt = math_regenerate_prompt_phy
                    a3_prompt = sum_regenerate_prompt_phy
            elif args.subject == 'chem':
                sys_regenerate_prompt = sys_regenerate_prompt_chem
                if received_agent == 'agent_1':
                    a1_prompt = chem_regenerate_prompt_chem
                    a2_prompt = math_sol_prompt_chem
                    a3_prompt = sum_regenerate_prompt_chem
                else:
                    a1_prompt = chem_sol_prompt_chem
                    a2_prompt = math_regenerate_prompt_chem
                    a3_prompt = sum_regenerate_prompt_chem

            if received_agent == 'agent_1':
                a1_feedback = info['feedback']['agent_1']
                a1_orig = info['agent_1_log']['messages'][2]['content']
                a1_before = None
                while not a1_before:
                    user_prompt = a1_prompt.format(question=question, agent_1_original_response=a1_orig, agent_1_feedback=a1_feedback)
                    a1_log = agent_1.call_agent(sys_prompt=sys_regenerate_prompt, user_prompt=user_prompt,
                                                temperature=0.0, max_tokens=4096, stop=None, n=1)
                    a1_before = extract_response(a1_log['messages'][2]['content'])
                    if not a1_before:
                        print("agent_1_before_rephrase is None, regenerating...")

                a1_rephrased_log = get_rephrase_response(agent_1, question, a1_before)
                a1_response = a1_rephrased_log['messages'][2]['content']
                info['re_agent_1_log_regenerate_raw'] = a1_log
                info['re_agent_1_before_rephrased'] = a1_response
                info['re_agent_1_rephrased_log'] = a1_rephrased_log
                info['re_agent_1_log'] = a1_log
                info['re_agent_1_log']['messages'][2]['content'] = a1_response

                a2_user_prompt = a2_prompt.format(question=question, agent_1_response=a1_response)
                a2_log = agent_2.call_agent(sys_prompt=sys_regenerate_prompt, user_prompt=a2_user_prompt,
                                            temperature=0.0, max_tokens=4096, stop=None, n=1)
                a2_response = a2_log['messages'][2]['content']
                info['re_agent_2_log'] = a2_log

            elif received_agent == 'agent_2':
                a1_response = info['re_agent_1_log']['messages'][2]['content']
                a2_feedback = info['feedback']['agent_2']
                a2_orig = info['agent_2_log']['messages'][2]['content']

                a2_user_prompt = a2_prompt.format(question=question, agent_2_original_response=a2_orig, agent_2_feedback=a2_feedback)
                a2_log = agent_2.call_agent(sys_prompt=sys_regenerate_prompt, user_prompt=a2_user_prompt,
                                            temperature=0.0, max_tokens=4096, stop=None, n=1)
                a2_before = extract_response(a2_log['messages'][2]['content'])
                if not a2_before:
                    print("agent_2_before_rephrase is None, regenerating...")
                a2_rephrased_log = get_rephrase_response(agent_2, question, a2_before)
                a2_response = a2_rephrased_log['messages'][2]['content']

                info['re_agent_2_log_regenerate_raw'] = a2_log
                info['re_agent_2_before_rephrased'] = a2_response
                info['re_agent_2_rephrased_log'] = a2_rephrased_log
                info['re_agent_2_log'] = a2_log
                info['re_agent_2_log']['messages'][2]['content'] = a2_response

            a3_user_prompt = a3_prompt.format(
                question=question,
                agent_1_regenerate_response=a1_response,
                agent_2_regenerate_response=a2_response,
                format_prompt=format_prompt
            )
            a3_log = agent_3.call_agent(sys_prompt=sys_regenerate_prompt, user_prompt=a3_user_prompt,
                                        temperature=0.0, max_tokens=4096, stop=None, n=1)
            a3_response = a3_log['messages'][2]['content']
            info['re_agent_3_log'] = a3_log
            answer = extract_answer(a3_response)
            info['re_answer'] = answer

            if isinstance(groundtruth, str):
                groundtruth = [groundtruth]
            is_correct = compare_answer_with_groundtruth(answer, *groundtruth)
            info['re_correct'] = is_correct

            f_sol.write(json.dumps(info, ensure_ascii=False) + '\n')
            if is_correct:
                f_correct.write(json.dumps(info, ensure_ascii=False) + '\n')
                correct += 1
            else:
                print("wrong:", info['index'])
                f_wrong.write(json.dumps(info, ensure_ascii=False) + '\n')

    acc = correct / len(input_logs) if input_logs else 0.0
    print(f"correct: {correct}")
    return acc, correct


if __name__ == '__main__':
    args = parse_args()
    print(args)
    start = datetime.datetime.now()

    acc, correct = get_regenerate(args, physicist, mathematician, summerizer)
    print(f"regenerate accuracy: {acc:.3f}")
    end = datetime.datetime.now()
    print('time cost:', round((end - start).total_seconds() / 60, 2), 'mins')
