import os
import json
import datetime

from args import parse_args
from prompt import *
from libs.data_loader import load_jsonl_objects
from Problem_solving.PhyChem.agent import critic_agent


def generate_critic_feedback(info, args, critic_agent, received_agent):
    question = info['question']
    agent_1_original_response = info['agent_1_log']['messages'][2]['content']
    agent_2_original_response = info['agent_2_log']['messages'][2]['content']
    agent_3_original_response = info['agent_3_log']['messages'][2]['content']
    correct_answer = info['groundtruth'][0]

    if args.subject == 'phy':
        critic_users_prompt_tmpl = critic_users_promt_phy
        critic_sys_prompt_tmpl = critic_sys_promt_one_phy
    elif args.subject == 'chem':
        critic_users_prompt_tmpl = critic_users_promt_chem
        critic_sys_prompt_tmpl = critic_sys_promt_one_chem
    else:
        raise ValueError(f"Unsupported subject")

    user_prompt = critic_users_prompt_tmpl.format(
        question=question,
        agent_1_original_response=agent_1_original_response,
        agent_2_original_response=agent_2_original_response,
        agent_3_original_response=agent_3_original_response,
        correct_answer=correct_answer
    )
    critic_sys_prompt = critic_sys_prompt_tmpl.format(agent_name=received_agent)

    feedback_log = critic_agent.call_agent(
        sys_prompt=critic_sys_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=4096,
        stop=None,
        n=1
    )

    feedback = info.copy()
    feedback['feedback'] = {}
    feedback['feedback'][received_agent] = feedback_log['messages'][2]['content']
    return feedback


def get_feedback(args, critic_agent, received_agent):
    print(f"ft_round: {args.ft_round} | sol_round: {args.sol_round} | fd_round: {args.fd_round} | re_round: {args.re_round}")

    inputfile = ""  
    feedback_file = "" 
    os.makedirs(os.path.dirname(feedback_file), exist_ok=True)

    input_logs = load_jsonl_objects(inputfile)
    feedback_logs = load_jsonl_objects(feedback_file) if os.path.exists(feedback_file) else []

    existing_indices = {log['index'] for log in feedback_logs}

    with open(feedback_file, 'a', encoding='utf-8') as f_feedback:
        for info in input_logs:
            if info['index'] in existing_indices:
                print(f"problem {info['index']} already got feedback")
                continue

            print(info['index'])
            new_feedback = generate_critic_feedback(info, args, critic_agent, received_agent)
            f_feedback.write(json.dumps(new_feedback, ensure_ascii=False) + '\n')

    return feedback_file


if __name__ == '__main__':
    args = parse_args()
    print(args)

    start_time = datetime.datetime.now()
    feedback_file = get_feedback(args, critic_agent, received_agent="agent_3")
    end_time = datetime.datetime.now()

    print(f"Feedback written to: {feedback_file}")
    print(f"Time cost: {round((end_time - start_time).total_seconds() / 60, 2)} mins")
