# from openai import OpenAI
import re
import os

import time
import datetime
import json
import pandas as pd
from Problem_solving.PubMed.args import parse_args
from Problem_solving.PubMed.prompt import *
from libs.data_loader import load_jsonl_objects
from Problem_solving.PubMed.agent import Agent, critic_agent

    
def get_feedback(args,critic_agent,received_agent): 
    inputfile = ""
    feedback_file= ""
    os.makedirs(os.path.dirname(feedback_file), exist_ok=True)

    with open(feedback_file, 'a', encoding='utf-8') as f_feedback:   
        input_logs=load_jsonl_objects(inputfile)
        feedback_logs=load_jsonl_objects(feedback_file)
        for info in input_logs:
            if any(log['index'] == info['index'] for log in feedback_logs):
                print(f"problem {info['index']} already got feedback")
                continue
            else:
                print(info['index'])
                new_feedback=generate_critic_feedback(info,args,critic_agent,received_agent)
                f_feedback.write(json.dumps(new_feedback) + '\n')
    return feedback_file





def generate_critic_feedback(info,args,critic_agent,received_agent):
    question=info['question']
    context=info['context']
    agent_2_original_response=info['agent_2_log']['messages'][2]['content']
    agent_1_original_response=info['agent_1_log']['messages'][2]['content']
    correct_answer=info['groundtruth'][0]
    if args.subject == 'PubMedQA':
        critic_users_promt=critic_users_promt_pubmed
        critic_sys_promt_one=critic_sys_promt_one_pubmed

    user_prompt=critic_users_promt.format(question=question,context=context,agent_1_original_response=agent_1_original_response,agent_2_original_response=agent_2_original_response,correct_answer=correct_answer)
    feedback = {}
    feedback = info.copy()
    feedback['feedback']={}
    critic_sys_promt=critic_sys_promt_one.format(agent_name=received_agent)
    feedback_agent_name=critic_agent.call_agent(sys_prompt=critic_sys_promt, user_prompt=user_prompt,temperature=0., max_tokens=4096, stop=None, n=1)
    feedback['feedback'][f'{received_agent}']=feedback_agent_name['messages'][2]['content']
    return feedback


if __name__ == '__main__':
    args = parse_args()  
    print(args)
    start_time = datetime.datetime.now()
    feedback_file=get_feedback(args,critic_agent,received_agent="")
    end_time = datetime.datetime.now()
    print('time cost:', round((end_time-start_time).total_seconds()/60,2), ' mins')
