from openai import OpenAI
import openai
import os
import datetime
import json
import pandas as pd
from tqdm import tqdm

from args import parse_args
from libs.data_loader import load_dataset,load_jsonl_objects,extract_answer_yesno
from prompt import *
from libs.utils import compare_answer_with_groundtruth
from Problem_solving.PubMed.agent import Agent, agent_1,agent_2
import  multiprocessing

DATA_BATCH_SIZE=1

def get_solve(prompt_type,temperature,max_tokens,rank, batched_input_data,log_dir,correct_count):
    correct=0
    
    for item in tqdm(batched_input_data, desc=str(rank), position=rank):
        
        item=item[0]
        question=item['question']
        groundtruth=item['groundtruth']
        index=item['index']
        print(index)
    
        sol_path = f'{log_dir}/sol/{index}_sol.jsonl'
        correct_path = f'{log_dir}/correct/{index}_correct.jsonl'
        wrong_path = f'{log_dir}/wrong/{index}_wrong.jsonl'
        sol_dict=f'{log_dir}/sol/'
        correct_dict= f'{log_dir}/correct/'
        wrong_dict= f'{log_dir}/wrong/'
        

        os.makedirs(os.path.dirname(sol_dict), exist_ok=True)
        os.makedirs(os.path.dirname(correct_dict), exist_ok=True)
        os.makedirs(os.path.dirname(wrong_dict), exist_ok=True)
        

        if os.path.exists( sol_path) :
            print(f"Problems {index} exist")       
            continue

        else:
            if prompt_type == 'multi_agent'  :
            
                extract_answer=extract_answer_yesno
                format_prompt=format_prompt_yesno
                context=item['context']
                sys_sol_prompt=sys_sol_prompt_pubmed

                agent_1_sol_prompt=context_analyst_sol_prompt_pubmed
                agent_2_sol_prompt=problem_solver_sol_prompt_pubmed 

                agent_1_user_prompt=agent_1_sol_prompt.format(context=context)
                agent_1_log = agent_1.call_agent(sys_prompt=sys_sol_prompt, user_prompt=agent_1_user_prompt, temperature=temperature, max_tokens=max_tokens)
                agent_1_response=agent_1_log['messages'][2]['content']
                
                agent_2_user_prompt=agent_2_sol_prompt.format(question=question, agent_1_response=agent_1_response,format_prompt=format_prompt)
                agent_2_log = agent_2.call_agent(sys_prompt=sys_sol_prompt, user_prompt=agent_2_user_prompt, temperature=temperature, max_tokens=max_tokens)
                agent_2_response=agent_2_log['messages'][2]['content']

                answer= extract_answer(agent_2_response[-15:])
                               
                if isinstance(groundtruth, str):
                    groundtruth = [groundtruth]
                if compare_answer_with_groundtruth(answer, *groundtruth):
                    correct += 1
                    info=item
                    info['score']=True
                    info['answer']=answer
                    info['agent_1_log']=agent_1_log
                    info['agent_2_log']=agent_2_log

                    json_line = json.dumps(info,ensure_ascii=False)
                    with open(sol_path, 'a', encoding='utf-8') as f_sol:
                        f_sol.write(json_line + '\n')
                    with open(correct_path, 'a', encoding='utf-8') as f_correct:
                        f_correct.write(json_line + '\n')
                else:
                    print("wrong: ",index)
                    info=item
                    info['score']=False
                    info['answer']=answer
                    info['agent_1_log']=agent_1_log
                    info['agent_2_log']=agent_2_log

                    json_line = json.dumps(info,ensure_ascii=False)
                    with open(sol_path, 'a', encoding='utf-8') as f_sol:
                        f_sol.write(json_line + '\n')
                    with open(wrong_path, 'a', encoding='utf-8') as f_wrong:
                        f_wrong.write(json_line + '\n')


    correct_count[rank] = correct




if __name__ == '__main__':
    args = parse_args()  

    print(args)
    start_time = datetime.datetime.now()
    print(f"ft_round:{args.ft_round} sol_round:{args.sol_round} fd_round:{args.fd_round} re_round:{args.re_round}")

    sol_logs = []
    wrong_logs=[]
    correct=0
    if args.sol_round ==0:
        input_datas = load_dataset(args)
    num_processes = 64
    processes = []
    batched_dataset = [input_datas[i : i + DATA_BATCH_SIZE] for i in range(0, len(input_datas), DATA_BATCH_SIZE)] 
    if args.prompt_type == 'multi_agent':
        
        log_dir = f'Problem_solving/PubMed/logs/{args.prompt_type}/{agent_1.model}_{agent_2.model}'
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    print(log_dir)
    manager = multiprocessing.Manager()
    correct_count = manager.dict()  # 
    prompt_type=args.prompt_type
    max_tokens=args.max_tokens
    temperature=args.temperature
    print(log_dir)
    print(len(batched_dataset))
    print( prompt_type)

    for i in range(num_processes):
        p = multiprocessing.Process(target=get_solve, args=(prompt_type,temperature,max_tokens,i, batched_dataset[i :: num_processes],log_dir,correct_count))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    total_correct = sum(correct_count.values())
    end_time = datetime.datetime.now()
    print(f"accuracy: {total_correct }")
    print('time cost:', round((end_time - start_time).total_seconds() / 60, 2), ' mins')