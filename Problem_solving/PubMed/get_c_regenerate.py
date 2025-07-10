
import os
import datetime
import json
import pandas as pd
from tqdm import tqdm


from Problem_solving.PubMed.args import parse_args
from libs.data_loader import load_jsonl_objects,extract_answer_letter,extract_answer_number,extract_response,extract_answer_yesno
from Problem_solving.PubMed.prompt import *
from libs.utils import compare_answer_with_groundtruth
from Problem_solving.PubMed.agent import Agent,agent_1,agent_2



def get_rephrase_response(agent,question,regenerate_response):
    user_prompt=rephrase_user_prompt.format(question=question,original_response=regenerate_response)
    rephrased_response = agent.call_agent(sys_prompt=rephrase_sys_prompt , user_prompt=user_prompt,temperature=0., max_tokens=4096, stop=None, n=1)
    return rephrased_response


def get_regenerate(args,agent_1,agent_2,recieved_agent):
   
    inputfile = ""
    regenerate_sol_file = ""
    regenerate_correct_file= ""
    regenerate_wrong_file = ""
    input_logs=load_jsonl_objects(inputfile)
    
    os.makedirs(os.path.dirname(regenerate_sol_file), exist_ok=True)
    os.makedirs(os.path.dirname(regenerate_wrong_file), exist_ok=True)
    os.makedirs(os.path.dirname(regenerate_correct_file), exist_ok=True)

    sol_logs = []
    wrong_logs=[]
    correct=0
    
    with open(regenerate_sol_file, 'a', encoding='utf-8') as f_sol, \
        open(regenerate_wrong_file, 'a', encoding='utf-8') as f_wrong,\
        open(regenerate_correct_file, 'a', encoding='utf-8') as f_correct:

        sol_logs=load_jsonl_objects(regenerate_sol_file)
        wrong_logs=load_jsonl_objects(regenerate_wrong_file)
        correct_logs=load_jsonl_objects(regenerate_correct_file)
        correct=len(correct_logs)
        for input_log in input_logs:
            if any(log['index'] ==  input_log['index'] for log in sol_logs):
                print(f"problem { input_log['index']} already got regenerate")
                continue
            else:
                print( input_log['index'])
                # task=input_log['task']
                if args.prompt_type == 'multi_agent'  :
                    if args.subject == 'PubMedQA':
                        extract_answer=extract_answer_yesno
                        format_prompt=format_prompt_yesno
                        context=input_log['context']
                        sys_sol_prompt=sys_regenerate_prompt_pubmed
                        if recieved_agent=='agent_1':
                            agent_1_regenerate_prompt=context_analyst_regenerate_prompt_pubmed
                            agent_2_regenerate_prompt=problem_solver_sol_prompt_pubmed
                            
                        elif recieved_agent=='agent_2':
                            agent_1_regenerate_prompt=context_analyst_sol_prompt_pubmed
                            agent_2_regenerate_prompt=problem_solver_regenerate_prompt_pubmed
                            
                    
                info= input_log
                question= input_log['question']
                
                groundtruth= input_log['groundtruth']
                sys_regenerate_prompt=sys_regenerate_prompt_pubmed
                if recieved_agent=='agent_1':
                    agent_1_feedback= input_log['feedback']['agent_1']
                    agent_1_original_response= input_log['agent_1_log']['messages'][2]['content']
                    agent_1_before_rephrase = None
                    while not agent_1_before_rephrase:
                        agent_1_regenerate_user_prompt = agent_1_regenerate_prompt.format(context=context, agent_1_original_response=agent_1_original_response, agent_1_feedback=agent_1_feedback)
                        agent_1_regenerate_log = agent_1.call_agent(sys_prompt=sys_regenerate_prompt ,user_prompt=agent_1_regenerate_user_prompt, temperature=0., max_tokens=4096, stop=None, n=1)
                        agent_1_before_rephrase = extract_response(agent_1_regenerate_log['messages'][2]['content'])
                        if not agent_1_before_rephrase:
                            print("agent_1_before_rephrase is None, regenerating...")
                
                    agent_1_rephrased_log = get_rephrase_response(agent_1,question,agent_1_before_rephrase)
                    agent_1_rephrased_response=agent_1_rephrased_log['messages'][2]['content']
                    agent_1_regenerate_response = agent_1_rephrased_response
                    info['re_agent_1_log_regenerate_raw']=agent_1_regenerate_log
                    info['re_agent_1_rephrased_log_raw']=agent_1_rephrased_log
                    
                    info['re_agent_1_log']= input_log['agent_1_log']
                    info['re_agent_1_log']['messages'][2]['content']= agent_1_rephrased_response
                    

            
                    agent_2_regenerate_user_prompt = agent_2_regenerate_prompt.format(question=question, agent_1_response=agent_1_regenerate_response,format_prompt=format_prompt)

                    agent_2_regenerate_log = agent_2.call_agent(sys_prompt=sys_regenerate_prompt,user_prompt=agent_2_regenerate_user_prompt, temperature=0., max_tokens=4096, stop=None, n=1)
                    agent_2_regenerate_response = agent_2_regenerate_log['messages'][2]['content']
                    info['re_agent_2_log']=agent_2_regenerate_log
                       

                elif recieved_agent=='agent_2':
                        agent_1_regenerate_response=input_log['re_agent_1_log']['messages'][2]['content']
                        agent_2_feedback= input_log['feedback']['agent_2']
                        agent_2_original_response= input_log['agent_2_log']['messages'][2]['content']
                        agent_2_regenerate_user_prompt = agent_2_regenerate_prompt.format(question=question, agent_2_original_response=agent_2_original_response, agent_2_feedback=agent_2_feedback)
                        agent_2_regenerate_log = agent_2.call_agent(sys_prompt=sys_regenerate_prompt,user_prompt=agent_2_regenerate_user_prompt, temperature=0., max_tokens=4096, stop=None, n=1)

                        agent_2_before_rephrase = extract_response(agent_2_regenerate_log['messages'][2]['content'])
                        if not agent_2_before_rephrase:
                            print("agent_2_before_rephrase is None, regenerating...")
                        agent_2_rephrased_log = get_rephrase_response(agent_2,question,agent_2_regenerate_response)
                        agent_2_rephrased_response=agent_2_rephrased_log['messages'][2]['content']
                        info['re_agent_2_log_regenerate_raw']=agent_2_regenerate_log
                        
                        info['re_agent_2_rephrased_log_raw']=agent_2_rephrased_log
                        info['re_agent_2_log']=input_log['agent_2_log']
                        info['re_agent_2_log']['messages'][2]['content']=agent_2_rephrased_response
                    
                            
                answer = extract_answer(agent_2_regenerate_response)
                
                info['re_answer']=answer
                
                
                
                    
                if isinstance(groundtruth, str):
                    groundtruth = [groundtruth]
                if compare_answer_with_groundtruth(answer, *groundtruth):
                    correct += 1
                    info['re_correct']=True

                    json_line = json.dumps(info,ensure_ascii=False)
                    f_sol.write(json_line + '\n')
                    f_correct.write(json_line + '\n')
                else:
                    print("wrong: ", input_log['index'])
                    info['re_correct']=False

                    json_line = json.dumps(info,ensure_ascii=False)
                    f_sol.write(json_line + '\n')
                    f_wrong.write(json_line + '\n')


    acc=correct/len( input_logs)
    print(f"correct: {correct}")

    return acc,correct



if __name__ == '__main__':
    args = parse_args()   
    print(args)
    print(agent_1.model)
    start_time = datetime.datetime.now()
    acc=get_regenerate(args,agent_1,agent_2,'agent_1')
    print(f"regenerate accuracy: {acc}")
    end_time = datetime.datetime.now()
    print('time cost:', ((end_time-start_time).total_seconds()/60,2), ' mins')
