
import json
import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from libs.data_loader import load_jsonl_objects
from args import parse_args
from prompt import sum_regenerate_prompt_phy,format_prompt_number,format_prompt_letter
args = parse_args()   
print(args)
# read
ditc="Problem_solving/PhyChem/logs/solve_phy_gpt-3.5-turbo"
sft_dic="Problem_solving/PhyChem/logs/solve_phy_gpt-3.5-turbo"
math_file = os.path.join(ditc, "finetune_math.jsonl")
phy_file = os.path.join(ditc,  "finetune_phy.jsonl")
sum_file = os.path.join(ditc,  "finetune_sum.jsonl")

for root, dirs, files in os.walk(ditc):
    # print(files )
   

    with open(phy_file, 'a') as phy_f, \
    open(math_file, 'a') as math_f, \
    open(sum_file, 'a') as sum_f:
        for file in files:
            if file=="correct.jsonl":
                print(file)
                correct_file = os.path.join(root, file)
                print(correct_file)
                # write
                correct_data = load_jsonl_objects(correct_file)
                for i, item in enumerate(correct_data):
                    # print(item)
                    phy_f.write(json.dumps(item['agent_1_log']) + '\n')
                    math_f.write(json.dumps(item['agent_2_log']) + '\n')
                    sum_f.write(json.dumps(item['agent_3_log']) + '\n')
            elif file.endswith("regenerate_correct.jsonl") or file.endswith("regenerate_correct2.jsonl"):
                print(file)
                correct_file = os.path.join(root, file)
                # write
                correct_data = load_jsonl_objects(correct_file)
                for i, item in enumerate(correct_data):
                    task=item['task']
                    if  'MMLU' in task or 'gpqa_diamond' in task:
                        format_prompt=format_prompt_letter
                        # extract_answer= extract_answer_letter
                        
                    elif  'theoremqa' in task or  'scibench' in task:
                        format_prompt=format_prompt_number
                        # extract_answer=extract_answer_number 
                    agent1=item['agent_1_log']
                    agent1['messages'][2]['content']=item['re_agent_1_log']['messages'][2]['content']
                    agent3=item['agent_3_log']
                    question=item['question']
                    agent_1_regenerate_response=item['re_agent_1_log']['messages'][2]['content']
                    agent_2_regenerate_response=item['re_agent_2_log']['messages'][2]['content']
                    agent3['messages'][1]['content']=sum_regenerate_prompt_phy.format(
                            question=question, agent_1_regenerate_response=agent_1_regenerate_response, agent_2_regenerate_response=agent_2_regenerate_response,format_prompt=format_prompt)
                    agent3['messages'][2]['content']=item['re_agent_3_log']['messages'][2]['content']
                    phy_f.write(json.dumps(agent1) + '\n')
                    math_f.write(json.dumps(item['re_agent_2_log']) + '\n')
                    sum_f.write(json.dumps(agent3) + '\n')

                


