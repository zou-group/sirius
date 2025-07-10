import math
import random
from openai import OpenAI
import openai
import os
import json
import time
import datetime

from args import parse_args
from Problem_solving.PhyChem.agent import Agent, summerizer,mathematician,physicist


client=OpenAI()



def fine_tune(args,agents):

    id_list={}
    for agent in agents:
       
        file_path=f'...'
        print(file_path)
        fine_tuned_object = agent.fine_tune(file_path)
        id=fine_tuned_object.id
        id_list[agent]=id
        print(f"{agent.name}'s fine-tuning job id: {id}")
        id_file=f'...'
        os.makedirs(os.path.dirname(id_file), exist_ok=True)
        with open(id_file, 'a') as id_file:
            id_file.write(json.dumps({"agent": agent.name, "id": id}) + '\n')
            
    all_completed = False
    while not all_completed:
        all_completed = True  
        for agent, id in id_list.items():
            fine_tuned_object=client.fine_tuning.jobs.retrieve(id)
            if fine_tuned_object.fine_tuned_model is None:
                all_completed = False  
                print(f"{agent.name}'s fine-tuning is still in progress. Waiting for completion...")     
        if not all_completed:
            time.sleep(60)  

    for agent, id in id_list.items():
        fine_tuned_object=client.fine_tuning.jobs.retrieve(id)
        print(fine_tuned_object.fine_tune_model)
        agent.model = fine_tuned_object.fine_tuned_model
        with open(id_file, 'a') as id_file:
            id_file.write(json.dumps({"agent": agent.name, "model": fine_tuned_object.fine_tuned_model}) + '\n')
        print(f"{agent.name}'s fine-tuning completed. Fine-tuned model: {agent.model}")


if __name__ == '__main__':

    args = parse_args()   
    print(args)
    agents = [physicist, mathematician, summerizer]  
    start_time = datetime.datetime.now()
    fine_tune(args,agents)
    end_time = datetime.datetime.now()
    print('time cost:', round((end_time-start_time).total_seconds()/60,2), ' mins')
