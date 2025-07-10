import math
import random
from openai import OpenAI
import openai
import os
import json
import time
import datetime
from agent import Agent
client=OpenAI()

api_type=OpenAI()
model = "gpt-3.5-turbo"

solver = Agent(
    name="solver", 
    model=model,
    next_agent=None,
    pre_agent=None ,
    api_type=api_type
)

analyst = Agent(
    name="analyst", 
    model=model,
    next_agent=solver,
    pre_agent=None,
    api_type=api_type
)

agents=[analyst,solver]

def fine_tune(model):
    id_list={}
    for agent in agents:
        role=agent.name 
        file_path=f'...'
        print(file_path)
        fine_tuned_object = agent.fine_tune(file_path)
        id=fine_tuned_object.id
        print(f"{model}'s fine-tuning job id: {id}")
        id_list[agent]=id
        id_file=f'...'
        os.makedirs(os.path.dirname(id_file), exist_ok=True)
        with open(id_file, 'a') as id_file:
            id_file.write(json.dumps({"agent": f"{role}", "id": id}) + '\n')

                
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
    
    start_time = datetime.datetime.now()
    
    fine_tune(model)
    end_time = datetime.datetime.now()
    print('time cost:', round((end_time-start_time).total_seconds()/60,2), ' mins')
