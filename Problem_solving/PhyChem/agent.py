import json
# import networkx as nx
from openai import OpenAI
import openai
import prompt as prompt

import time
import datetime
from args import parse_args



# -----------Initialize agents structure
class Agent:
    def __init__(self, api_type,name, model,next_agent,pre_agent):
        self.name = name  
        self.model = model  
        self.next_agent=next_agent
        self.pre_agent=pre_agent
        self.api_type=api_type
        self.client=api_type
        

    def call_agent(self, sys_prompt,user_prompt, temperature=0., max_tokens=5500, stop=None, n=1):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        client=self.client
        attempt = 0
        while attempt < 50:
            try:
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    stop=stop,
                    temperature=temperature,
                    n=n
                )
                assistant_message = {"role": "assistant", "content": completion.choices[0].message.content}
                messages.append(assistant_message)
                log = {
                    "messages": messages
                }
                return log 

            except openai.error.OpenAIError as e: 
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                attempt += 1
                if attempt < 50:
                    time.sleep(10)  
                else:
                    raise  

        return None 
    

    def fine_tune(self,training_dataset_filename):
        client=self.client

        if isinstance(self.api_type, OpenAI):
            file_object=client.files.create(
                file=open(training_dataset_filename, "rb"),
                purpose="fine-tune"
                )
            
            file_id=file_object.id
            print(file_id)
            fine_tuned_object=client.fine_tuning.jobs.create(
                training_file=file_id, 
                model=self.model
                )
    

        return fine_tuned_object




args = parse_args() 
api_type=OpenAI()
model=args.model

summerizer = Agent(
    name="sum",
    model=model,
    next_agent=None,
    pre_agent=None,
    api_type=api_type
)

mathematician = Agent(
    name="math", 
    model=model,
    next_agent=[summerizer],
    pre_agent=None ,
    api_type=api_type
)

physicist = Agent(
    name="phy", 
    model=model,
    next_agent=[mathematician],
    pre_agent=None,
    api_type=api_type
)

single_agent=Agent(
    name="single", 
    model=model,
    next_agent=None,
    pre_agent=None,
    api_type=api_type
)
        
critic_agent= Agent(
    name="ctrict", 
    model=model,
    next_agent=None,
    pre_agent=None, 
    api_type=api_type
)





