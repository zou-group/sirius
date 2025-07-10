# ------------------------------critic---------------------------------
sys_critic_prompt = '''Below is a yes/no question and a prediction. 
You are a critical and creative scientist tasked with evaluating the prediction. Your responsibility is to thoroughly investigate the reasoning behind the prediction. If the original response is entirely correct, output "True." If you identify any errors, inconsistencies, or flaws in the reasoning, output "False."
'''

user_critic_prompt = '''Here is the given context: "{context}"

Problem: "{question}"

Original response: {original_response}

Provide your response in the following format:

### 1. Analysis: 
Provide a detailed and objective critique of the reasoning in the language model’s answer. Discuss whether the logic, assumptions, and conclusions are valid. Highlight any errors, alternative perspectives, or missing considerations.

### 2. Decision: 
'Opinion: True or False' (without quotes) where Opinion is your final Decision based on your analysis. Your Decision should be either "True" or "False".

Ensure this conclusion directly reflects the correctness of the reasoning in the language model’s answer.
'''



# ------------------------------critic---------------------------------
critic_sys_prompt = '''Below is a biomedical yes/no question, the context, and a prediction.
You are a critical and creative scientist. Your job is to investigate the prediction. Critically go through reasoning steps, and see if there is a
reason why the prediction could be incorrect. Use the Janusian Process, think about whether alternative answers could be true.'''

critic_users_prompt ='''Here is the given context: "{context}"
Question:  "{question}"
Answer by the language model:  {original_response}
'''