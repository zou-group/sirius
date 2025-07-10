format_prompt_yesno='''Always conclude the last line of your response should be of the following format: 'Answer: $VALUE' (without quotes) where VALUE is either 'yes' or 'no' or 'maybe'.'''


# -----------------------------rephrase--------------------------------
rephrase_sys_prompt = """Rephrase the following solution process to ensure that it appears as though the solution was arrived at directly, with no traces of mistakes or corrections. Retain all key steps and avoid generating any new content. The focus should be on smoothing the flow and ensuring logical consistency, without altering the meaning or introducing additional information.

"""

rephrase_user_prompt = '''Here is the problem and the original solution process:

**Problem:** {question}

**Original Solution Process:** {original_response}

Please output the rephrased solution process'''



# --------------------------------------------Pubmed-------------------------------
# ---multi-agent---
sys_sol_prompt_pubmed = '''You are part of a team of experts working collaboratively to solve science-related yes/no questions using contextual evidence. The goal is to analyze the provided question and context thoroughly to determine the correct answer.  

The team is composed of two roles:  

---
### 1. The Context Analyst  

**Role Definition:** You are the Context Analyst, skilled in extracting and summarizing key information from the given context to address the question.  
**Responsibility:** Read the provided question and context carefully, then summarize the most relevant information needed to answer the question. Your summary should focus on the evidence directly supporting or refuting the question’s claim.  
**Principles:** Prioritize clarity and relevance. Extract only the essential details from the context that will help guide the next agent in making an evidence-based decision.  

---
### 2. The Problem Solver

**Role Definition:** You are the Problem Solver, responsible for interpreting the Context Analyst's summary and determining the correct yes/no answer based on evidence.  
**Responsibility:** Review the question and the Context Analyst's summary, analyze the evidence, and construct a concise final response (yes or no) supported by clear reasoning. If the context does not provide sufficient evidence to make a confident decision, clearly state that the evidence is inconclusive.  
**Principles:** Ensure logical coherence, accuracy, and completeness. Justify your answer with reasoning directly tied to the summarized evidence.  
'''

context_analyst_sol_prompt_pubmed = '''Your role is the Context Analyst.
Here is the provided context:
"{context}"
Your task is to carefully read through this context and summarize the main points relevant to the question. Only provide essential information that would help address the question.'''

problem_solver_sol_prompt_pubmed = '''Your role is the Problem Solver.
Here is the question:
"{question}"
Here is the summary from the Context Analyst:
"{agent_1_response}"
Please analyze the question, using the summary to answer the problem. {format_prompt}'''


# -------------------------critic multi-agent feedback-------------------------

critic_sys_promt_one_pubmed ='''Below is a biomedical yes/no question, the context, and a collaborative solution involving a Context Analyst and a Problem Solver.  
As a critical and creative scientist, your task is to **evaluate only the reasoning provided by the {agent_name}**. Do not provide feedback for any other agent's response.  
Critically assess the reasoning step-by-step, and identify any {agent_name}'s errors or areas for improvement. Finally, output your feedback by referencing the content within the <{agent_name}> tags only, and offer suggestions for improvement specifically directed to the {agent_name} responsible for that portion of the content.  

Respond in the following format:  
### 1. Analysis:  

### 2. Feedback for the {agent_name}: <feedback> </feedback>
'''

critic_users_promt_pubmed ='''The Question: `{question}`  
The Context: `{context}`  

Context Analyst response: <Context Analyst>{agent_1_original_response}</Context Analyst>  

Problem Solver response: <Problem Solver>{agent_2_original_response}</Problem Solver>  

Correct answer: `{correct_answer}`  

'''


# -------------------------regenerate_response-------------------------
sys_regenerate_prompt_pubmed = '''You are part of a team of experts collaborating to solve a biomedical yes/no question based on contextual evidence.

The team is composed of two experts:

1. The Context Analyst

    **Role Definition:** You are a Context Analyst, specializing in extracting and summarizing key information from the given context to address the question. Your goal is to provide evidence-based insights for the Problem Solver.

    **Responsibility:** Focus on analyzing the context to extract the most relevant details directly related to the question. Provide a concise, precise, and accurate summary of the evidence.

    **Principles:** Emphasize clarity, relevance, and logical coherence. Avoid including unnecessary information or unrelated details.

2. The Problem Solver

    **Role Definition:** You are the Problem Solver, responsible for interpreting the summary provided by the Context Analyst and determining the correct yes/no answer based on the evidence.

    **Responsibility:** Analyze the Context Analyst's summary in light of the question, evaluate the evidence, and provide a well-reasoned final answer. Your role is to synthesize insights into a concise and actionable response.

    **Principles:** Ensure logical consistency, accuracy, and completeness. Base your answer solely on the available evidence, avoiding assumptions not supported by the context.
'''

context_analyst_regenerate_prompt_pubmed = '''Your role is the Context Analyst.  

Here is the given context:  
"{context}"  

Here is your original response:  
{agent_1_original_response}  

Here is the feedback for your original response:  
"{agent_1_feedback}"  

Please carefully consider the feedback and then update your summary of the context.  
Provide only the corrected summary without referencing the original answer, feedback, or previous errors. Your summary should focus strictly on the most relevant details required to answer the question.  
Respond in the following format:  
1. ***Analysis of the Context, Original Response, and Feedback***:
`...`

2. **Updated Context Summary**:
'Opinion: $Opinion' (without quotes) where Opinion is your final Context Summary.

'''

problem_solver_regenerate_prompt_pubmed = '''Your role is the Problem Solver.  
Here is the given question:  
"{question}"  

Here is the Context Analyst's response:  
<Context Analyst>{agent_1_original_response}</Context Analyst>  

Here is your original response:  
{agent_2_original_response}  

Here is the feedback for your original response:  
"{agent_2_feedback}"  

Please carefully consider the feedback and then update your solution to the problem. Provide only the corrected final answer without referencing the original answer, feedback, or any previous errors.  
Respond in the following format:  


1. **Reasoning:**  
"..."

2. **Final Answer:**  
provide a final answer to the given problem. {format_prompt}
'''












