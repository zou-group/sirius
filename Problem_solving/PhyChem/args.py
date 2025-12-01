import argparse

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--n_generate_sample', type=int, default=1)
    args.add_argument('--temperature', type=float, default=0.) 
    args.add_argument('--max_tokens', type=int, default=4096) 


    args.add_argument('--model', type=str, default= 'gpt-3.5-turbo', choices=['gpt-3.5-turbo', "gpt-4o-mini-2024-07-18"])
    args.add_argument('--task', type=str, default='MMLU_physics',choices=['theoremqa_physics', "gpqa_diamond_physics", "MMLU_physics"])
    args.add_argument('--prompt_type', type=str,default='multi_agent', choices=['single_agent_zeroshot','multi_agent'])  
    
    args.add_argument('--ft_round', type=int, default=0) 
    args.add_argument('--sol_round', type=int, default=0) 
    args.add_argument('--fd_round', type=int, default=0) 
    args.add_argument('--re_round', type=int, default=0) 

    args.add_argument('--mode', type=str, default= 'generate', choices=['eval', 'generate'])
    args.add_argument('--subject', type=str, default='phy', choices=['phy', 'chem',"PubMedQA"])

    args = args.parse_args()

    return args
