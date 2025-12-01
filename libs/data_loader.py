import json
import os
import re
import pandas as pd


def extract_answer_letter(input_string):
    ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-D])"
    match = re.search(ANSWER_PATTERN_MULTICHOICE, input_string)
    extracted_answer = match.group(1) if match else input_string
    return extracted_answer

def extract_answer_number(input_string):
    ANSWER_PATTERN_NUMBER = r"(?i)Answer\s*:\s*(-?\d+\.?\d*)"
    match = re.search(ANSWER_PATTERN_NUMBER, input_string)
    extracted_answer = match.group(1) if match else input_string
    return extracted_answer
def extract_answer_yesno(input_string):
    ANSWER_PATTERN_YESNO = r"(?i)\s*(yes|no|maybe|Yes|No|Maybe)"
    match = re.search(ANSWER_PATTERN_YESNO, input_string)
    extracted_answer = match.group(1) if match else input_string
    return extracted_answer


def extract_response(input_string):
    OPINION_PATTERN = r"(?i)Opinion\s*:(.*)"
    match = re.search(OPINION_PATTERN, input_string,re.DOTALL)
    extracted_response = match.group(1) if match else input_string
    return extracted_response



def load_jsonl_objects(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        objects = [json.loads(line) for line in f]
    return objects

def load_dataset(args):
    questions = []
    answers = []
    task=[]
    subject = args.subject
    # file_path = f"dataset/{args.task}.jsonl"
    if args.mode == "eval":
        file_path=f"dataset/{subject}_test.jsonl"
        print(file_path)
    if args.mode == "generate":
        file_path=f"dataset/{subject}_train.jsonl"
    
    datas=load_jsonl_objects(file_path)
   
    for row in datas:
        if isinstance(row['groundtruth'], bool):
           row['groundtruth']=[str(row['groundtruth']), None]
        elif isinstance(row['groundtruth'], (list, int, float)):
            row['groundtruth']=[str(row['groundtruth']), row['groundtruth']]
        else:
            row['groundtruth']=[str(row['groundtruth']), None]
                
    return datas
