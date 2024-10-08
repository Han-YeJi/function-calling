import os
os.environ["CUDA_VISIBLE_DEVICES"]= "3"
os.environ["WORLD_SIZE"] = "1"

import json 
import torch
from tqdm import tqdm
import pandas as pd 
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.cuda.device_count()
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

fold_path = "4batch_3epoch"
data_path = "../data/valid_data.json"
data = load_json(data_path)

model_path = f"../results/{fold_path}/qwen-lora-best"
model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        ).to(torch_device)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", padding_side='right')
        
SYSTEM_PROMPT = """You are an intelligent AI. Given a command or request from the user,
call one of your functions to complete the request. If the request is ambiguous or unclear, reject the request."""

model_outputs, true_labels, questions = [], [], []
for i in tqdm(data):
    chat = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': i['question']}
        ]            

    chat_template_data = tokenizer.apply_chat_template(chat, tokenize= True, return_tensors='pt').to(torch_device)
    output = model.generate(chat_template_data, max_length=8000)
    output_text = tokenizer.decode(output[0][chat_template_data.size(1):], skip_special_tokens = True)
    
    true_labels.append(i['answer'])
    model_outputs.append(output_text)
    questions.append([i['question']])
    
df = pd.DataFrame({
                    'Question':questions,
                    'Ground_Truth':true_labels,
                    'Model output':model_outputs
                })


output_path = "../results/table_output/test.xlsx"
os.makedirs(output_path, exist_ok=True)

df.to_excel(output_path, index=False)
