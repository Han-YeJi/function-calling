#%%
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "3"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
                
model_path = "4batch_3epoch"
base = "Qwen/Qwen2-7B-Instruct"
lora =  f"../results/{model_path}/checkpoint-1000"
save =  f"../results/{model_path}/qwen-lora-best"

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map='auto')

model = PeftModel.from_pretrained(model, lora)
model = model.merge_and_unload()
model.save_pretrained(save)
tokenizer.save_pretrained(save)
