import json
from config import getConfig

from torch.utils.data import Dataset
from transformers import AutoTokenizer

args = getConfig()

SYSTEM_PROMPT = """You are an intelligent AI that controls a drone. Given a command or request from the user,
call one of your functions to complete the request. If the request cannot be completed by your available functions, call the reject_request function.
If the request is ambiguous or unclear, reject the request."""

class DataForLoRA:
    def __init__(self, data_path, args):
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side='right')
        
    def preprocess(self, mode):
        preprocessed_data = []
        
        if mode == "train":
            for i in range(len(self.data)):
                answer = []
                for x in self.data[i]['answer']:
                    answer.append({
                        'tool': x['tool'],
                        'tool_input': json.loads(x['tool_input']),
                    })
                            
                chat = [
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': self.data[i]['question']},
                        {'role': 'assistant', 'content': str(answer)}, 
                    ]
                data = self.tokenizer.apply_chat_template(chat, tokenize= False)
                preprocessed_data.append({'text':data})
        else:
            for i in range(len(self.data)):
                chat = [
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': self.data[i]['question']},
                        {'role': 'assistant', 'content': json.dumps(self.data[i]['answer'])}, 
                    ]            
        
                data = self.tokenizer.apply_chat_template(chat, tokenize= False)
                preprocessed_data.append({'text':data})
        
        return preprocessed_data
