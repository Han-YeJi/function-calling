#%%
import json 
import random 

random.seed(42)

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def save_to_json(data, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
#%%    
# 학습데이터 만들기
# function call + xdr rag + code interpreter
    
function_call_data = load_jsonl("train_data.jsonl")
xdr_rag_data = load_json("General_QA-0801.json")
code_data = load_json("질문의도분류_training_data.json")

xdr_rag_train = random.sample(xdr_rag_data, 1000)
xdr_rag_test = [item for item in xdr_rag_data if item not in xdr_rag_train]

code_interpreter_train = random.sample(code_data, 1000)
code_interpreter_test = [item for item in code_data if item not in code_interpreter_train]

# xdr과 code interpreter는 함수 호출 쿼리가 아니므로 처리를 해준다. 
xdr_rag_train = [{"question":prompt['question'], "answer":[{"tool": 'reject_request', "tool_input": "{}"}]} for prompt in xdr_rag_train]
code_interpreter_train = [{"question":prompt['question'], "answer":[{"tool": 'reject_request', "tool_input": "{}"}]} for prompt in code_interpreter_train]

train_data = function_call_data + xdr_rag_train + code_interpreter_train

random.shuffle(train_data)
save_to_json(train_data, "train_data.json")

print("function_call_data size:", len(function_call_data))
print("xdr_rag_train size:", len(xdr_rag_train))
print("code_interpreter_train size:", len(code_interpreter_train))
print("train data size:", len(train_data))

# %%
# 평가데이터 만들기
xdr_rag_valid = random.sample(xdr_rag_test, 100)
code_interpreter_valid = random.sample(code_interpreter_test, 100)

# function call 평가 데이터가 따로 존재하여 다음과 같이 진행함.
file_path = "function_call_ground_truth_yj.json"
with open(file_path, 'r', encoding='utf-8') as f:
    function_call_test_data = json.load(f)

function_call_valid = []
for data in function_call_test_data:
    function_call_valid.append({"question":data['question'], "answer":data['expected_call_str']})

function_call_valid = random.sample(function_call_valid, 60)

# 학습데이터와 마찬가지로 xdr과 code interpreter는 함수 호출 쿼리가 아니므로 처리를 해준다.
xdr_rag_valid = [{"question":prompt['question'], "answer":[{"tool": 'reject_request', "tool_input": "{}"}]} for prompt in xdr_rag_valid]
code_interpreter_valid = [{"question":prompt['question'], "answer":[{"tool": 'reject_request', "tool_input": "{}"}]} for prompt in code_interpreter_valid]

valid_data = code_interpreter_valid + xdr_rag_valid + function_call_valid
save_to_json(valid_data, "valid_data.json")
