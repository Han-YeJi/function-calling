#%%
import json 
import numpy as np
import os
from IPython.display import display
import pandas as pd
from openai import OpenAI
import itertools
import time
import base64
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Any, Dict, List, Generator
import ast
#%%
client = OpenAI(api_key="")

#%%
def save_list_to_json(data_list, filepath):
    with open(filepath, 'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, ensure_ascii=False, indent=4)

def remove_sequences(input_string):
    # Replace the specific sequences with an empty string
    cleaned_string = input_string.replace("```json", "")  # Remove "```json" first
    cleaned_string = cleaned_string.replace("```", "")  # Then remove "```"
    return json.loads(cleaned_string)

def get_chat_completion(
    messages: list[dict[str, str]],
    model: str = "gpt-3.5-turbo",
    max_tokens=500,
    temperature=0.0,
    stop=None,
    tools=None,
    seed=42,
    functions=None,
    tool_choice=None,
) -> str:
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "tools": tools,
        "seed": seed,
        "tool_choice": tool_choice,
    }
    if functions:
        params["functions"] = functions

    completion = client.chat.completions.create(**params)
    return completion.choices[0].message
#%%
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

def generate_permutations(
    params: Dict[str, Dict[str, Any]]
) -> Generator[Dict[str, Any], None, None]:
    """
    Generates all possible permutations for given parameters.

    :param params: Parameter dictionary containing required and optional fields.
    :return: A generator yielding each permutation.
    """

    # Extract the required fields from the parameters
    required_fields = params.get("required", [])

    # Generate permutations for required fields
    required_permutations = generate_required_permutations(params, required_fields)

    # Generate optional permutations based on each required permutation
    for required_perm in required_permutations:
        yield from generate_optional_permutations(params, required_perm)

def generate_required_permutations(
    params: Dict[str, Dict[str, Any]], required_fields: List[str]
) -> List[Dict[str, Any]]:
    """
    Generates permutations for the required fields.

    :param params: Parameter dictionary.
    :param required_fields: List of required fields.
    :return: A list of permutations for required fields.
    """

    # Get all possible values for each required field
    required_values = [get_possible_values(params, field) for field in required_fields]

    # Generate permutations from possible values
    return [
        dict(zip(required_fields, values))
        for values in itertools.product(*required_values)
    ]
    
def generate_optional_permutations(
    params: Dict[str, Dict[str, Any]], base_perm: Dict[str, Any]
) -> Generator[Dict[str, Any], None, None]:
    """
    Generates permutations for optional fields based on a base permutation.

    :param params: Parameter dictionary.
    :param base_perm: Base permutation dictionary.
    :return: A generator yielding each permutation for optional fields.
    """

    # Determine the fields that are optional by subtracting the base permutation's fields from all properties
    optional_fields = set(params["properties"]) - set(base_perm)

    # Iterate through all combinations of optional fields
    for field_subset in itertools.chain.from_iterable(
        itertools.combinations(optional_fields, r)
        for r in range(len(optional_fields) + 1)
    ):

        # Generate product of possible values for the current subset of fields
        for values in itertools.product(
            *(get_possible_values(params, field) for field in field_subset)
        ):

            # Create a new permutation by combining base permutation and current field values
            new_perm = {**base_perm, **dict(zip(field_subset, values))}

            yield new_perm
            
def get_possible_values(params: Dict[str, Dict[str, Any]], field: str) -> List[Any]:
    """
    Retrieves possible values for a given field.

    :param params: Parameter dictionary.
    :param field: The field for which to get possible values.
    :return: A list of possible values.
    """

    # Extract field information from the parameters
    field_info = params["properties"][field]

    # Based on the field's type or presence of 'enum', determine and return the possible values
    if "enum" in field_info:
        return field_info["enum"]
    elif field_info["type"] == "integer":
        return [placeholder_int]
    elif field_info["type"] == "string":
        return [placeholder_string]
    elif field_info["type"] == "boolean":
        return [True, False]
    elif field_info["type"] == "array" and "enum" in field_info["items"]:
        enum_values = field_info["items"]["enum"]
        all_combinations = [
            list(combo)
            for i in range(1, len(enum_values) + 1)
            for combo in itertools.combinations(enum_values, i)
        ]
        return all_combinations
    return []

def save_to_jsonl(data, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def load_from_jsonl(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line.strip()))
    return data_list

def create_commands(invocation_list):
    example_list = []
    for i, invocation in enumerate(invocation_list):
        if i < 100:
            print(
                f"\033[34m{np.round(100*i/len(invocation_list),1)}% complete\033[0m")
            if type(invocation) == str or "json" in invocation:
                invocation = remove_sequences(invocation)
            print(invocation)

        # Format the prompt with the invocation string
        request_prompt = COMMAND_GENERATION_PROMPT.format(
            invocation=invocation)

        messages = [{"role": "user", "content": f"{request_prompt}"}]
        completion = get_chat_completion(model="gpt-4o", messages=messages, temperature=0.8) # model 을 gpt-4o로 변경!
        command_dict = {"Input": invocation, "Prompt": completion.content}
        example_list.append(command_dict)
    return example_list

def remove_descriptions(function_list):
    for function in function_list:
        func = function["function"]
        if "description" in func:
            del func["description"]

        params = func["parameters"]
        if "properties" in params:
            for param in params["properties"].values():
                if "description" in param:
                    del param["description"]

    return function_list

#%%
placeholder_int = "fill_in_int"
placeholder_string = "fill_in_str"

INVOCATION_FILLER_PROMPT = """
1) Input reasonable values for 'fill_in_string' and 'fill_in_int' in the invocation here: {invocation}. Reasonable values are determined by the function definition. Use the
the entire function provided here :{function} to get context over what proper fill_in_string and fill_in_int values would be.
Example:

Input: invocation: {{
    "name": "get_xdr_risk_report",
    "arguments": {{
      "start_date":"fill_in_str",
      "end_date":"fill_in_str"
    }}
}},
function:{function}

Output: invocation: {{
    "name": "get_xdr_risk_report",
    "arguments": {{
      "start_date": "2024-05-07",
      "end_date": "2024-05-07"
    }}
}}


MAKE SURE output is just a dictionary with keys 'name' and 'arguments', no other text or response.

Input: {invocation}
Output:
"""

COMMAND_GENERATION_PROMPT = """
You are to output 5 commands, questions or statements that would generate the inputted function and parameters.
Please make the commands or questions natural, as a person would ask, and the command or questions should be varied and not repetitive.
It should not always mirror the exact technical terminology used in the function and parameters, rather reflect a conversational and intuitive request.
For instance, the prompt should not be 'turn on the dome light', as that is too technical, but rather 'turn on the inside lights'.
Another example, is the prompt should not be 'turn on the HVAC', but rather 'turn on the air conditioning'. Use language a normal driver would use, even if
it is technically incorrect but colloquially used.

RULES: ALWAYS put a backwards slash before an apostrophe or single quote '. For example, do not say don't but say don\'t.
Prompts MUST be in double quotes as well.

Example

Input: {{'name': 'get_xdr_risk_report','arguments': {{'start_date': '2024-04-23', 'end_date': '2024-04-23'}}'' }}
Prompt: ["What caused the high risk score on 2024-04-23?","Tell me the high risk score on April 23, 2024","Today is April 23, 2024. Today's risk report.","Today is May 23, 2024. Any critical risk 30 days ago?","What new risks have arisen yesterday? Today is April 24, 2024","The current date is April 24, 2024. What new risks caused at yesterday?","what new risks have arisen yesterday. today is 2024-04-24"]

Input: {{'name': 'get_xdr_risk_statistics','arguments': {{'start_date': '2024-04-20', 'end_date': '2024-04-26'}}'' }}
Prompt: ["Today is Friday, April 26, 2024. What are the most common risk factors detected this week?","The current date is April 26, 2024. Risk statistics for last 7 days.","Please tell me the number of risks by log level in the past week as of April 26, 2024.","Analyze risks from April 20 to April 26 2024", "Today is April 26. how many e-mail risk factors for last 7 days?","Today is Tuesday, April 30, 2024. Last week"s Risk Statistics","Today is Tuesday, April 30, 2024. Risk Statistics for last week","Today is Monday, April 29, 2024. What are the most common risk factors detected last week?","Today is April 29, 2024. Please tell me the number of risks in the last 7 days?"]

Input: {{'name': 'get_xdr_devices_by_department,'arguments': {{'department':'Saas Development Team', 'start_date': '2024-04-01', 'end_date': '2024-04-30'}}'' }}
Prompt: ["Today is April 30, 2024. List of devices relevant to Saas Development Team department, covering the period from April 1 to today.","Today is May 2, 2024. List of devices relevant to Saas Development Team department, covering the period from April 1 to April 30.","The current date is April 30, 2024. Give list of devices about Saas Development Team department of this month.","Today is April 30, 2024. List of devices relevant to Saas Development Team department on a month.","Today is May 5, 2024. Give list of devices about Saas Development Team department of last month?"]

Input: {invocation}
Prompt:
"""

#%%
function_list = load_from_jsonl("function_call_list.jsonl")

input_objects = []
for function in function_list[:2]:
    func_name = function["function"]["name"]
    params = function["function"]["parameters"]
    for arguments in generate_permutations(params):
        if any(val in arguments.values() for val in ["fill_in_int", "fill_in_str"]):
            input_object = {"name": func_name, "arguments": arguments}
            messages = [
                {
                    "role": "user",
                    "content": INVOCATION_FILLER_PROMPT.format(
                        invocation=str(input_object), function=function
                    ),
                }
            ]
    
            input_object = get_chat_completion(
                model="gpt-4o", messages=messages, max_tokens=200, temperature=0.1
            ).content
        else:
            input_object = {"name": func_name, "arguments": arguments}

        input_objects.append(input_object)

training_examples_unformatted = create_commands(input_objects)
modified_function_list = remove_descriptions(function_list)

#%%
training_examples = []
for prompt in training_examples_unformatted:
    # adjust formatting for training data specs

    # if its not a dict, convert to dict
    if type(prompt["Input"]) != dict:
        prompt["Input"] = ast.literal_eval(prompt["Input"])
        print("dot dict")
    prompt["Input"]["arguments"] = json.dumps(prompt["Input"]["arguments"])
    try:
        prompt["Prompt"] = json.loads(prompt["Prompt"])
    except:
        continue
    for p in prompt["Prompt"]:
        tool_calls = [
            {"id": "call_id", "type": "function", "function": prompt["Input"]}
        ]
        training_examples.append(
            {
                "input": p,
                "output": [{"tool": prompt["Input"]['name'], "tool_input": prompt["Input"]['arguments']}]
            }
        )
# %%
# 학습 데이터 저장
save_to_jsonl(training_examples, 'function_call_data.jsonl')

