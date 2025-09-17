import os
import json
import ast
import re
import numpy as np
def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        try:
           data = json.load(f)
        except json.JSONDecodeError:
            return None
        else:
            return data
def save_json(json_obj, file_name):
    if not os.path.exists(file_name):
        dir_path = os.path.dirname(file_name)
        if not os.path.exists(dir_path) and dir_path != "":
            # create directory
            os.makedirs(dir_path)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)

def save_json_add(json_obj, file_name):
    res = load_json(file_name)
    if res is None:
        res = []
    res.extend(json_obj)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


def save_to_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            # JSON string
            json_line = json.dumps(item, ensure_ascii=False)
            # adds a newline
            f.write(json_line + '\n')

def load_to_jsonl(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Removes possible line breaks and Spaces
            stripped_line = line.strip()
            if stripped_line:
                data = json.loads(stripped_line)
                data_list.append(data)
    return data_list

def changeStrToJson(str:str):
    # Extract content
    pattern = re.compile(r'\{((?:[^{}]*|\{.*?\})*)\}', re.DOTALL)
    json_content = pattern.search(str)
    if json_content is None:
        return None
    json_str = "{" + json_content.group(1) + "}"
    json_str = json_str.replace('json', '')
    json_str = json_str.replace('`', '')
    try:
        data = json.loads(json_str)
        return data
    except Exception as e:
        print(e)
        return None

def changeStrToList(str:str):
    pattern = re.compile(r'\[.*?\]', re.DOTALL)
    list_content = pattern.search(str)
    if list_content == None:
        return None
    list_content = list_content.group()
    try:
        result = ast.literal_eval(list_content)
        if not isinstance(result, list):
            return None
        return result
    except:
        return None

def cosine_similarity_manual(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    sim = dot_product / (norm_x[:, np.newaxis] * norm_y)
    return sim
