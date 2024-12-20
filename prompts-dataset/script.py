import json
import random

# File paths
input_file_path = 'APIQuestionsandResponses.jsonl'
output_file_path = 'final_apiQuestions_andAnswers.jsonl'

def safe_json_loads(line):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None

with open(input_file_path, 'r', encoding='utf-8') as infile:
    lines = infile.readlines()

json_objects = [safe_json_loads(line) for line in lines]
json_objects = [obj for obj in json_objects if obj is not None]

selected_objects = random.sample(json_objects, min(200, len(json_objects)))

def rename_second_key(obj):
    keys = list(obj.keys())
    if len(keys) > 1:
        obj['answer'] = obj.pop(keys[1])
    return obj

updated_objects = [rename_second_key(obj) for obj in selected_objects]

with open(output_file_path, 'w', encoding='utf-8') as outfile:
    for obj in updated_objects:
        outfile.write(json.dumps(obj, ensure_ascii=False) + '\n')

print(f"{len(updated_objects)} random objects updated and saved to {output_file_path}")
