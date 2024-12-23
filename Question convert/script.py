import json
import re

input_file_path = 'filtered_prompts.jsonl'  
output_file_path = 'newfiltered.jsonl'  
def word_count(text):
    """Count the number of words in a given text."""
    return len(re.findall(r'\w+', text))

with open(input_file_path, 'r', encoding='utf-8') as infile, \
     open(output_file_path, 'a', encoding='utf-8') as outfile:
    
    for line in infile:
        try:
            obj = json.loads(line)
            
            if 'question' in obj and word_count(obj['question']) < 30:
                outfile.write(json.dumps(obj, ensure_ascii=False) + '\n')
        
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line: {line.strip()}")
        except Exception as e:
            print(f"An error occurred: {e}")

print(f"Filtered questions saved to {output_file_path}")
