import pandas as pd
import re
import json

file_path = 'huggingfacedata1.csv'  
output_path = 'filtered_prompts.jsonl' 

start_row = 15000  
num_rows = 5000    

def clean_text(text):
    #remove the useless #s and additional stuff
    if pd.isna(text):  
        return ""
    text = re.sub(r'###\s*(Instruction|Input|Response):', '', text)  
    text = re.sub(r'#.*', '', text)  
    text = text.replace('\n', ' ')  
    return text.strip() 

def word_count(text):
    """Count words in the text."""
    return len(text.split())

df = pd.read_csv(file_path, header=None, skiprows=start_row, nrows=num_rows)

df[0] = df[0].apply(clean_text)
df[1] = df[1].apply(clean_text)

filtered_df = df[df[0].apply(word_count) < 30]

with open(output_path, 'w', encoding='utf-8') as f:
    for row in filtered_df.itertuples(index=False):
        json_line = {"question": row[1], "answer": row[0]}
        f.write(json.dumps(json_line, ensure_ascii=False) + '\n')

print(f"Filtered prompts saved to {output_path}")
