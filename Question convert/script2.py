import pandas as pd
import re
import json

# File paths
file_path = 'huggingfacedata1.csv'  # Change to your file path
output_path = 'filtered_prompts.jsonl'  # Output file name

# Parameters
start_row = 15000  # Row index to start reading from (0-based)
num_rows = 5000    # Number of rows to read

def clean_text(text):
    """Remove specified prefixes, hashtags, newline characters, and clean extra whitespace."""
    if pd.isna(text):  # Handle empty or NaN cells
        return ""
    text = re.sub(r'###\s*(Instruction|Input|Response):', '', text)  # Remove ### prefixes
    text = re.sub(r'#.*', '', text)  # Remove remaining hashtags
    text = text.replace('\n', ' ')  # Replace newline characters with spaces
    return text.strip()  # Remove leading/trailing whitespace

def word_count(text):
    """Count words in the text."""
    return len(text.split())

# Read the specified portion of the CSV file
df = pd.read_csv(file_path, header=None, skiprows=start_row, nrows=num_rows)

# Assuming column 0 contains prompts and column 1 contains answers; adjust if necessary
df[0] = df[0].apply(clean_text)
df[1] = df[1].apply(clean_text)

# Filter rows where the prompt has less than 30 words
filtered_df = df[df[0].apply(word_count) < 30]

# Write the data as a JSONL file
with open(output_path, 'w', encoding='utf-8') as f:
    for row in filtered_df.itertuples(index=False):
        json_line = {"question": row[1], "answer": row[0]}
        f.write(json.dumps(json_line, ensure_ascii=False) + '\n')

print(f"Filtered prompts saved to {output_path}")
