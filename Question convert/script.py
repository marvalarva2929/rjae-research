import json
import re

# Define file paths
input_file_path = 'filtered_prompts.jsonl'  # Input JSONL file
output_file_path = 'newfiltered.jsonl'  # Output JSONL file

def word_count(text):
    """Count the number of words in a given text."""
    return len(re.findall(r'\w+', text))

# Open the input and output files
with open(input_file_path, 'r', encoding='utf-8') as infile, \
     open(output_file_path, 'a', encoding='utf-8') as outfile:
    
    # Process each line in the input file
    for line in infile:
        try:
            # Parse the JSON object
            obj = json.loads(line)
            
            # Check if 'question' field exists and has fewer than 30 words
            if 'question' in obj and word_count(obj['question']) < 30:
                # Write the JSON object to the output file
                outfile.write(json.dumps(obj, ensure_ascii=False) + '\n')
        
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line: {line.strip()}")
        except Exception as e:
            print(f"An error occurred: {e}")

print(f"Filtered questions saved to {output_file_path}")
