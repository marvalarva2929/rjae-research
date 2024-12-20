import csv
import json

input_file = 'all_clean_data.csv'  
output_file = 'API Questions and Responses.jsonl' 

with open(input_file, mode='r', newline='', encoding='utf-8') as csv_file, \
     open(output_file, mode='a', encoding='utf-8') as jsonl_file:
    
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  

    for row in csv_reader:
        if len(row) == 2:
            question, api_response = row
            json_line = {"question": question, "api_response": api_response}
            jsonl_file.write(json.dumps(json_line) + '\n')

print(f"Conversion complete. Data appended to {output_file}")
