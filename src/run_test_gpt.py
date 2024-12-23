import json
import openai
import os
import time


#openai.api_key = '<OUR KEY>'

queriesFile = "C:/Users/LEGION\Downloads/rjae-research-rep/rjae-research/prompts-dataset/final_apiQuestions_andAnswers.jsonl"
toolDescFile = 'C:/Users/LEGION/Downloads/rjae-research-rep/rjae-research/prompts-dataset/externaltools.jsonl'
#need to mofidy

def load_tool_descriptions():
    with open(toolDescFile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

tool_list = load_tool_descriptions()

formatted_tool_list = "\n".join([f"{name}: {desc}" for name, desc in tool_list.items()])

prompts = []
prompts.append("What's 8 +4?")
prompts.append("Who 'discovered' the United States?")
with open(queriesFile, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()  
        if not line: 
            continue
        try:
            # Parse each line as a JSON object
            data = json.loads(line)
            # Check if "question" key exists
            if 'question' in data:
                prompts.append(data['question'])
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON line: {line} due to {e}")

prompts.append("What's 2+8?")

count = 0
for prompt in prompts:
    print(prompt)
    if(count > 10):
        break
    count += 1
    system_message = (
        "You are a helpful AI assistant. Your task is to choose the appropriate tool (including APIs and Plugins) to solve the user's query.\n"
        "Try to answer the question, but if you feel that there's a suitable tool in the list attached below, respond with it instead.\n" #test first, later change to otherwise, return an answer without using API
        "List of tools:\n"
        f"{formatted_tool_list}\n"
        "\n Example Response: Can I find any peer-reviewed papers? ResearchHelper\n"
        "Can I generate bibtex bibliographies? ResearchHelper"
        "Simply include the tool name, don't give any long sentences."
        "If you ABSOLUTELY CANNOT find an item in the list that suits the task, just answer the question."
    )

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7,
    )

    generated_text = response.choices[0].message.content.strip()
    print(generated_text)

    time.sleep(1)

