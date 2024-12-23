"""
Use FastChat with Hugging Face generation APIs.

Usage:
python3 -m fastchat.serve.huggingface_api --model lmsys/vicuna-7b-v1.3
python3 -m fastchat.serve.huggingface_api --model lmsys/fastchat-t5-3b-v1.0
"""

# Import necessary libraries
import os
import time
import torch  
from transformers import AutoTokenizer, AutoModelForCausalLM  
from huggingface_hub import snapshot_download 
from fastchat.model import load_model, get_conversation_template, add_model_args  # FastChat utilities
import logging  
import argparse  
import json  
import traceback  
import file_utils  

def download_model_if_needed(model_name, cache_dir=None):
    try:
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            local_files_only=True
        )
        print(f"Model already cached at: {model_path}")
    except FileNotFoundError:
        print(f"Model '{model_name}' not found locally. Downloading...")
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir
        )
        print(f"Model downloaded and cached at: {model_path}")
    return model_path

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Function to load the prompt template and tools
def load_prompt_and_tools(prompt_file, tools_file=os.path.join("..", "prompts-dataset", "externaltools.jsonl")):
    # Read the base prompt template
    prompt_template = file_utils.read_json(prompt_file)

    tools = []
    with open(tools_file, 'r') as f:
        for line in f:
            tools.append(json.loads(line))  

    tools_list = "\n".join([f"- {tool['name']}: {tool['description']}" for tool in tools])
    final_prompt = prompt_template + "\n" + "Selection of tools:\n" + tools_list
    return final_prompt

# generates output per line:
def generation(input, prompt_template, tokenizer, model):
    message = input
    conv = get_conversation_template(args.model_path)
    conv.set_system_message(prompt_template)
    conv.append_message(conv.roles[0], message)  
    conv.append_message(conv.roles[1], None) 
    prompt = conv.get_prompt() 

    # Tokenize and process the input
    inputs = tokenizer([prompt])
    inputs = {k: torch.tensor(v).to(args.device) for k, v in inputs.items()}

    # Generate the model's response
    output_ids = model.generate(
        **inputs,
        do_sample=True if args.temperature > 1e-5 else False, #greedy decoding/no
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,  # Penalty for repetition
        max_new_tokens=args.max_new_tokens,  
    )

    # Decode the output based on the model type
    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(inputs["input_ids"][0]):]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    return outputs

def run_single_test(args):
    model_mapping = {  # Mapping model paths to simplified names
        "baichuan-inc/Baichuan-13B-Chat": "baichuan-13b",
        "baichuan-inc/Baichuan2-13B-chat": "baichuan2-13b",  
        "THUDM/chatglm2-6b": "chatglm2",
        "lmsys/vicuna-13b-v1.3": "vicuna-13b",
        "lmsys/vicuna-7b-v1.3": "vicuna-7b",
        "lmsys/vicuna-33b-v1.3": "vicuna-33b",
        "meta-llama/Llama-2-7b-chat-hf": "llama2-7b",
        "meta-llama/Llama-2-13b-chat-hf": "llama2-13b",
        'TheBloke/koala-13B-HF': "koala-13b",
        "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5": "oasst-12b",
        "WizardLM/WizardLM-13B-V1.2": "wizardlm-13b"
    } 

    model_cache_dir = os.path.expanduser("~/.cache/huggingface")
    download_model_if_needed(args.model_path, cache_dir=model_cache_dir)

    # Load the model and tokenizer
    model, tokenizer = load_model(
        args.model_path,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        revision=args.revision,
        debug=args.debug,
    )

    test_type = args.test_type
    model_name = model_mapping[args.model_path]
    print(f"This run tests the output of {model_name}")
    if test_type == 'test_with_reason':
        test_with_reason(model_name=model_name, model=model, tokenizer=tokenizer)
    elif test_type == 'test_answer_only':
        test_answer_only(model_name=model_name, model=model, tokenizer=tokenizer)
    else:
        print("Invalid test_type. Please provide a valid test_type.")
        return None
    return "OK"

# Function to test with reason included in the response
def test_with_reason(model_name, model, tokenizer):
    model_name = model_name.lower()  
    output_file = os.path.join('test_results', f"{model_name}_results.jsonl")
    final_prompt_template = load_prompt_and_tools(os.path.join("..", "prompt_template", "use_api_or_no.txt"))
    all_data = []

    with open(os.path.join("..", "prompts-dataset", "final_apiQuestions_andAnswers.jsonl"), 'r') as f:
        data = [json.loads(line) for line in f]
        for el in data:
            try:
                response = generation(
                    input=el["question"],
                    prompt_template=final_prompt_template,
                    tokenizer=tokenizer,
                    model=model,
                )
                el["model_response"] = response
            except Exception as e:
                el["model_response"] = f"Error: {str(e)}"
                print(el["model_response"])
            
            all_data.append(el)

    # Save results to a JSONL file
    with open(output_file, 'w') as out_f:
        for item in all_data:
            out_f.write(json.dumps(item) + "\n")

# Function to test answers only (currently not implemented)
def test_answer_only(model_name, model, tokenizer):
    pass

# Main function to execute the script
@torch.inference_mode()
def main(args, max_retries=20, retry_interval=3):
    for attempt in range(max_retries):
        try:
            state = run_single_test(args)
            print(f"Test successful on attempt {attempt + 1}")
            return state  
        except Exception as e:
            traceback.print_exc()
            print(f"Error: {str(e)}. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)
    return None

# Logging configuration
timestamp = time.strftime("%Y%m%d%H%M%S")
log_filename = f"test_log_{timestamp}.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Entry point for the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser) 
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_type", type=str, default='test_with_reason')
    args = parser.parse_args()
    state = main(args)
    print(state)
