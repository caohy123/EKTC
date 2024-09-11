import argparse
import json
import os

import openai
import datetime
import requests
from datasets import load_dataset
from tqdm import tqdm
from llmtuner import ChatModel

from constants import SYSTEM_PROMPT, generate_relation_tool

knowledge_origin = "Don't rush to reply, I can provide the following additional knowledge to help you provide a better reply. The following are the definitions of the five commonsense relations, followed by the content of the five relations extracted from the existing conversation. You can combine them and the dialogue context generates the final reply."

descriptions='''
xIntent represents their intent before the event.
xNeed represents what they need in order for the event to happen.
xWant represents what they would want after the event.
xEffect represents the effect of the event on the person.
xReact represents their reaction to the event.
''' 
def get_args():
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("--dataset", default=None, type=str, help="The dataset to use for evaluation")
    parser.add_argument("--save_dir", default=None, type=str, help="The directory to save the relation")
    return parser.parse_args()

def EmotionKnowledgeBase(prompt: dict, save_dir: str):
    request_url = "http://localhost:9091/v1/generate"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    prompt.update({"save_dir": save_dir})
    response = requests.post(request_url, json=prompt)
    
    return response.json()

def get_model_response(client: openai.OpenAI, messages: list, tools: list, verbose: bool = False):
    if verbose:
        print(f"Messages: {messages}")
    response = client.chat.completions.create(
        model="tool_relation_v1",
        messages=messages,
        tools=tools,
        max_tokens = 300,
        temperature = 0.1
    )
    return response


def eval(client: openai.OpenAI, conversations: list[dict], tools: list, relation_save_dir: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    messages_2 = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    historys = []
    results = []
    for idx, conversation in enumerate(conversations):
        print("len: "+str(len(conversations)))
        role, content = conversation.get("from"), conversation.get("value")
        print(role+":"+content)
        if role == "user":
            print("idx(user): "+str(idx))
            if idx >= len(conversations)-1:
                break
            result = dict()
            history = []
            
            history.append(content)
            result["query"] = content
            messages.append({"role": "user", "content": content})
            # get client response
            response = get_model_response(client, messages, tools)
            print(response.choices[0])
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "tool_calls":
                tool_calls = response.choices[0].message.tool_calls[0]
                function_name = tool_calls.function.name
                function_argument = tool_calls.function.argument
                function_call = f"Action: {function_name}\nAction Input: {function_argument}"
                print(function_call)
                # get tool function
                if function_name not in TOOL_MAP.keys():
                    observation = dict(
                        status_code = 400, detail = f"Function {function_name} not found"
                    )
                    raise ValueError(f"Function {function_name} not found")
                else:
                    tool_result = TOOL_MAP[function_name](
                        prompt = json.loads(function_argument), save_dir = relation_save_dir
                    )
                    result["pred_relation_metadata"] = tool_result
                    
                    observation = dict(
                        status_code = 200, relation = tool_result["relation"]
                    )
                # get response
                add_info = knowledge_origin +'\n'+ descriptions + '\n' + tool_result["relation"]
                messages_2.append({"role": "user", "content": content+add_info})
                response = get_model_response(
                        client,
                        messages_2,
                        tools
                    )
            print(response.choices[0].message.content)
            result["pred_text_response"] = response.choices[0].message.content    
            
        elif role == "assistant":
            print("idx(assistant): "+str(idx))
            history.append(content)
            historys.append(history)
            result['response_text'] = content
            result['history'] = historys.copy()
            results.append(result)
            messages.append({"role": "assistant", "content": content})
            
        else:
            messages.append({"role": role, "content": content})
            if role == "observation":
                print("idx(observation): "+str(idx))
                relation = json.loads(content).get("relation")
                result["response_relation_metadata"] = dict(
                    relation = relation, description = description)
                
                result["response_relation"] = [relation]
            elif role == "function_call":
                print("idx(function_call): "+str(idx))
                description = json.loads(content)["arguments"]["prompt"]

    return results

def main():
    args = get_args()
    dataset = load_dataset('json', data_files=args.dataset, split='train')
    print(dataset)
    
    client = openai.OpenAI(
        base_url = "http://localhost:9095/v1",
        api_key = "0"
    )
    tools = [dict(
        type = "function", function = generate_relation_tool
    )]
    
    results = []
    
    save_dir = os.path.join(
        args.save_dir, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    )
    os.makedirs(save_dir)
    relation_save_dir = os.path.join(
        save_dir, "relation"
    )
    if not relation_save_dir:
        os.makedirs(relation_save_dir)
        
    for data in tqdm(dataset):
        results.extend(
            eval(client, data["conversations"], tools = tools, relation_save_dir=relation_save_dir)
        )
    
    print(f"Length of Results: {len(results)}")
    json.dump(
        results, 
        open(os.path.join(save_dir, "result.json"), "w"), 
        indent=4, ensure_ascii=False
    )
    

TOOL_MAP = {
    "EmotionKnowledgeBase": EmotionKnowledgeBase
}

if __name__ == "__main__":
    main()