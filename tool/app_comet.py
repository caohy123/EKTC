import datetime
import os

import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from comet import Comet
import torch
import nltk
import inspect

WORD_PAIRS = {
    "it's": "it is",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "you'd": "you would",
    "you're": "you are",
    "you'll": "you will",
    "i'm": "i am",
    "they're": "they are",
    "that's": "that is",
    "what's": "what is",
    "couldn't": "could not",
    "i've": "i have",
    "we've": "we have",
    "can't": "cannot",
    "i'd": "i would",
    "i'd": "i would",
    "aren't": "are not",
    "isn't": "is not",
    "wasn't": "was not",
    "weren't": "were not",
    "won't": "will not",
    "there's": "there is",
    "there're": "there are",
}

### 任务说明（Instruction）+对话语境

reldef = '''xIntent represents their intent before the event.
xNeed represents what they need in order for the event to happen.
xWant represents what they would want after the event.
xEffect represents the effect of the event on the person.
xReact represents their reaction to the event.
'''
relations = ["xIntent", "xNeed", "xWant", "xEffect", "xReact"]
rels = ["x_intent: ", "x_need: ", "x_want: ", "x_effect: ", "x_react: "]

knowledge_origin = "Don't rush to reply, I can provide the following additional knowledge to help you provide a better reply. The following are the definitions of the five commonsense relations, followed by the content of the five relations extracted from the existing conversation. You can combine them and the dialogue context generates the final reply."

class GenerateRequest(BaseModel):
    prompt: str


class GenerateRelationOutput(BaseModel):
    relation: str
    generate_time: datetime.datetime = Field(default=datetime.datetime.now())

def process_sent(context):
    context = context.lower()
    for k, v in WORD_PAIRS.items():
        context = context.replace(k, v)
    context = nltk.word_tokenize(context)
    return context

class KnowledgeModel:
    def __init__(self, model_path):
        self.comet = Comet(model_path, torch.device("cuda"))

    def get_commonsense(self, sen, others=None):
        cs_list = []
        input_event = " ".join(sen)
        relations = ["xIntent", "xNeed", "xWant", "xEffect", "xReact"] 
        for rel in relations:
            cs_res = self.comet.generate(input_event, rel)
            cs_res = [process_sent(item) for item in cs_res]
            cs_list.append([" ".join(i) for i in cs_res])
            
        return cs_list
        

app = FastAPI()

@app.post("/v1/generate", status_code=status.HTTP_200_OK)
async def generate_relation(request: GenerateRequest) -> GenerateRelationOutput:
    add_info=""
    context = request.prompt
    processed_sentence = process_sent(context)
    cs_list = comet_model.get_commonsense(processed_sentence)
    output = cs_list
    for rel, con in zip(rels, output):
        add_info = add_info + rel + str(con) + '\n'
    print(add_info)
    created_time = datetime.datetime.now()
    return GenerateRelationOutput(
        relation = add_info,
        generate_time = created_time
    )

if __name__ == "__main__":
    comet_model = KnowledgeModel(model_path="/Comet/comet-atomic_2020_BART")
    uvicorn.run(
        app, host = "0.0.0.0", port=int(os.environ.get("API_PORT", 9091)), workers=1
    )    
