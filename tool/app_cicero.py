import re
import json
import pickle
import random
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    T5ForConditionalGeneration
)
import datetime
import os

import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from cicero import Cicero
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
relation = ['What is the possible emotional reaction of the listener in response to target? <sep> target: ',
            'What is or could be the cause of target? <sep> target: ',
            'What subsequent event happens or could happen following the target? <sep> target: ',
            'What is or could be the motivation of target? <sep> target: ']
sep = ' <sep> context: '
utt = ', <utt> '

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
        self.cicero = Cicero(model_path, torch.device("cuda"))

    def get_commonsense(self, sen, others=None):
        cs_list = []
        input_event = " ".join(sen)
        for rel in relation:
            input=rel+input_event+sep
            cs_res= self.cicero.generate(input)  
            cs_list.append([cs_res])
        return cs_list


        

app = FastAPI()

@app.post("/v1/generate", status_code=status.HTTP_200_OK)
async def generate_relation(request: GenerateRequest) -> GenerateRelationOutput:
    add_info=""
    print("*****request.prompt*****")
    context = request.prompt
    processed_sentence = process_sent(context)
    cs_list = cicero_model.get_commonsense(processed_sentence)
    output = cs_list
    for con in zip(output):
        add_info = add_info+ str(con) + '\n'
    print(add_info)
    created_time = datetime.datetime.now()
    return GenerateRelationOutput(
        relation = add_info,
        generate_time = created_time
    )

if __name__ == "__main__":
    cicero_model = KnowledgeModel(model_path="/CICERO")
    uvicorn.run(
        app, host = "0.0.0.0", port=int(os.environ.get("API_PORT", 9092)), workers=1
    )    
