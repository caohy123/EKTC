import argparse
import os
import json
import pandas as pd
from bert_score import score
from cider.cider import Cider
from datasets import load_dataset
from evaluate import load
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from tqdm.auto import tqdm


def calc_distinct_n(n, candidates):
    dict = {}
    total = 0
    candidates = [word_tokenize(candidate) for candidate in candidates]
    for sentence in candidates:
        for i in range(len(sentence) - n + 1):
            ney = tuple(sentence[i : i + n])
            dict[ney] = 1
            total += 1
    score = len(dict) / (total + 1e-16)

    if True:
        print(f"***** Distinct-{n}: {score*100} *****")

    return {
        f"distinct_{n}": score
    }

def calc_distinct(predictions, references):
    scores = {}
    for i in range(2):
        score = calc_distinct_n(i + 1, predictions)
        scores.update(score)
    print(scores)
    return scores

###########


def calculate_rouge_score(predictions, references):
    rouge = load('rouge')
    results = rouge.compute(predictions=predictions, references=references)
    return results

def calculate_cider_score(predictions, references):
    scorer = Cider()
    predictions = {idx: [data] for idx, data in enumerate(predictions)}
    references = {idx: [data] for idx, data in enumerate(references)}
    
    return {
        "cider-score": scorer.compute_score(gts=references, res=predictions)[0]
    }

def calculate_meteor_score(predictions, references):
    scores = []
    for pred, ref in zip(predictions, references):
        tokenized_pred = word_tokenize(pred)
        tokenized_ref = word_tokenize(ref)
        score = meteor_score(references=[tokenized_ref], hypothesis=tokenized_pred)
        scores.append(score)
    return {
        "meteor-score": sum(scores) / len(scores)
    }

def calculate_bert_score(predictions, references):
    P, R, F1 = score(predictions, references, lang='en', verbose=False)
    return {
        "bert-score-precision": P.mean().item(),
        "bert-score-recall": R.mean().item(),
        "bert-score-f1": F1.mean().item()
    }

    
def calculate_corpus_bleu_score(predictions, references):
    """
    Calculate BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores for a list of predictions and references.

    :param predictions: A list of predicted sentence strings.
    :param references: A list of reference sentence strings. Each reference should correspond to the prediction at the same index.
    :return: A dictionary with BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores for the entire corpus.
    """
    # Tokenize each sentence in the predictions and references
    tokenized_predictions = [word_tokenize(pred) for pred in predictions]
    tokenized_references = [[word_tokenize(ref)] for ref in references]  # Each reference should be wrapped in another list

    smooth=SmoothingFunction()
    # Calculate BLEU scores for the entire corpus
    bleu_scores = {
        "BLEU-1": corpus_bleu(tokenized_references, tokenized_predictions, weights=(1, 0, 0, 0), smoothing_function=smooth.method1),
        "BLEU-2": corpus_bleu(tokenized_references, tokenized_predictions, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth.method1),
        "BLEU-3": corpus_bleu(tokenized_references, tokenized_predictions, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth.method1),
        "BLEU-4": corpus_bleu(tokenized_references, tokenized_predictions, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
    }

    return bleu_scores

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--data_type', type=str, default='pegs')
    
    return parser.parse_args()

def save_result(result: dict, filename: str):
    data = pd.DataFrame([result])
    print(data.head())
    data.to_csv(filename, index=False)
    
def main(predictions, references):
    results = {}
    tasks = [
        (calculate_rouge_score, "Calculating ROUGE Score"),
        (calculate_bert_score, "Calculating BERT Score"),
        (calculate_corpus_bleu_score, "Calculating Corpus BLEU Scores"),
        (calculate_meteor_score, "Calculating METEOR Score"),  # Ensure this is the correct function name
        (calculate_cider_score, "Calculating CIDEr Score"),      # Ensure this is the correct function name
        (calc_distinct, "Calculating Distinct")
    ]

    for func, description in tqdm(tasks, desc="Evaluating Metrics"):
        tqdm.write(description)
        results.update(func(predictions=predictions, references=references))

    return results

if __name__ == '__main__':
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    config = get_args()
    
    data_path = os.path.join(config.data_path, 'result.json')
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(dataset)

    predictions = []
    references = []
    
    none_value = 0
    for data in dataset:
        gt = data.get('origin_text', '').strip()
        pr = data.get('pred_response', '').strip() if data.get('pred_response') is not None else ''


        if len(pr) == 0:
            none_value += 1
        else:
            predictions.append(pr)
            references.append(gt)
        
    results = main(predictions=predictions, references=references)
    results.update({"none_text_response": none_value})
    
    save_dir = os.path.join(config.data_path, 'metric')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, 'response_metric.csv')
    
    save_result(result=results, filename=filename)