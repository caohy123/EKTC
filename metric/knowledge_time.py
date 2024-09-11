import argparse
import os

from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from tqdm.auto import tqdm
import json

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="Path to the input file.")  

config = parser.parse_args()


data_path = os.path.join(config.data_path, 'result.json')
with open(data_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

length = len(dataset)

gt_sticker_label = [0]*length
pred_sticker_label = [0]*length

for idx, data in enumerate(dataset):
    pred_image = data.get("pred_knowledge", None)
    groud_truth_image = data.get("origin_knowledge", None)
    
    gt_sticker_label[idx] = 1 if groud_truth_image else 0
    pred_sticker_label[idx] = 1 if pred_image else 0




accuracy = accuracy_score(gt_sticker_label, pred_sticker_label)
f1 = f1_score(gt_sticker_label, pred_sticker_label)
recall = recall_score(gt_sticker_label, pred_sticker_label)
precision = precision_score(gt_sticker_label, pred_sticker_label)


output_dict = {
    'accuracy': accuracy,
    'f1': f1,
    'recall': recall,
    'precision': precision,
    'gt_freq': sum(gt_sticker_label)/length,
    'pred_freq': sum(pred_sticker_label)/length,

}

save_dir = os.path.join(config.data_path, 'metric')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(os.path.join(save_dir, 'reflect_toolusetime.json'), 'w') as f:
    json.dump(output_dict, f, indent=4, ensure_ascii=False)