import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from tqdm import tqdm

from datasets import Dataset
from transformers import (Trainer, 
                          TrainingArguments, 
                          AutoTokenizer, 
                          DataCollatorWithPadding,
                          AutoModelForSequenceClassification,)


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt-dir', required=True)
parser.add_argument('--test-dir', required=True)
parser.add_argument('--output-file', required=True)
args = parser.parse_args()

def read_files_from_directory(directory, label):
    """
    从给定目录中读取文本文件，并为其标记指定的标签。
    """
    texts = []
    labels = []
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]
    with ThreadPoolExecutor() as executor:
        for text in tqdm(executor.map(read_file, files), total=len(files), desc=f'''Reading {label} texts from `{'/'.join(directory.split('/')[-4:])}`'''):
            texts.append(text)
            labels.append(1 if label == 'pos' else 0)
    return texts, labels

def read_file(file_path):
    """
    读取单个文件并返回其内容。
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def load_dataset(directory):
    """
    从指定目录加载数据集，该目录应该包含 'pos' 和 'neg' 子目录。
    """
    pos_directory = os.path.join(directory, 'pos')
    neg_directory = os.path.join(directory, 'neg')

    pos_texts, pos_labels = read_files_from_directory(pos_directory, 'pos')
    neg_texts, neg_labels = read_files_from_directory(neg_directory, 'neg')

    texts = pos_texts + neg_texts
    labels = pos_labels + neg_labels

    return Dataset.from_dict({'text': texts, 'label': labels})

imdb_test_dataset = load_dataset(args.test_dir)

# 加载模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir)


def preprocess_function(examples):
    # tokenizer参数请参考：https://huggingface.co/docs/transformers/main_classes/tokenizer
    return tokenizer(examples["text"], 
                     max_length=512, 
                     truncation=True,)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 利用函数`preprocess_function`对所有文本进行tokenization，通过设置batched=True加快处理速度（一次处理多条数据）
tokenized_test_imdb = imdb_test_dataset.map(preprocess_function, batched=True)

# 加载 TrainingArguments 实例
training_args = torch.load(f"{args.ckpt_dir}/training_args.bin")

# 定义评价指标，在文本分类任务中是准确率
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    correct = (predictions == labels).sum()
    total = len(labels)
    accuracy = correct / total
    return {"accuracy": accuracy}

# 开始训练前确定标签和id的映射关系
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# 模型实例化
model = AutoModelForSequenceClassification.from_pretrained(
    args.ckpt_dir, num_labels=2, id2label=id2label, label2id=label2id
)

# 创建 Trainer 实例
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 在测试数据集上进行推理
prediction_output = trainer.predict(tokenized_test_imdb)

# Writing the prediction output to a jsonl file
with open(args.output_file, 'w') as file:
    file.write(
        json.dumps({
            'metrics': prediction_output.metrics,
            'predictions': prediction_output.predictions.tolist(),
            'label_ids': prediction_output.label_ids.tolist(),
        }))