"""
本脚本参考huggingface教程：https://huggingface.co/docs/transformers/tasks/sequence_classification
"""

import argparse
import os
import yaml
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from transformers import (set_seed,
                          AutoTokenizer, 
                          DataCollatorWithPadding,
                          AutoModelForSequenceClassification, 
                          TrainingArguments, 
                          Trainer,
                          EarlyStoppingCallback,)

# # 直接使用huggingface提供的方法导入imdb数据集
# from datasets import load_dataset
# imdb = load_dataset("imdb")  # 无科学情况下时而可以时而不行

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
args = parser.parse_args()

# 读取YAML文件
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

# 遍历配置字典，打印参数及其数据类型
# for key, value in config.items():
#     print(f"Parameter: {key}, Value: {value}, Type: {type(value).__name__}")
# exit()
    
# 发现科学计数法形式的学习率无法被自动解析为浮点数，所以需要额外转化一下
config['learning_rate'] = eval(config['learning_rate'])
    
# 设置随机种子来保证代码可复现
set_seed(config['seed'])

# 如果输出目录不存在，就创建这个目录，以防因为目录不存在而报错
# 如果目录存在，就什么都不做
os.makedirs(config['output_dir'], exist_ok=True)

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

# 使用函数并为训练集和测试集提供相应的顶层文件夹路径
imdb_train_dataset = load_dataset(config['input_dir'])
# print(type(imdb_train_dataset))  # <class 'datasets.arrow_dataset.Dataset'>
# exit()

# 切分数据集为训练集和验证集，这里以80-20的比例为例
# 注意这里的train_test_split是<class 'datasets.arrow_dataset.Dataset'>对象采用的方法
train_test_split = imdb_train_dataset.train_test_split(test_size=0.2)

# 分别获取切分后的训练集和验证集
train_set = train_test_split['train']
validation_set = train_test_split['test']

# 如果你需要一个包含所有分割的数据集字典，可以这样做
imdb = DatasetDict({
    'train': train_test_split['train'],
    'valid': train_test_split['test']

})

# 尝试打印出一个例子
print(f'''\nBelow is an example from test set：\n{"-" * 20}\n{imdb["valid"][0]}\n{"-" * 20}\n''')

# exit()

##########################################################
######################## 预处理 ###########################
##########################################################
tokenizer = AutoTokenizer.from_pretrained(config['model_dir'])

# =================================================
# ============ 首先统计一下token数量的分布 ============
# =================================================

# # 定义一个函数来统计tokens
# def count_tokens(text):
#     # 使用encode方法tokenize文本，不添加特殊符号
#     tokens = tokenizer.encode(text, add_special_tokens=False)
#     # 返回tokens的数量
#     return len(tokens)

# # 对数据集中的每个文本样本应用count_tokens函数并收集统计信息
# lengths = [count_tokens(entry['text']) for entry in imdb['train']]

# # 计算统计信息
# min_len = np.min(lengths)
# max_len = np.max(lengths)
# mean_len = np.mean(lengths)
# median_len = np.median(lengths)
# percentile_25 = np.percentile(lengths, 25)
# percentile_75 = np.percentile(lengths, 75)

# # 打印统计信息
# print("Token Counts Statistics:")
# print(f"Minimum token count: {min_len}")
# print(f"Maximum token count: {max_len}")
# print(f"Mean token count: {mean_len:.2f}")
# print(f"Median token count: {median_len}")
# print(f"25th percentile: {percentile_25}")
# print(f"75th percentile: {percentile_75}")
# print()
# exit()

# =================================================

## 正式定义编码函数，实现补全、截断、添加特殊符号([CLS], [SEP])等更复杂的功能

# 编写一个用于将文本转化为词元（token，这个过程叫做tokenize）的预处理函数
def preprocess_function(examples):
    # tokenizer参数请参考：https://huggingface.co/docs/transformers/main_classes/tokenizer
    return tokenizer(examples["text"], 
                     max_length=512, 
                    #  padding="max_length",
                     truncation=True,)

# 利用函数`preprocess_function`对所有文本进行tokenization，通过设置batched=True加快处理速度（一次处理多条数据）
tokenized_imdb = imdb.map(preprocess_function, batched=True, batch_size=32)

# 实例化一个data collator，可以动态地在每一个batch内将文本padding到相同长度，而不必总padding到512
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

##########################################################
##################### 定义评估指标 #########################
##########################################################

# # 使用huggingface的evaluate库导入，无科学会很慢
# import evaluate
# accuracy = evaluate.load("accuracy")

# 定义评价指标，在文本分类任务中是准确率
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    correct = (predictions == labels).sum()
    total = len(labels)
    accuracy = correct / total
    return {"accuracy": accuracy}
    # return accuracy.compute(predictions=predictions, references=labels)  # 如果用了evaluate就需要这样写

##########################################################
######################## 训 练 ###########################
##########################################################

# 开始训练前确定标签和id的映射关系
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# # 模型实例化
# model = AutoModelForSequenceClassification.from_pretrained(
#     model_dir, num_labels=2, id2label=id2label, label2id=label2id
# )  # 这样写的话模型初始化参数无法被随机种子控制，科研中不建议采用

# 定义模型初始化函数，然后将这个函数作为参数传递给Trainer
# 这种方法初始化的模型的参数在初始化时会被随机种子控制，方便复现实验结果
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
    config['model_dir'], num_labels=2, id2label=id2label, label2id=label2id
)
    



# 设置训练参数
training_args = TrainingArguments(
    output_dir=config['output_dir'],
    learning_rate=config['learning_rate'],
    per_device_train_batch_size=config['per_device_train_batch_size'],
    per_device_eval_batch_size=config['per_device_train_batch_size'],
    num_train_epochs=config['num_train_epochs'],
    weight_decay=config['weight_decay'],
    evaluation_strategy=config['evaluation_strategy'],
    save_strategy=config['save_strategy'],
    logging_steps=config['trainer_logging_steps'],
    save_steps=config['trainer_save_steps'],
    eval_steps=config['trainer_eval_steps'],
    warmup_steps=config['warmup_steps'],
    save_total_limit=config['save_total_limit'],
    load_best_model_at_end=True,
    logging_dir=config['logging_dir'],
    metric_for_best_model=config['metric_for_best_model'],
)

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=config['early_stopping_patience'])
    ],
)

trainer.train()