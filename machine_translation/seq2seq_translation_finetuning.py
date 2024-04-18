import argparse
import yaml

import numpy as np
from datasets import (Dataset, 
                      DatasetDict, 
                      concatenate_datasets,)
from transformers import (set_seed,
                          AutoTokenizer,
                          DataCollatorForSeq2Seq,
                          AutoModelForSeq2SeqLM,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer,
                          EarlyStoppingCallback,)
from sacrebleu import corpus_bleu
# import warnings

# warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
args = parser.parse_args()

# 读取YAML文件
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)
    
# # 遍历配置字典，打印参数及其数据类型
# for key, value in config.items():
#     print(f"Parameter: {key}, Value: {value}, Type: {type(value).__name__}")

# 发现科学计数法形式的学习率无法被自动解析为浮点数，所以需要额外转化一下
config['learning_rate'] = eval(config['learning_rate'])

# 设置随机种子来保证代码可复现
set_seed(config['seed'])

train_data = Dataset.from_csv(f'''{config['input_dir']}/train.csv''')
val_data = Dataset.from_csv(f'''{config['input_dir']}/valid.csv''')
test_data = Dataset.from_csv(f'''{config['input_dir']}/test.csv''')


dataset = DatasetDict({
    'train': train_data,
    'valid': val_data.select(range(512)),
    'test': test_data.select(range(512)),
})

if config['debug']:
    dataset = DatasetDict({
        'train': train_data.select(range(32)),
        'valid': val_data.select(range(32)),
        'test': test_data.select(range(32))
    })

    config['trainer_logging_steps'] = 10
    config['trainer_save_steps'] = 10
    config['trainer_eval_steps'] = 10
    
    config['max_input_length'] = 320
    config['max_target_length'] = 320

# print(dataset['valid'])
# exit()

# Load tokenizer of BART
tokenizer = AutoTokenizer.from_pretrained(config['model_dir'])

# concatenated_dataset = concatenate_datasets([dataset['train'], dataset['valid'], dataset['test']])

# # The maximum total input sequence length after tokenization.
# # Sequences longer than this will be truncated, sequences shorter will be padded.
# tokenized_inputs = concatenated_dataset.map(
#     lambda x: tokenizer(x["Classical"], truncation=True), batched=True, remove_columns=["Classical", "Modern", "Source"])
# input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
# # take 85 percentile of max length for better utilization
# # max_source_length = int(np.percentile(input_lenghts, 85))
# max_source_length = max(input_lenghts)
# print(f"古文最大长度: {max_source_length}")

# # The maximum total sequence length for target text after tokenization.
# # Sequences longer than this will be truncated, sequences shorter will be padded."
# tokenized_targets = concatenated_dataset.map(
#     lambda x: tokenizer(x["Modern"], truncation=True), batched=True, remove_columns=["Classical", "Modern", "Source"])
# target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
# # take 90 percentile of max length for better utilization
# max_target_length = max(target_lenghts)
# print(f"现代文最大长度: {max_target_length}")

data_collator = DataCollatorForSeq2Seq(tokenizer)

# prefix = "将古代汉语翻译为现代汉语: "
prefix = ""

def preprocess_function(sample):
    
    inputs = [prefix + text for text in sample["Classical"]]
    targets = [text for text in sample["Modern"]]

    # tokenize inputs
    model_inputs = tokenizer(
        inputs, max_length=config['max_input_length'], padding="max_length", truncation=True)
    
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=config['max_target_length'], truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
        
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    print(decoded_preds)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    bleu_score = corpus_bleu(hypotheses=decoded_preds, references=decoded_labels)
    
    # print(type(bleu_score))
    # print(bleu_score)
    
    result = {"bleu": bleu_score.score}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    # result = {k: round(v, 4) for k, v in result.items()}
    return result


tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=[
                                "Classical", "Modern", "Source"])

def model_init():
    model = AutoModelForSeq2SeqLM.from_pretrained(config['model_dir'], trust_remote_code=True)

    # 新版本的`transformers`库将不再支持通过在模型配置中包含`max_length`属性来控制文本生成模型生成文本的长度
    # 所以需要通过添加`max_new_tokens`来设置（新）生成文本的最大token数，否则会出现警告或报错
    model.generation_config.max_new_tokens = config['max_new_tokens']

    # print(model.generation_config)

    return model

# print(model.config.max_length)
# print(model.config.max_new_tokens)
# exit()

"""
predict_with_generate 是一个布尔参数。当设置为 True 时，意味着在评估（validation）和测试（test）阶段，模型将使用生成式方法来预测输出序列。也就是说，在评价模型性能时，模型不仅仅会计算损失函数（通常是交叉熵损失），而是会实际生成整个序列，这样可以使用诸如 BLEU、ROUGE 或其他针对完整序列生成的评估指标。
"""

training_args = Seq2SeqTrainingArguments(
    output_dir=config['output_dir'],
    learning_rate=config['learning_rate'],
    per_device_train_batch_size=config['per_device_train_batch_size'],
    per_device_eval_batch_size=config['per_device_eval_batch_size'],
    num_train_epochs=config['num_train_epochs'],
    weight_decay=config['weight_decay'],
    evaluation_strategy=config['evaluation_strategy'],
    save_strategy=config['save_strategy'],
    logging_steps=config['trainer_logging_steps'],
    save_steps=config['trainer_save_steps'],
    eval_steps=config['trainer_eval_steps'],
    warmup_steps=config['warmup_steps'],
    save_total_limit=config['save_total_limit'],
    logging_dir=config['logging_dir'],
    metric_for_best_model=config['metric_for_best_model'],
    load_best_model_at_end=True,
    predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    # model=model,
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=config['early_stopping_patience'])
    ],
)

trainer.train()
