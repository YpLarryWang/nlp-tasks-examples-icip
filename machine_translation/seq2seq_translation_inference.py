import argparse
import torch

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (AutoTokenizer,
                          DataCollatorForSeq2Seq,
                          AutoModelForSeq2SeqLM,
                          Seq2SeqTrainer,)
from sacrebleu import corpus_bleu

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt-dir', required=True)
parser.add_argument('--test-dir', required=True)
parser.add_argument('--max-input-length', required=True, type=int)
parser.add_argument('--max-target-length', required=True, type=int)
parser.add_argument('--max-new-tokens', required=True, type=int)
parser.add_argument('--output-dir', required=True)
parser.add_argument('--subset-size', required=True, type=int)
args = parser.parse_args()

for arg in vars(args):
    value = getattr(args, arg)
    print(f'{arg}: {value} ({type(value).__name__})')

test_data = Dataset.from_csv(args.test_dir)
test_data = test_data.select(range(args.subset_size))

# Load tokenizer of BART
# tokenizer = BertTokenizer.from_pretrained(args.ckpt_dir)
tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir)

data_collator = DataCollatorForSeq2Seq(tokenizer)

# prefix = "将古代汉语翻译为现代汉语: "
prefix = ""

def preprocess_function(sample):
    
    inputs = [prefix + text for text in sample["Classical"]]
    targets = [text for text in sample["Modern"]]

    # tokenize inputs
    model_inputs = tokenizer(
        inputs, max_length=args.max_input_length, padding="max_length", truncation=True)
    
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=args.max_target_length, truncation=True)

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

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    bleu_score = corpus_bleu(hypotheses=decoded_preds, references=decoded_labels)
    
    result = {"bleu": bleu_score.score}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    # result = {k: round(v, 4) for k, v in result.items()}
    return result


tokenized_test_data = test_data.map(preprocess_function, batched=True, remove_columns=[
                                "Classical", "Modern", "Source"])

model = AutoModelForSeq2SeqLM.from_pretrained(args.ckpt_dir, trust_remote_code=True)

# 现在模型配置中不再有max_length属性，可以按需设置其他参数
model.generation_config.max_new_tokens = args.max_new_tokens  # 例如设置生成的新token的最大数量

"""
predict_with_generate 是一个布尔参数。当设置为 True 时，意味着在评估（validation）和测试（test）阶段，模型将使用生成式方法来预测输出序列。也就是说，在评价模型性能时，模型不仅仅会计算损失函数（通常是交叉熵损失），而是会实际生成整个序列，这样可以使用诸如 BLEU、ROUGE 或其他针对完整序列生成的评估指标。
"""

# 加载 TrainingArguments 实例
training_args = torch.load(f"{args.ckpt_dir}/training_args.bin")

# 使用自定义的生成方法
# class CustomTrainer(Seq2SeqTrainer):
#     def generate(self, input_ids, **generate_kwargs):
#         return super().generate(input_ids, max_new_tokens=50, **generate_kwargs)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 在测试数据集上进行推理
prediction_output = trainer.predict(tokenized_test_data)

print(prediction_output.metrics)

results = {
    '人类翻译': [],
    '模型翻译': []
}

for preds, labels in zip(prediction_output.predictions, prediction_output.label_ids):
    
    # 这里preds和labels都是<class 'numpy.ndarray'>
    
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # print(tokenizer.batch_decode(preds, skip_special_tokens=True))
    # print(tokenizer.batch_decode(labels, skip_special_tokens=True))
    
    results['人类翻译'].append(''.join(tokenizer.batch_decode(labels, skip_special_tokens=True)).strip())
    results['模型翻译'].append(''.join(tokenizer.batch_decode(preds, skip_special_tokens=True)).strip())


# Writing the prediction output to a csv file
with open(f"{args.output_dir}/bart_result_{args.subset_size}.csv", 'w') as file:
    file.write(f"古文,人类翻译,模型翻译\n")
    
    for anc, mod_human, mod_machine in zip(test_data['Classical'], results['人类翻译'], results['模型翻译']):
        file.write(f"{anc},{mod_human},{mod_machine}\n")
