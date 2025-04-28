from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, HfArgumentParser, Trainer
import torch
from dataclasses import dataclass, field
from peft import LoraConfig, TaskType, get_peft_model


@dataclass
class CustomArguments:
    # 微调参数
    # field：dataclass 函数，用于指定变量初始化
    dataset_name_or_path: str = field(default="timdettmers/openassistant-guanaco")
    pretrained_model_name_or_path: str = field(default="Qwen/Qwen1.5-0.5B-Chat")
    padding_side: str = field(default="left")

# 用于处理数据集的函数
def process_func(example):
    MAX_LENGTH = 128
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|im_start|>system\n你是一位善意的助手.<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer("<|im_start|>assistant\n" + example["output"] + "<|im_end|>\n", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  # Qwen的特殊构造就是这样的
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


if "__main__" == __name__:
    # 解析参数
    # Parse 命令行参数
    custom_args, training_args = HfArgumentParser(
        (CustomArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    # 处理数据集
    # 将JSON文件转换为CSV文件
    df = pd.read_json(custom_args.dataset_name_or_path, lines=True)
    ds = Dataset.from_pandas(df)
    # 加载tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(custom_args.pretrained_model_name_or_path, use_fast=False, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(custom_args.pretrained_model_name_or_path, use_fast=False, trust_remote_code=True, padding_side=custom_args.padding_side)

    # 将数据集变化为token形式
    tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)

    # model = AutoModelForCausalLM.from_pretrained(custom_args.pretrained_model_name_or_path, device_map="auto",torch_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(custom_args.pretrained_model_name_or_path)

    # model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
    
    print(model.dtype)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False, # 训练模式
        r=8, # Lora 秩
        lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1# Dropout 比例
    )
    print(peft_config)
    
    model = get_peft_model(model, peft_config)
    
    print(model.print_trainable_parameters())
    
    output_subdir = f"{training_args.output_dir}/bsz{training_args.per_device_train_batch_size}_gradacc{training_args.gradient_accumulation_steps}_lr{training_args.learning_rate:.1e}"
    
    args = TrainingArguments(
        output_dir=output_subdir,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        logging_steps=training_args.logging_steps,
        num_train_epochs=training_args.num_train_epochs,
        save_steps=training_args.save_steps,
        learning_rate=training_args.learning_rate,
        save_strategy=training_args.save_strategy,
        save_total_limit=training_args.save_total_limit,
        # gradient_checkpointing=True
    )

    # 使用trainer训练
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    
    trainer.train() # 开始训练
    