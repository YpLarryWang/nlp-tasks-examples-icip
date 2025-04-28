from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, HfArgumentParser, Trainer, BitsAndBytesConfig # <--- 导入 BitsAndBytesConfig
import torch
from dataclasses import dataclass, field
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training # <--- 导入 prepare_model_for_kbit_training

@dataclass
class CustomArguments:
    # 微调参数
    dataset_name_or_path: str = field(default="timdettmers/openassistant-guanaco")
    pretrained_model_name_or_path: str = field(default="Qwen/Qwen1.5-0.5B-Chat")
    padding_side: str = field(default="left")
    quantization_bit: int = field(default=None, metadata={"help": "Quantization bit (4 or 8). Default is None (no quantization)."}) # <--- 添加量化位数参数 (可选，但方便)

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
    custom_args, training_args = HfArgumentParser(
        (CustomArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    # --- 量化配置 ---
    quantization_config = None
    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 # 推荐使用 bfloat16 (如果硬件支持)
    if custom_args.quantization_bit == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # NF4 量化类型，推荐用于 LLM
            bnb_4bit_compute_dtype=model_dtype, # 计算时使用的类型 (bfloat16 or float16)
            bnb_4bit_use_double_quant=True,     # 使用双重量化节省更多显存
        )
        print("Loading model with 4-bit quantization (QLoRA)")
    elif custom_args.quantization_bit == 8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True, # 8-bit 不需要指定 compute_dtype, 它有自己的处理方式
            llm_int8_enable_fp32_cpu_offload=True,
        )
        print("Loading model with 8-bit quantization (QLoRA)")
    else:
        print(f"Loading model with dtype: {model_dtype}")
    # ---------------

    # 处理数据集 (代码不变)
    df = pd.read_json(custom_args.dataset_name_or_path, lines=True)
    ds = Dataset.from_pandas(df)

    tokenizer = AutoTokenizer.from_pretrained(custom_args.pretrained_model_name_or_path, use_fast=False, trust_remote_code=True, padding_side=custom_args.padding_side)
    # Qwen 特殊 token 处理，如果 tokenizer 没有 pad_token，则设置为 eos_token
    if tokenizer.pad_token is None:
        print("Set pad_token_id to eos_token_id as pad_token is None")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)

    # --- 修改模型加载 ---
    print(f"Loading model to CPU first with quantization_config: {quantization_config}") # 添加日志
    model = AutoModelForCausalLM.from_pretrained(
        custom_args.pretrained_model_name_or_path,
        quantization_config=quantization_config, # <--- 应用量化配置
        trust_remote_code=True,                 # <--- 对于 Qwen 可能需要
        torch_dtype=model_dtype if quantization_config is None else None # <--- 如果没量化，指定模型dtype
    )
    print(f"Model loaded on device: {model.device}") # 确认模型在 CPU 上
    # --------------------

    # 如果 tokenizer 没有 pad_token，模型可能也没有相应的 embedding，需要 resize
    # 这通常在加载 tokenizer 后、加载模型前或后完成
    # model.resize_token_embeddings(len(tokenizer)) # 如果 pad_token 是新加的，可能需要这步

    # --- 准备 QLoRA 模型 ---
    # 根据量化位数决定是否启用梯度检查点
    use_gradient_checkpointing = custom_args.quantization_bit == 4
    print(f"Gradient Checkpointing: {'Enabled' if use_gradient_checkpointing else 'Disabled'}")

    if quantization_config:
        # 为了稳定性和梯度检查点，对量化模型进行预处理
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
        # 如果启用了梯度检查点，必须执行 enable_input_require_grads
        # prepare_model_for_kbit_training 应该已经处理了梯度检查点的输入梯度设置
        # 重新启用梯度检查点后，我们还是先信任 prepare_model_for_kbit_training，不显式调用 enable_input_require_grads
        # if use_gradient_checkpointing:
        #      model.enable_input_require_grads()
    # --------------------

    print(f"Model loaded. Dtype: {model.dtype}") # 打印实际加载后的数据类型 (可能是 torch.int8)

    # --- PEFT 配置 (基本不变，但 target_modules 可能需要确认) ---
    # 对于 Qwen，这些 target_modules 通常是正确的，也可以尝试让 PEFT 自动寻找
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # 对 Qwen 应该适用
        # 或者尝试自动寻找: target_modules=None, # 让 peft 自动寻找 Lora 目标模块 (某些模型需要)
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    print(peft_config)

    model = get_peft_model(model, peft_config) # <--- 应用 PEFT

    print(model.print_trainable_parameters()) # 应该只显示 LoRA 参数是可训练的

    output_subdir = f"{training_args.output_dir}/quant{custom_args.quantization_bit}_bsz{training_args.per_device_train_batch_size}_gradacc{training_args.gradient_accumulation_steps}_lr{training_args.learning_rate:.1e}"

    # --- 更明确地设置 bf16/fp16 --- 
    bf16_setting = False
    fp16_setting = False
    if custom_args.quantization_bit == 4:
        # 4-bit: 依赖于 compute_dtype (即 model_dtype)
        if model_dtype == torch.bfloat16:
            bf16_setting = True
        elif model_dtype == torch.float16:
            fp16_setting = True
    elif custom_args.quantization_bit is None:
        # 非量化: 依赖于 model_dtype
        if model_dtype == torch.bfloat16:
            bf16_setting = True
        elif model_dtype == torch.float16:
            fp16_setting = True
    # 对于 8-bit 量化 (quantization_bit == 8), bf16_setting 和 fp16_setting 保持 False
    # -----------------------------------

    args = TrainingArguments(
        output_dir=output_subdir,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        logging_steps=training_args.logging_steps,
        num_train_epochs=training_args.num_train_epochs,
        save_steps=training_args.save_steps, # 注意 save_strategy="steps" 时才生效
        learning_rate=training_args.learning_rate,
        save_strategy=training_args.save_strategy,
        save_total_limit=training_args.save_total_limit,
        gradient_checkpointing=use_gradient_checkpointing, # <--- 使用条件变量
        optim="paged_adamw_32bit" if quantization_config else "adamw_torch", # <--- QLoRA 推荐使用 paged optimizer 节省显存
        # 使用上面计算好的设置
        bf16=bf16_setting,
        fp16=fp16_setting,
        ddp_find_unused_parameters=False, # <--- 在使用 DDP 和 gradient checkpointing 时可能需要
    )
    # ----------------------------

    # 使用trainer训练 (代码不变)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train() # 开始训练

    # 显式销毁进程组 (代码不变)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
