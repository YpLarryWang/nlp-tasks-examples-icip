import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import pandas as pd # Using pandas to easily read various file formats if needed

def main(args):
    # --- 量化配置 ---
    quantization_config = None
    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 # 选择合适的dtype
    if args.quantization_bit == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_dtype,
            bnb_4bit_use_double_quant=True,
        )
        print("Loading base model with 4-bit quantization...")
    elif args.quantization_bit == 8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        print("Loading base model with 8-bit quantization...")
    else:
        print(f"Loading base model with dtype: {model_dtype} (no quantization)")
    # ---------------

    # 加载tokenizer
    # 注意：推理时通常也将 padding_side 设为 'left'，与训练匹配且便于生成
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("Setting pad_token_id to eos_token_id")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载基础模型 (应用量化配置)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        quantization_config=quantization_config,
        device_map="auto", # 自动将模型分发到可用设备
        torch_dtype=model_dtype if quantization_config is None else None, # 只有非量化时指定全局dtype
        trust_remote_code=True
    )
    print(f"Base model loaded. Dtype: {model.dtype}")

    # 加载LoRA权重 (如果提供了路径)
    if args.lora_path:
        print(f"Loading LoRA weights from: {args.lora_path}")
        model = PeftModel.from_pretrained(model, model_id=args.lora_path)
        print("LoRA weights loaded.")
    else:
        print("No LoRA path provided, using the base model directly.")

    model.eval() # 设置为评估模式

    # 读取问题
    try:
        # 尝试读取 txt 文件
        with open(args.question_file, "r", encoding='utf-8') as f:
            questions = [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e_txt:
        print(f"Failed to read as txt: {e_txt}. Trying other formats with pandas...")
        try:
            # 尝试读取其他常见格式 (csv, jsonl, excel)
            if args.question_file.endswith('.csv'):
                df = pd.read_csv(args.question_file)
            elif args.question_file.endswith('.jsonl'):
                df = pd.read_json(args.question_file, lines=True)
            elif args.question_file.endswith('.xlsx'):
                df = pd.read_excel(args.question_file)
            else:
                raise ValueError("Unsupported file format. Please use txt, csv, jsonl, or xlsx.")
            # 假设问题在名为 'prompt' 或 'question' 的列，或者第一列
            if 'prompt' in df.columns:
                questions = df['prompt'].astype(str).tolist()
            elif 'question' in df.columns:
                questions = df['question'].astype(str).tolist()
            else:
                questions = df.iloc[:, 0].astype(str).tolist()
        except Exception as e_pandas:
            print(f"Failed to read with pandas: {e_pandas}")
            print("Please ensure the input file is a valid txt, csv, jsonl, or xlsx file.")
            return

    print(f"Loaded {len(questions)} questions.")

    # 准备输入
    input_list = []
    for question in questions:
        # 使用与训练时类似的模板，确保 system prompt 等一致性
        message = [
            {"role": "system", "content": "你是一位善意的助手."}, # 根据你的训练调整
            {"role": "user", "content": question}
        ]
        # tokenize=False 先获取完整文本，再批量 tokenize
        text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        input_list.append(text)

    # 批量 tokenize
    # padding=True 很重要，因为输入序列长度不同
    model_inputs = tokenizer(input_list, return_tensors="pt", padding=True, truncation=True, max_length=args.max_input_length).to(model.device) # 移动到模型所在设备

    print("Generating responses...")
    # 生成回复
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=args.max_new_tokens,
        # 可选：添加其他生成参数，如 do_sample, temperature, top_p 等
        # do_sample=True,
        # temperature=0.7,
        # top_p=0.9,
    )

    # 解码回复 (去除输入部分)
    output_ids = generated_ids[:, model_inputs.input_ids.shape[1]:] # 只取生成的新 token
    responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    print("\n--- Results ---")
    # 打印结果
    for q, a in zip(questions, responses):
        print(f"\nQuestion: {q}")
        print(f"Answer: {a}")
        print("-" * 10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with QLoRA quantized models.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base pre-trained model.")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to the LoRA adapter weights (checkpoint directory).")
    parser.add_argument("--quantization_bit", type=int, choices=[4, 8], default=None, help="Quantization bit (4 or 8). Default: No quantization.")
    parser.add_argument("--question_file", type=str, required=True, help="Path to the file containing questions (one per line for txt, or 'prompt'/'question'/first column for csv/jsonl/xlsx).")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate.")
    parser.add_argument("--max_input_length", type=int, default=1024, help="Maximum input length for truncation.")

    args = parser.parse_args()
    main(args) 