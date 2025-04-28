import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

mode_path = '/media/disk2/public/Qwen1.5-0.5B-Chat'
lora_path = 'llm-sft/output/ruozhiba/bsz4_gradacc1_lr1.0e-04/checkpoint-90'
question_path = 'llm-sft/data/ruozhiba_test.txt'

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path, padding_side="left")

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto", torch_dtype=torch.bfloat16)

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

with open(question_path, "r") as f:
    questions = [line.strip() for line in f.readlines()]
    # print(questions)
    # exit()
    
input_list = []

for question in questions:
    message = [
        {"role": "system", "content": "你是一位善意的助手."},
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    input_list.append(text)

model_inputs = tokenizer(input_list, return_tensors="pt", padding=True, truncation=True).to('cuda')

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

for Q, A in zip(questions, responses):
    print(f"\n{Q=}\n{A=}\n")