# 运行 4-bit QLoRA
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 sft_qwen_quantization.py \
    --dataset_name_or_path data/ruozhiba.jsonl \
    --pretrained_model_name_or_path "/path/to/your/base/qwen_model" \
    --output_dir="output/ruozhiba_qlora4" \
    --padding_side="left" \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-4 \
    --num_train_epochs=3 \
    --logging_steps=10 \
    --save_strategy="epoch" \
    --save_total_limit=1 \
    --quantization_bit=4