#!/bin/zsh

openai_chat_request_file="llm-api/requests/openai_chat_example.jsonl"
openai_chat_save_file="llm-api/results/openai_chat_example.jsonl"

openai_embedding_request_file="llm-api/requests/openai_embedding_example.jsonl"
openai_embedding_save_file="llm-api/results/openai_embedding_example.jsonl"

openai_chat_url="https://api.openai.com/v1/chat/completions"
openai_embedding_url="https://api.openai.com/v1/embeddings"
my_openai_api_key="your_openai_api_key_here"

aliyun_request_file="llm-api/requests/qwen_chat_example.jsonl"
aliyun_save_file="llm-api/results/qwen_chat_example.jsonl"

aliyun_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
my_aliyun_api_key="your_aliyun_api_key_here"

# deepinfra_request_file="llm-api/requests/llama3_70b_example_deepinfra.jsonl"
# deepinfra_save_file="llm-api/results/llama3_70b_example_deepinfra.jsonl"
deepinfra_request_file="llm-api/requests/mistral8x22b_example_deepinfra.jsonl"
deepinfra_save_file="llm-api/results/mistral8x22b_example_deepinfra.jsonl"
# deepinfra_request_file="llm-api/requests/wizardllm8x22b_example_deepinfra.jsonl"
# deepinfra_save_file="llm-api/results/wizardllm8x22b_example_deepinfra.jsonl"

deepinfra_url="https://api.deepinfra.com/v1/openai/chat/completions"
my_deepinfra_api_key="your_deepinfra_api_key_here"

deepseek_request_file="llm-api/requests/deepseek_chat_example.jsonl"
deepseek_save_file="llm-api/results/deepseek_chat_example.jsonl"

deepseek_url="https://api.deepseek.com/chat/completions"
my_deepseek_api_key="your_deepseek_api_key_here"

python llm-api/api_request_parallel_processor.py \
--requests_filepath "${deepinfra_request_file}" \
--save_filepath "${deepinfra_save_file}" \
--request_url "${deepinfra_url}" \
--api_key "${my_deepinfra_api_key}" \
--max_requests_per_minute 300 \
--max_tokens_per_minute 300000 \
--seconds_to_sleep_each_loop 0.05 \
--token_encoding_name cl100k_base \
--max_attempts 5 \
--logging_level 20