#!/bin/zsh

openai_request_file="llm-api/requests/openai_chat_example.jsonl"
openai_save_file="llm-api/results/openai_chat_example.jsonl"

openai_url="https://api.openai.com/v1/chat/completions"
my_openai_api_key="your_openai_api_key_here"

aliyun_request_file="llm-api/requests/qwen_chat_example.jsonl"
aliyun_save_file="llm-api/results/qwen_chat_example.jsonl"

aliyun_url="https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
my_aliyun_api_key="your_aliyun_api_key_here"

# deepinfra_request_file="llm-api/requests/llama3_70b_example_deepinfra.jsonl"
# deepinfra_save_file="llm-api/results/llama3_70b_example_deepinfra.jsonl"
deepinfra_request_file="llm-api/requests/mistral8x22b_example_deepinfra.jsonl"
deepinfra_save_file="llm-api/results/mistral8x22b_example_deepinfra.jsonl"
# deepinfra_request_file="llm-api/requests/wizardllm8x22b_example_deepinfra.jsonl"
# deepinfra_save_file="llm-api/results/wizardllm8x22b_example_deepinfra.jsonl"

deepinfra_url="https://api.deepinfra.com/v1/openai/chat/completions"
my_deepinfra_api_key="your_deepinfra_api_key_here"

python llm-api/api_request_parallel_processor_0504.py \
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