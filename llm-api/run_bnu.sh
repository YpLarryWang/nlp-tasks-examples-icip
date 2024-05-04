#!/bin/zsh

# bnu_request_file="llm-api/requests/claude_example_bnu.jsonl"
# bnu_save_file="llm-api/results/claude_example_bnu.jsonl"
bnu_request_file="llm-api/requests/openai_example_bnu.jsonl"
bnu_save_file="llm-api/results/openai_example_bnu.jsonl"
bnu_url="http://ICIP_IP_ADDRESS:PORT/claude"
bnu_api_key="your_bnu_username_here"

python llm-api/api_request_parallel_processor_0504.py \
--requests_filepath "${bnu_request_file}" \
--save_filepath "${bnu_save_file}" \
--request_url "${bnu_url}" \
--api_key "${bnu_api_key}" \
--max_requests_per_minute 50 \
--max_tokens_per_minute 10000 \
--seconds_to_sleep_each_loop 0.05 \
--token_encoding_name cl100k_base \
--max_attempts 5 \
--logging_level 20