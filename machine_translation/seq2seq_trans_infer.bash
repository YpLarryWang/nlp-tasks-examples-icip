python seq2seq_translation_inference.py \
--ckpt-dir path/to/checkpoint-xxxx \
--test-dir path/to/test.csv \
--max-input-length 382 \
--max-target-length 382 \
--max-new-tokens 382 \
--output-dir path/to/results \
--subset-size 512