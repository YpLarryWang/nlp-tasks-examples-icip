# 参考资料：
参考huggingface和datawhale的文档、教程、示例：
- Qwen1.5使用指南：https://github.com/datawhalechina/self-llm/tree/master/Qwen1.5
- 使用大型语言模型生成文本：https://huggingface.co/docs/transformers/v4.40.1/en/llm_tutorial#wrong-padding-side
- 使用huggingface的TRL库进行监督微调和强化学习：https://huggingface.co/docs/trl/sft_trainer
- 使用llama-factory对LLM进行预训练、监督微调、强化学习（Colab笔记本）：https://colab.research.google.com/drive/1d5KQtbemerlSDSxZIfAaWXhKr30QypiK?usp=sharing#scrollTo=psywJyo75vt6
- 使用Ollama在本地部署大模型：https://ollama.com/

# 数据来源

- m-a-p/COIG-CQIA: https://huggingface.co/datasets/m-a-p/COIG-CQIA，在`ruozhiba`目录下
- 论文：COIG-CQIA: Quality is All You Need for Chinese Instruction Fine-tuning

# 安装库
## 必须安装的库
```
pip install transformers
pip install accelerate
pip install datasets
```

## 可选的库
```
pip install trl
```
如果要使用`tensorboard`，可下载下面的包
```
pip install tensorboard
```

# 设置多卡并行环境
```
accelerate config
```

# 多卡训练
```
accelerate launch --num_processes 2 seq2seq_translation_finetuning.py --config config_atm_trans_mengzi-t5-base.yaml
```
参考资料如下（在右侧栏点击“特定GPU选择”）
- https://huggingface.co/docs/transformers/main/zh/main_classes/trainer


# 可能存在的问题

- hugginfgace总结了[文本生成时常见的坑](https://huggingface.co/docs/transformers/v4.40.1/en/llm_tutorial#common-pitfalls)