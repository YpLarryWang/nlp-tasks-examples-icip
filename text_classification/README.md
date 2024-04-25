# 基于Hugging Face的transformers库的文本分类任务

## 参考资料
- huggingface文本分类教程：https://huggingface.co/docs/transformers/tasks/sequence_classification
- huggingface文本分类示例代码（使用DistilBERT）：https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/pytorch/sequence_classification.ipynb
- huggingface BERT使用指南：https://huggingface.co/docs/transformers/model_doc/bert

## 数据来源

- **IMDB**：可以从该项目的[官方网站](https://ai.stanford.edu/~amaas/data/sentiment/)下载数据，也可以从huggingface上下载数据(https://huggingface.co/datasets/stanfordnlp/imdb).

## 安装库
### 必须安装的库
```bash
pip install torch
pip install transformers
pip install accelerate
pip install datasets
```

### 可选的库
```bash
pip install evaluate
```

## 模型

`bert-base-uncased`是一个预训练模型，可以用于文本分类任务。位置：`/media/disk2/public/bert-base-uncased`

## 设置多卡并行环境
```bash
accelerate config
```

## 多卡训练
```bash
accelerate launch --num_processes 2 sequence_classification_finetuning.py --config config_seq_cls.yaml
```

了解多卡训练参考如下资料（注意右侧栏选择“特定GPU选择”）
- https://huggingface.co/docs/transformers/main/zh/main_classes/trainer

## 推理

```bash
bash text_classification/seq_cls_infer.bash
```