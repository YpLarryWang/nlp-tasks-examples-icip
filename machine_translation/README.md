# 文白机器翻译

## 参考资料：
参考huggingface上seq2seq的文档、教程、示例：
- 序列到序列（seq2seq）模型：https://huggingface.co/learn/nlp-course/chapter1/7
- BART模型官方文档：https://huggingface.co/docs/transformers/model_doc/bart
- T5模型官方文档：https://huggingface.co/docs/transformers/model_doc/t5
- 使用T5进行机器翻译（colab notebook）：https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation.ipynb
- 使用T5进行文本摘要（colab notebook）：https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization.ipynb

## 数据来源

- 小牛翻译：https://github.com/NiuTrans/Classical-Modern

## 可用模型及其位置

**建议**：当一个模型可以同时在huggingface和modelscope上下载时，且确认modelscope版本和huggingface版本相同时，**优先选择modelscope上的下载链接**，因为modelscope对网络环境的要求更低，陆上玩家往往可以跑满带宽。

- `fnlp/bart-base-chinese`：可以[从huggingface上下载](https://huggingface.co/fnlp/bart-base-chinese)，建议用于文白机器翻译；
- `Langboat/mengzi-t5-base`：可以[从huggingface上下载](https://huggingface.co/Langboat/mengzi-t5-base)，也可以[从modelscope上下载](https://www.modelscope.cn/models/langboat/mengzi-t5-base/summary)，不建议用于文白机器翻译；
- `uer/t5-base-chinese-cluecorpussmall`：可以[从huggingface上下载](https://huggingface.co/uer/t5-base-chinese-cluecorpussmall)，非常不建议用于文白机器翻译；

上面三个模型均在133服务器的`/media/disk2/public`下存有压缩包。

此外，还有一些支持中文但没有经过测试的模型，同学们也可以自行选择来进行实验。

- `uer/bart-base-chinese-cluecorpussmall`：可以[从huggingface上下载](https://huggingface.co/uer/bart-base-chinese-cluecorpussmall)；
- `google/mt5-base`：可以[从huggingface上下载](https://huggingface.co/google/mt5-base)；
- `iic/nlp_mt5_zero-shot-augment_chinese-base`：全任务零样本学习-mT5分类增强版-中文-base，可以[从modelscope上下载](https://modelscope.cn/models/iic/nlp_mt5_zero-shot-augment_chinese-base/summary)；

## 安装库
### 必须安装的库
```
pip install transformers
pip install accelerate
pip install datasets
pip install sacrebleu
```

### 可选的库
```
pip install evaluate
```
如果要使用`mengzi-t5-base`，需要下载下面的包
```
pip install protobuf
```

## 设置多卡并行环境
```
accelerate config
```

## 多卡训练
```
accelerate launch --num_processes 2 seq2seq_translation_finetuning.py --config config_atm_trans_mengzi-t5-base.yaml
```
参考资料如下（在右侧栏点击“特定GPU选择”）
- https://huggingface.co/docs/transformers/main/zh/main_classes/trainer


## 可能存在的问题

- 预训练数据/任务构造
- 微调数据构造
- 历史背景与文化常识
- 评估指标：参考更多指标如rouge：https://zhuanlan.zhihu.com/p/504279252