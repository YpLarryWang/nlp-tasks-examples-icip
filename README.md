# 自然语言处理课程示例代码

本仓库包含北京师范大学数字人文系开设的自然语言处理课程的示例代码，也包括Python语言程序设计与数据分析课程的部分示例代码。

## 2025年示例代码

- [大语言模型API调用](llm-api/)
- PyTorch教程
  - [常见问题与解答](torch-memo/)
  - Stanford CS224N PyTorch Tutorial: [[原版]](torch-memo/SP_24_CS224N_PyTorch_Tutorial.ipynb),[[备注版]](notebook/Revised_SP_24_CS224N_PyTorch_Tutorial.ipynb)
  - [使用MLP判断6位2进制数的对称性](notebook/detect_symmetry_mlp.ipynb)
  - [使用TextCNN进行IMDB电影评论情感分类](notebook/SP2025_TEXT_CLS_CNN.ipynb)
  - [使用LSTM进行IMDB电影评论情感分类](notebook/SP25_LSTM_huggingface.ipynb)
  - [使用BERT进行IMDB电影评论情感分类并利用梯度积分做可解释性分析](notebook/SP25_imdb_bert_huggingface.ipynb)
  - [使用弱智吧数据微调Qwen2.5-0.5B-Instruct](llm-sft/)

## 往年示例代码

### 2024年

- 基于PyTorch使用CNN/LSTM进行imdb情感分类
- 基于HuggingFace的transformers库使用BERT进行imdb情感分类
- 基于HuggingFace的transformers库使用bart/t5进行文白机器翻译
- 大语言模型API调用
- 基于HuggingFace的transformers库在弱智吧数据集上对qwen系列模型进行监督微调（SFT）

### 2023年

- 使用全连接神经网络进行豆瓣书评情感二分类: `fc_torch_text_classification.ipynb`
- 使用预训练模型BERT进行豆瓣书评情感二分类: `douban-classification-bert.ipynb`


## FAQ

**Q**: 在服务器上下载package或模型太慢，甚至下载失败，怎么办？

**A**: 下载package可以尝试配置清华源，参考[官方指南](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)；模型下载建议优先考虑阿里云魔搭平台(ModelScope)，参考[官方指南](https://modelscope.cn/docs/models/download)。 **为了能够成功连接魔搭平台，需要在服务器上连接校园网，这一步非常重要，请参考兆基师兄开发的[连接助手](https://github.com/frederick-wang/bnu-cernet-cli)。** 按照说明在服务器上配置好就可以使用自己的校园网账号连接校园网了（ *注意连上以后就会用你的流量，所以用完记得登出* ）。需注意，有些模型无法在魔搭上找到，这时只能先下载到自己电脑上然后上传服务器。

**Q**: 使用conda安装PyTorch失败怎么办？

**A**: PyTorch已不再支持使用conda安装，请参考[官方指南](https://pytorch.org/get-started/locally/)使用pip安装。