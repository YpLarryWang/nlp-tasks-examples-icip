# 自然语言处理课程示例代码

本仓库包含北京师范大学数字人文系开设的自然语言处理课程的示例代码，也包括Python语言程序设计与数据分析课程的部分示例代码。

## 历年代码实例大纲

### 2025年

- [大语言模型API调用](llm-api/)
- PyTorch教程
  - [常见问题与解答](torch-memo/)
  - [PyTorch Tutorial (斯坦福CS224N原版notebook)](torch-memo/SP_24_CS224N_PyTorch_Tutorial.ipynb)
  - [PyTorch Tutorial (CS224N+备注版)](notebook/Revised_SP_24_CS224N_PyTorch_Tutorial.ipynb)
  - [使用MLP判断6位2进制数的对称性](notebook/detect_symmetry_mlp.ipynb)

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

**A**: 下载package可以尝试配置清华源，参考[官方指南](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)；模型下载建议优先考虑阿里云魔搭平台(ModelScope)，参考[官方指南](https://modelscope.cn/docs/models/download)。**此外，为了能够成功连接魔搭平台，需要在服务器上连接校园网，这一步非常重要，请参考兆基师兄开发的[连接助手](https://github.com/frederick-wang/bnu-cernet-cli)。按照说明在服务器上配置好就可以使用自己的校园网账号连接校园网了（注意连上以后就会用你的流量，所以用完记得登出）。**

**Q**: 使用conda安装PyTorch失败怎么办？

**A**: PyTorch已不再支持使用conda安装，请参考[官方指南](https://pytorch.org/get-started/locally/)使用pip安装。