# 通过异步调用API服务来使用大型语言模型

在通过OpenAI等大模型厂商提供的API服务来使用大型语言模型时，可以通过异步（或非阻塞）方式调用多个API，这样可以在不必等待每个单独请求返回的情况下，提高处理速度和应用程序的响应能力。OpenAI提供了一个简单的并行调用API的[示例](https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py)。但其他大模型服务厂商，如Aliyun，则并没有提供相应的示例。事实上，通过`curl`调用各家的API服务时，所用的参数非常相似，因为只需对OpenAI的示例进行一定的改写就可以实现对其他厂商的API的异步调用。

在脚本（`api_request_parallel_processor_openai_qwen.py`）中，只需要将密钥和URL修改为需要的值，且注意将输入的数据的格式调整为目标服务要求的格式即可实现异步调用API。该脚本修改自OpenAI的示例，但增加了对一些其他API服务的支持，同时添加了参数`seconds_to_sleep_each_loop`，用于控制每个API请求之间的时间间隔。OpenAI的API服务没有秒级的调用频率限制（使用时可设为0.01或忽略），但其他API服务有秒级的调用频率限制，因此需要设置该参数（比如Aliyun就有秒级限制，虽然没有在官网明说，使用Qwen时需要摸索尝试一下，先设置0.15-0.5为宜）。

目前，该脚本支持的API服务有：
- OpenAI: chat completion
- Aliyun: text generation
  
后续，我们将持续更新，使该脚本能够支持更多API服务，包括但不限于，MistraAI、Microsoft Azure、Google Cloud、ZhipuAI、无问芯穹等。欢迎同学们提交PR，和我们一起维护这个脚本。

调用方法如下：

```bash
sh llm-api/run.sh
```

这里，`run.sh`是一个命令行脚本（command-line scripts）。命令行脚本是一种由命令行解释器（如bash或zsh）执行的脚本文件，用于自动化执行一系列命令或操作。如果你要在linux机器上使用该命令行脚本，一般需要修改[shebang](https://zh.wikipedia.org/wiki/Shebang)为`#!/bin/bash`。

其他资料：
- 视频：[OpenAI开发者大会：如何最大限度优化大型语言模型的表现](https://www.youtube.com/watch?v=ahnGLM-RC1Y)