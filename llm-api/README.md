在调用OpenAI或Aliyun提供的API来使用大型语言模型时，可以通过异步（或非阻塞）方式调用多个API，这样可以在不必等待每个单独请求返回的情况下，提高处理速度和应用程序的响应能力。OpenAI提供了一个简单的并行调用API的[示例](https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py)。但其他大模型服务厂商，如Aliyun，则并没有提供相应的示例。事实上，通过`curl`调用各家的API服务时，所用的参数非常相似，因为只需对OpenAI的示例进行一定的改写就可以实现对其他厂商的API的异步调用。

目前，我们提供的脚本（`api_request_parallel_processor_openai_qwen.py`）可以兼容OpenAI和Aliyun的API调用，只需要将密钥和URL修改为需要的值，且注意将输入的数据的格式调整为目标服务要求的格式即可。后续，我们将持续更新，使该脚本能够支持更多API服务，包括但不限于，MistraAI、Microsoft Azure、Google Cloud、ZhipuAI、无问芯穹等。

调用方法如下：

```bash
sh llm-api/run.sh
```