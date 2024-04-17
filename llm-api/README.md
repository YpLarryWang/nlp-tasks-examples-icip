在调用OpenAI或Aliyun提供的API来使用大型语言模型时，往往需要通过并行调用多个API来提高速度。OpenAI提供了一个简单的并行调用API的[示例](https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py)，Aliyun则没有提供。但由于通过`curl`进行APi调用时的命令非常相似，因为可以对OpenAI的示例进行一定的改写就可以实现对Aliyun的API的并行调用。

这里提供的代码可以兼容OpenAI和Aliyun的API调用，只需要将`API_KEY`和`API_URL`改为对应的值，且注意输入的数据格式差异即可。

调用方法如下：

```bash
sh run.sh
```