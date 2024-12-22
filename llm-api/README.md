# 通过异步调用API服务来使用大型语言模型

## 常规用法

在使用 OpenAI 等大模型厂商提供的 API 服务来调用大型语言模型时，可以通过异步（或非阻塞）方式发送请求。这样可以避免 CPU 花费大量时间等待网络反馈，而是在等待响应的同时处理其他需要 CPU 资源的任务，例如发送其他请求和写入已接收到的反馈结果，从而缩短整个程序的运行时间。

本项目通过命令行脚本命令行脚本（command-line script）`run.sh`来运行`api_request_parallel_processor.py`。命令行脚本是一种由命令行解释器（如 `bash` 或 `zsh`）执行的脚本文件，用于自动化执行一系列命令或操作。在本项目中，只需要将URL和密钥修改为特定的值，且注意将输入的数据的格式调整为目标服务要求的格式即可成功调用API。运行前，你必须安装两个python包：

```bash
pip install aiohttp
pip install tiktoken
```

脚本`api_request_parallel_processor_0512.py`修改自OpenAI的[示例](https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py)，但增加了对一些其他API服务的支持，同时添加了参数`seconds_to_sleep_each_loop`，用于控制每个API请求之间的时间间隔。OpenAI的API服务没有秒级的调用频率限制（使用时可设为0.01或忽略），但其他API服务有秒级的调用频率限制，因此需要设置该参数（比如Aliyun就有秒级限制，虽然没有在官网明说，使用Qwen时需要摸索尝试一下，先设置0.15-0.5为宜）。

目前，该脚本支持的API服务有：
- OpenAI: chat completion
- Aliyun: text generation
- DeepInfra: text generation
- DeepSeek: chat completion
  
后续，我们将持续更新，使该脚本能够支持更多API服务，包括但不限于，MistraAI、Microsoft Azure、Google Cloud、ZhipuAI、无问芯穹等。欢迎同学们提交PR，和我们一起维护这个脚本。

调用方法如下：

```bash
sh llm-api/run.sh
```

目前的 `run.sh` 写法是针对 MacOS 系统的，[shebang](https://zh.wikipedia.org/wiki/Shebang)行（`run.sh`脚本的第一行）指明了使用 `zsh` 作为命令行解释器。如果你希望在 Unix/Linux 系统上使用该命令行脚本，需要修改 Shebang 行为`#!/bin/bash`。如果你要在 Windows 系统上使用该命令行脚本，可以先安装`git`或者`WSL`并修改 Shebang 行为`#!/bin/bash`，即要求使用`bash`作为命令行解释器。在 Windows 上使用命令行脚本的具体细节可参考[这个帖子](https://stackoverflow.com/questions/6413377/is-there-a-way-to-run-bash-scripts-on-windows).

"Shebang" 行是一种特殊的注释行，位于脚本文件的开头，用来指定解释器的路径。它具有一种特殊格式：开头是 #!，后面跟着解释器的路径。例如：
```sh
#!/bin/bash
```

这个行告诉操作系统用 `/bin/bash` 来执行这个脚本。之所以称为 "Shebang"，是因为 `#!` 符号的发音：

- `#` 被称为 "hash" 或 "sharp"。
- `!` 被称为 "bang"。

二者组合起来就是 "shebang"。

当你在命令行中运行一个脚本时，操作系统会读取文件的第一行。如果第一行以 `#!` 开头，操作系统会根据后面的路径调用相应的解释器来执行脚本，而不是使用默认的 shell。Shebang 行的格式通常是 `#!/path/to/interpreter`，例如：
    - `#!/bin/bash`：使用 Bash 解释器。
    - `#!/usr/bin/env python3`：使用当前环境中的 Python 3 解释器。
    - `#!/bin/zsh`：使用 Zsh 解释器。

## 组内用法

欲使用实验室 API 接口，请联系胡老师。通过后老师会给大家发送一个`.ipynb`文件，将其中的`bnu_api_key`作为`api_key`参数传入即可。其他和**常规用法**里的操作相同。具体命令如下：

```bash
zsh llm-api/run_bnu.sh
```

## 更新记录

- 2024.4.27更新：支持通过组内跳板机调用，但暂不支持在请求中添加metadata.
- 2024.5.4更新：可以使用和**常规用法**中的例子相同的数据格式走组内的跳板机调用渠道，metadata也已支持。只需将大家收到的`.ipynb`文件中的`bnu_api_key`作为`api_key`参数传入即可.
- 2024.5.12更新：进一步修正了组内渠道请求错误时总是报出`KeyError`的问题，现在会在重试指定次数后记录错误信息并正常退出.
- 2024.12.22更新：阿里云已经支持OpenAI格式调用，修改了对应规则和请求样例.

## 参考资料

- 视频：[OpenAI开发者大会：如何最大限度优化大型语言模型的表现](https://www.youtube.com/watch?v=ahnGLM-RC1Y)