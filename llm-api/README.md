# 通过异步调用API服务来使用大型语言模型

## 常规用法

在使用 OpenAI 等大模型厂商提供的 API 服务来调用大型语言模型时，可以通过异步（或非阻塞）方式发送请求。这样可以避免 CPU 花费大量时间等待网络反馈，而是在等待响应的同时处理其他需要 CPU 资源的任务，例如发送其他请求和写入已接收到的反馈结果，从而缩短整个程序的运行时间。

本项目通过命令行脚本命令行脚本（command-line script）`run.sh`来运行`api_request_parallel_processor.py`。命令行脚本是一种由命令行解释器（如 `bash` 或 `zsh`）执行的脚本文件，用于自动化执行一系列命令或操作。在本项目中，只需要将URL和密钥修改为特定的值，且注意将输入的数据的格式调整为目标服务要求的格式即可成功调用API。运行前，你必须安装两个python包：

```bash
pip install aiohttp
pip install tiktoken
```

本目录下的脚本`api_request_parallel_processor.py`修改自OpenAI的[示例](https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py)，但增加了对一些其他 API 服务的支持，同时添加了参数`seconds_to_sleep_each_loop`，用于控制每个 API 请求之间的时间间隔。OpenAI 的 API 服务没有秒级的调用频率限制（使用时可设为0.01或忽略），但其他 API 服务可能存在秒级的调用频率限制，因此需要设置该参数（比如 Aliyun 就有秒级限制，虽然没有在官网明说，使用 Qwen 系列时需要摸索尝试一下，先设置0.1-0.3为宜）。

目前，该脚本支持的API服务有：
- OpenAI: chat completion
- Aliyun: ~~text generation~~ 已更新为 chat completion
- DeepInfra: text generation
- DeepSeek: chat completion
  
后续，我们将持续更新，使该脚本能够支持更多API服务，包括但不限于，MistraAI、Microsoft Azure、Google Cloud、ZhipuAI、无问芯穹等。欢迎同学们提交PR，和我们一起维护这个脚本。

调用方法如下：

```bash
sh llm-api/run.sh
```

目前的 `run.sh` 写法是针对 MacOS 系统的，[shebang](https://zh.wikipedia.org/wiki/Shebang)行（`run.sh`脚本的第一行）指明了使用 `zsh` 作为命令行解释器。如果你希望在 Unix/Linux 系统上使用该命令行脚本，需要修改 Shebang 行为`#!/bin/bash`，即要求使用`bash`作为命令行解释器。如果你要在 Windows 系统上使用该命令行脚本，可以先安装`git`或者`WSL`并修改 Shebang 行为`#!/bin/bash`。在 Windows 上使用命令行脚本的具体细节可参考[这个帖子](https://stackoverflow.com/questions/6413377/is-there-a-way-to-run-bash-scripts-on-windows).

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

*对于选择“Python程序设计与数据分析”、“面向对象程序设计”、“自然语言处理”课程的同学，我们在扩展讲解阶段暂时不涉及组内 API 渠道，故请优先参考**常规用法**。*

欲使用组内的 API 接口，请联系胡老师。通过后老师会给大家发送一个`.ipynb`文件，将其中的`bnu_api_key`作为`api_key`参数传入即可。其他和**常规用法**里的操作相同。具体命令如下：

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

## FAQ

**Q**: 我的程序出现443报错怎么办？
\
**A**: 一般在通过OpenAI官方URL进行调用时可能出现，是用科学的方法方可解决。复制终端代理命令并粘贴到你正工作的终端中即可。4开头的错误都是用户端的问题，而不是服务端的问题，因此遇到4开头的三位数报错都请先检查自己的网络连接情况。

**Q**: 我的程序出现500报错怎么办？
\
**A**: 5开头的报错都是服务端问题，不是我们用户能解决的。可以先做别的事等待官方debug。

**Q**: 我的程序在最后一条请求上卡了很长时间怎么办？
\
**A**: 这个问题源于某些模型在特定情况下会发疯似的一直输出空格或者某个标点符号。
- 想要预防或缓解这种情况，可以控制max_tokens参数在一个较小的值（只要这个值可以满足你问题的需要）;
- 当出现这种情况时，可以选择等待或直接中断程序，然后更换随机种子重新发送出错的请求，最后再把返回结果和之前正常的返回结果合起来;
- 根据经验，越强的模型出现这种情况的概率越低，例如早期的gpt-3.5很容易出现这样的情况，到了gpt-4o时已经没有遇到过这样的情况.

**Q**: 我得到的返回结果顺序和请求顺序完全不一样，我没办法匹配上了怎么办？
\
**A**: 由于我们的程序是异步调用的 API，所以返回结果的顺序和请求时的顺序不同是完全正常的。因为我们允许程序不等待与当前请求对应的返回结果就会发送下一个请求，这样短的回复会更快返回，长的回复会更慢返回，导致返回结果的文件中样本的顺序和发出时的顺序完全不同。为了避免这种对不上号的情况，请一定要设置metadata参数，在这里面你可以以字典形式添加任意参数，比如样本的ID，从而使得结果和请求能够进行匹配（请参考`requests/`目录下的样例）。

**Q**: 程序返回文件中过的token数量和我用阿里云官方的token技术器计算的token数量不一样，这是为什么？\
**A**: 这是因为本程序使用的是`tiktoken`对token进行计数，这个库仅适用于OpenAI系列模型，且需要根据不同的模型设定不同的参数（详情请见[这个链接](https://github.com/openai/tiktoken)）。对于其他模型，目前这个程序计算的token数仅仅只能作为一个参考帮助进行费用估算。