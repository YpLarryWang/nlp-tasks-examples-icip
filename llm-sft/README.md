# 基于HuggingFace微调 Qwen2.5-0.5B模型

## 数据来源

- `m-a-p/COIG-CQIA`: 可以[从huggingface上下载](https://huggingface.co/datasets/m-a-p/COIG-CQIA)（在`ruozhiba`目录下）
- 对应论文：[COIG-CQIA: Quality is All You Need for Chinese Instruction Fine-tuning](https://arxiv.org/abs/2403.18058)

## 模型下载

参考脚本`download_from_modelscope.py`使用`modelscope`下载模型。

## 环境配置

```bash
# 使用 Conda 环境文件创建环境
conda env create -f environment.yml

# 或者，先创建基础 Conda 环境，然后安装 pip 包
conda env create -f environment_conda_only.yml
conda activate 你的环境名称
pip install -r requirements.txt
```

注意到，`bitsandbytes`库的安装对环境有着较高的要求，请参考[官方文档](https://huggingface.co/docs/bitsandbytes/main/installation)。在159服务器上，受到GCC版本限制，我们无法使用0.42版本以上的`bitsandbytes`。因此无法使用`bitsandbytes`的量化功能。此时，无需严格遵循这里提供的环境配置，安装当前系统允许的最高版本即可。

## LoRA 微调

本项目提供了一个使用 LoRA 技术对 Qwen 模型进行微调的示例脚本。

通过运行 `run_qwen_no_sft_trainer.bash` 脚本来启动训练。

```bash
bash run_qwen_no_sft_trainer.bash
```

该脚本会使用 `torchrun` 在指定的 GPU 上启动 `run_qwen_no_sft_trainer.py` 脚本进行分布式训练。

### 主要参数配置

你可以在 `run_qwen_no_sft_trainer.bash` 文件中修改以下关键参数：

*   `CUDA_VISIBLE_DEVICES`: 指定使用的 GPU ID (例如 `0,1`)。
*   `--nproc_per_node`: 指定使用的 GPU 数量，应与 `CUDA_VISIBLE_DEVICES` 中的数量一致。
*   `--dataset_name_or_path`: 指定训练数据集文件的路径 (JSONL 格式)。
*   `--pretrained_model_name_or_path`: 指定预训练 Qwen 模型的路径或 Hugging Face Hub 名称。
*   `--output_dir`: 指定训练输出（检查点、日志等）保存的基础目录。脚本会自动在此目录下创建包含量化位数、批次大小等信息的子目录。
*   `--padding_side`: Tokenizer 的填充方向，对于 Qwen 通常建议使用 "left"。
*   `--per_device_train_batch_size`: 每个 GPU 的训练批量大小。
*   `--gradient_accumulation_steps`: 梯度累积步数。有效批量大小 = `per_device_train_batch_size` * `nproc_per_node` * `gradient_accumulation_steps`。
*   `--learning_rate`: 学习率。
*   `--num_train_epochs`: 训练轮数。
*   `--logging_steps`: 每隔多少步记录一次日志。
*   `--save_strategy`: 保存策略 ("epoch" 或 "steps")。
*   `--save_total_limit`: 最多保存多少个检查点。

## QLoRA 量化微调

本项目还提供了一个使用 QLoRA 技术对 Qwen 模型进行量化微调的示例脚本。

## 基本用法

通过运行 `run_qwen_quantization.bash` 脚本来启动训练。

```bash
bash run_qwen_quantization.bash
```

该脚本会使用 `torchrun` 在指定的 GPU 上启动 `sft_qwen_quantization.py` 脚本进行分布式训练。

### 主要参数配置

你可以在 `run_qwen_quantization.bash` 文件中修改以下关键参数：

*   `--quantization_bit`: **量化位数**。设置为 `4` 使用 4-bit QLoRA，设置为 `8` 使用 8-bit 量化。设置为其他值或不设置则不进行量化。

此脚本的其他参数和 `run_qwen_no_sft_trainer.py` 脚本相同，这里不再赘述。

### 重要发现：量化与梯度检查点 (Gradient Checkpointing)

在当前的开发和测试环境 (`transformers`, `peft`, `accelerate`, `bitsandbytes` 等库的特定版本组合) 下，我们观察到以下现象：

1.  **4-bit QLoRA ( `--quantization_bit=4` )**: **可以** 成功启用并使用**梯度检查点 (Gradient Checkpointing)**。这符合 QLoRA 的设计预期，梯度检查点可以有效降低显存消耗。脚本已配置为在 4-bit 模式下自动启用梯度检查点。
2.  **8-bit 量化 ( `--quantization_bit=8` )**: 在当前环境下，**无法** 成功启用**梯度检查点**。尝试启用会导致训练过程中出现 `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` 错误，表明梯度流在反向传播时被中断。脚本已配置为在 8-bit 模式下自动禁用梯度检查点，以确保训练能够运行。

#### 原因分析

需要强调的是，8-bit 量化训练**并非本质上不能**使用梯度检查点。我们遇到的这个问题**很可能源于当前环境中特定库版本之间的兼容性问题**。

*   梯度检查点通过重新计算而非存储所有中间激活值来节省显存。
*   量化（尤其是 `bitsandbytes` 实现的 8-bit 和 4-bit）改变了模型层的计算方式。
*   PEFT (LoRA) 修改了模型的结构，只训练适配器参数。

这三者（梯度检查点、量化、PEFT）需要底层库（`transformers`, `accelerate`, `bitsandbytes`, `peft`, `torch`）进行精密的协调才能正确工作。当这些库的某个版本组合存在未预料到的交互或 bug 时，就可能导致梯度检查点在特定量化模式下失效。

**未来可能的解决方案：**

*   **更新库版本**：随着库的更新，这类兼容性问题可能会被修复。定期尝试更新 `transformers`, `peft`, `accelerate`, `bitsandbytes` 等库到最新版本，并重新测试 8-bit + 梯度检查点的组合。
*   **查找特定版本组合**：社区或库的 issue tracker 中可能已经有人报告并解决了类似问题，可能需要安装一组特定的、已知兼容的库版本。

目前，脚本中的条件逻辑（仅在 4-bit 时启用梯度检查点）是基于当前环境验证结果的一个实用性措施。 

## 使用 LoRA 模型进行推理

项目包含一个推理脚本 `infer_qlora.py`，用于加载训练好的 QLoRA 模型（基础模型 + LoRA 适配器）或仅加载量化的基础模型进行批量推理。

### 推理脚本用法

```bash
python infer_qlora.py \
    --base_model_path /path/to/your/base/qwen_model \
    --lora_path /path/to/your/qlora_checkpoint \
    --quantization_bit 4 \
    --question_file /path/to/your/questions.txt \
    --max_new_tokens 512
```

### 推理参数说明

*   `--base_model_path` (必需): 指向基础 Qwen 模型的路径。
*   `--lora_path` (可选): 指向训练好的 LoRA 适配器权重目录 (例如 `output/ruozhiba_qlora4/checkpoint-XXX`)。如果省略，则只使用基础模型进行推理。
*   `--quantization_bit` (可选): 指定加载基础模型时使用的量化位数 (`4` 或 `8`)。应与训练时使用的位数一致。如果省略，则以默认精度加载基础模型（非量化）。
*   `--question_file` (必需): 包含待提问问题的文件的路径。支持：
    *   `.txt`: 每行一个问题。
    *   `.csv`, `.jsonl`, `.xlsx`: 脚本会尝试读取名为 `prompt` 或 `question` 的列，如果找不到，则读取第一列作为问题列表。
*   `--max_new_tokens` (可选, 默认 512): 控制模型生成回复的最大 token 数量。
*   `--max_input_length` (可选, 默认 1024): 输入文本的最大 token 长度，超过部分会被截断。

该脚本会加载模型，读取问题文件，使用 Qwen 的聊天模板格式化输入，进行批量推理，并将问题和对应的回复打印到控制台。 


## 参考资料：
参考huggingface和datawhale的文档、教程、示例：
- 使用大型语言模型生成文本：https://huggingface.co/docs/transformers/main/en/llm_tutorial
- hugginfgace总结了文本生成时常见的坑：https://huggingface.co/docs/transformers/main/en/llm_tutorial#pitfalls
- 使用huggingface的TRL库进行监督微调和强化学习：https://huggingface.co/docs/trl/sft_trainer
- 使用llama-factory对LLM进行预训练、监督微调、强化学习（Colab笔记本）：https://colab.research.google.com/drive/1d5KQtbemerlSDSxZIfAaWXhKr30QypiK?usp=sharing#scrollTo=psywJyo75vt6
- Qwen2.5使用指南：https://github.com/datawhalechina/self-llm/tree/master/models/Qwen2.5
- 使用Ollama在本地部署大模型：https://ollama.com/
- OneFlow的分布式训练教程：https://docs.oneflow.org/master/parallelism/01_introduction.html