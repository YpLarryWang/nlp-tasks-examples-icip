'''
参考：
- https://huggingface.co/docs/transformers/main_classes/tokenizer
- https://huggingface.co/docs/transformers/pad_truncation
- https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertTokenizer
'''

from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel

model_dir = '/media/disk2/public/bert-base-uncased'

# 初始化分词器和模型
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertModel.from_pretrained(model_dir)
# tokenizer = AutoTokenizer.from_pretrained(model_dir)
# model = AutoModel.from_pretrained(model_dir)

print('''
\n################################################################
######################## 看看基础参数  #########################
################################################################\n
''')

# 位置编码和句子类型嵌入是在 BERT 模型内部处理的
# 我们不能直接从模型输出中获取它们
# 但我们可以从模型的源代码中了解其实现方式
# 以下是如何理解位置编码(positional encoding)和句子类型嵌入(segment embedding)类型嵌入:
# - 位置编码是在 BERT 嵌入层中自动添加的，它是可学习的参数。
# - 句子类型嵌入是用于区分两个句子的（在句子对任务中）。

# 注意：如果你需要直接获取位置编码或句子类型嵌入的值，
# 你可能需要深入模型的源代码或者自己构建这些部分。

# 打印一些关于模型的信息
print()
print(f"Model embedding size: {model.config.hidden_size}")
print(f"Vocabulary size: {model.config.vocab_size}")
print(f"Positional encoding max position: {model.config.max_position_embeddings}")
print(f"Number of token type embeddings: {model.config.type_vocab_size}")

# exit()

# print()
# print('=' * 30)
# print()

print('''
\n################################################################
######################## 传入一个句子  #########################
################################################################\n
''')

# 示例句子
text = "The quick brown fox jumps over the lazy dog."

# 分词
encoded_input = tokenizer(text, return_tensors='pt', max_length=56, padding='max_length')

print(encoded_input)
print()

# 输出分词的结果
print(f"Tokens: {tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])}\n")

# 获取模型的输出，包括隐藏状态和注意力机制
outputs = model(**encoded_input)

# 最后一层隐藏状态的形状是 (batch_size, sequence_length, hidden_size)
# 在这个例子中，`hidden_states` 是嵌入表示
hidden_states = outputs.last_hidden_state
print(f"Embedding shape: {hidden_states.shape}\n")
print("Last layer hidden state:")
print(hidden_states)

# exit()

# print()
# print('=' * 30)
# print()

print('''
\n################################################################
######################### 传入句子对  ##########################
################################################################\n
''')

# 两个示例句子
text1 = "The quick brown fox jumps over the lazy dog."
text2 = "Colorless green ideas sleep furiously."

# 使用分词器处理两个句子，注意`return_tensors`设置为'pt'以获得PyTorch张量
encoded_input = tokenizer(text1, text2, return_tensors='pt')

# 输出分词的结果
print(f"Tokens: {tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])}\n")
print(f"Token Type IDs: {encoded_input['token_type_ids'][0]}\n")

# 获取模型的输出，包括隐藏状态和注意力机制
outputs = model(**encoded_input)

# 最后一层隐藏状态的形状是 (batch_size, sequence_length, hidden_size)
# 在这个例子中，`hidden_states` 是嵌入表示
hidden_states = outputs.last_hidden_state
print(f"Embedding shape: {hidden_states.shape}")

# print()
# print('=' * 30)
# print()

print('''
\n################################################################
####################### 同时处理多个句子 #######################
################################################################\n
''')

# 两个示例句子
text1 = "The quick brown fox jumps over the lazy dog."
text2 = "Colorless green ideas sleep furiously."

# 分别对两个句子进行分词，并将结果放在列表中
encoded_inputs = [tokenizer.encode(text, add_special_tokens=True) for text in [text1, text2]]

# 转换成tokens
tokens_list = [tokenizer.convert_ids_to_tokens(encoded_input) for encoded_input in encoded_inputs]

# 打印tokens
for tokens in tokens_list:
    print(tokens)

# print()
# print('=' * 30)
# print()

print('''
\n################################################################
####################### 传入多个句子对 #########################
################################################################\n
''')
    
# 四个示例句子，组成两个句子对
text1_A = "The quick brown fox jumps over the lazy dog."
text1_B = "A fast dark-colored fox leaps above a sleepy canine."

text2_A = "Colorless green ideas sleep furiously."
text2_B = "Colorful white ideas eat furiously."

# 将句子对作为列表传入分词器
encoded_inputs = tokenizer([text1_A, text2_A], [text1_B, text2_B], return_tensors='pt', padding=True)

# 查看编码结果
print("Last layer hidden state:")
print(encoded_inputs)
    
# 转换成tokens
tokens_list = [tokenizer.convert_ids_to_tokens(encoded_input.tolist()) for encoded_input in encoded_inputs['input_ids']]

print("\nTokens:")

# 打印tokens
for tokens in tokens_list:
    print(tokens)