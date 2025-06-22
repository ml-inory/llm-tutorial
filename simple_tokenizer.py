# 读取一本小说
import requests, os


def download_text(txt_file):
    if not os.path.exists(txt_file):
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
        response = requests.get(url)
        raw_text = response.text
        with open(txt_file, "w") as f:
            f.write(raw_text)
        return raw_text
    else:
        with open(txt_file, "r") as f:
            raw_text = f.read()
        return raw_text
    
raw_text = download_text("the-verdict.txt")
print("Total number of characters:", len(raw_text))
print(raw_text[:99])

# 按空格字符分割文本
import re
text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
print(result)

# 逗号和句号也要分割
result = re.split(r'([,.]|\s)', text)
print(result)

# 去除空白字符
result = [item.strip() for item in result if item.strip()]
print(result)

# 添加其它特殊字符
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)

# 现在我们已经让一个基础的分词器开始运行了，让我们将它部署到埃迪斯·华顿的整个短篇小说上：
preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])

# 创建一个包含所有独特token的列表，并按字母顺序排序以确定词汇表的大小
all_words = sorted(list(set(preprocessed)))
vocab_size = len(all_words)
print("vocab_size: ", vocab_size)

# 创建词汇表
vocab = {s : i for i, s in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i > 50:
        break

# 创建简单的tokenizer
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i : s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," Mrs. Gisburn
said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))

# text = "Hello, do you like tea?"
# tokenizer.encode(text)

# 添加特殊token
# |<unk>| 表示未知的单词
# |<endoftext>| 表示段落的开始或结束
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|unk|>", "<|endoftext|>"])
vocab = {s : i for i, s in enumerate(all_tokens)}
print(len(vocab.items()))
print(list(vocab.items())[-5:])


# 创建能处理未知单词的tokenizer
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

tokenizer = SimpleTokenizerV2(vocab)
ids = tokenizer.encode(text)
print(ids)

print(tokenizer.decode(tokenizer.encode(text)))