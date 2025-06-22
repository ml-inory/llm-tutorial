# 使用滑动窗口进行数据采样
import requests, os
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

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

text = download_text("the-verdict.txt")
tokenizer = tiktoken.get_encoding("gpt2")
all_tokens = tokenizer.encode(text)

enc_sample = all_tokens[50:]
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "--->", desired)

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "--->", tokenizer.decode([desired]))

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        token_ids = tokenizer.encode(txt)
        self.input_ids = []
        self.target_ids = []
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i + 1 : i + 1 + max_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


def create_dataloader_v1(txt, batch_size, max_length=256, stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader


with open("the-verdict.txt", "r") as f:
    txt = f.read()
    dataloader = create_dataloader_v1(txt, batch_size=1, max_length=4, stride=1, shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)

    second_batch = next(data_iter)
    print(second_batch)