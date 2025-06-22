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

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

raw_text = download_text("the-verdict.txt")
max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print(f"Token IDs:\n {inputs}")
print(f"\nInputs.shape: \n {inputs.shape}")

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

context_size = max_length
pos_embedding_layer = torch.nn.Embedding(context_size, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_size))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)