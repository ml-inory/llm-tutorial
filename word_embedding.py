import tiktoken
import torch

vocab_size = 6
output_dim = 3
input_ids = torch.tensor([2, 3, 5, 1])

torch.manual_seed(233)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print(embedding_layer.weight)

print(embedding_layer(torch.tensor([3])))

print(embedding_layer(input_ids))