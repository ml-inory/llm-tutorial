import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of some"
ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(ids)

strings = tokenizer.decode(ids)
print(strings)