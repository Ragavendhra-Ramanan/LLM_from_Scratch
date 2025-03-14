from importlib.metadata import version
import tiktoken
print("tiktoken version",version("tiktoken"))

#initialize get encoding

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
)

integers = tokenizer.encode(text,allowed_special={"<|endoftext|>"})

print("Encoded integers:", integers)

strings = tokenizer.decode(integers)

print("Decoded strings:", strings)

#data sampling
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

