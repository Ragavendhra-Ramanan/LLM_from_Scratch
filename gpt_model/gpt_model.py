import torch
import torch.nn as nn
from transformer_model import TransformerBlock,LayerNorm

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"],config["emb_dim"])
        self.pos_emb = nn.Embedding(config["context_length"],config["emb_dim"])
        self.drop_emb = nn.Dropout(config["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config)
              for _ in range(config["n_layers"])]
        )
        self.ln_f = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(config['emb_dim'],config['vocab_size'],bias=False)
    
    def forward(self, input_ids):
        batch_size,seq_len = input_ids.shape
        x = self.tok_emb(input_ids)
        x += self.pos_emb(torch.arange(seq_len, device=input_ids.device))
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.ln_f(x)
        logits = self.out_head(x)
        return logits

def generate_text_simple(model, input_ids,max_new_tokens,context_size):
    for _ in range(max_new_tokens):
        idx_cond = input_ids[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        next_token_logits = logits[:, -1, :]
        probas = torch.softmax(next_token_logits,dim=-1)
        idx_next = torch.argmax(probas,dim=-1,keepdim=True)
        input_ids = torch.cat((input_ids, idx_next), dim=-1)
    return input_ids

import tiktoken
tokenizer = tiktoken.get_encoding('gpt2')
batch = []
text1 = "Every effort moves you"
text2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(text1)))
batch.append(torch.tensor(tokenizer.encode(text2)))
batch = torch.stack(batch,dim=0)
print(batch)
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length":1024,
    "emb_dim":768,
    "n_heads":12,
    "n_layers":12,
    "drop_rate":0.1,
    "qkv_bias":False
}

torch.manual_seed(123)
model = GPTModel(config=GPT_CONFIG_124M)

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0) #batch dimension
model.eval()

generated_text = generate_text_simple(model, encoded_tensor, max_new_tokens=20, context_size=GPT_CONFIG_124M['context_length'])
decoded_text = tokenizer.decode(generated_text.squeeze(0).tolist())

print(decoded_text)

def text_to_token(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

input_text = "I have a dream"
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model,
    text_to_token(input_text, tokenizer),
    max_new_tokens=20,
    context_size=GPT_CONFIG_124M['context_length']
)

generated_text = token_to_text(token_ids, tokenizer)

print(generated_text)