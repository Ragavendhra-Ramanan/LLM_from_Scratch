GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length":1024,
    "emb_dim":768,
    "n_heads":12,
    "n_layers":12,
    "drop_rate":0.1,
    "qkv_bias":False
}

import torch 
import torch.nn as nn

class DummyGPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"],config["emb_dim"])
        self.pos_emb = nn.Embedding(config["context_length"],config["emb_dim"])
        self.drop_emb = nn.Dropout(config["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(config)
              for _ in range(config["n_layers"])]
        )
        self.ln_f = DummyLayerNorm(config["emb_dim"])
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

class DummyTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
    
    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
    def forward(self, x):
        return x

import tiktoken
tokenizer = tiktoken.get_encoding('gpt2')
batch = []
text1 = "Every effort moves you"
text2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(text1)))
batch.append(torch.tensor(tokenizer.encode(text2)))
batch = torch.stack(batch,dim=0)
print(batch)

torch.manual_seed(123)
model = DummyGPTModel(config=GPT_CONFIG_124M)

output = model(batch)
print(output.shape)
