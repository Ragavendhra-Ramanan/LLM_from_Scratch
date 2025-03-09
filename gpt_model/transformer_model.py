import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from text_data.multihead_attention import MultiHeadAttention

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length":1024,
    "emb_dim":768,
    "n_heads":12,
    "n_layers":12,
    "drop_rate":0.1,
    "qkv_bias":False
}

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * x + self.shift
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
#gelu has non zero gradient for negative except at x=-0.75 ..smaller smooth negative value [non zero]

class FeedForward(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config['emb_dim'], 4 * config['emb_dim']),
            GELU(),
            nn.Linear(4 * config['emb_dim'], config['emb_dim'])
        )
    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in = config['emb_dim'],
            d_out=config['emb_dim'],
            context_length=config['context_length'],
            dropout=config['drop_rate'],
            num_heads=config['n_heads'],
            qkv_bias=config['qkv_bias']
        )
        self.feed_forward = FeedForward(config)
        self.layer_norm1 = LayerNorm(config['emb_dim'])
        self.layer_norm2 = LayerNorm(config['emb_dim'])
        self.dropout = nn.Dropout(config['drop_rate'])
    
    def forward(self,x):
        shortcut = x
        x = self.layer_norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = shortcut + x

        shortcut = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = shortcut + x
        return x

torch.manual_seed(123)
x= torch.rand(2,4,768)

gpt_model = TransformerBlock(GPT_CONFIG_124M)

output = gpt_model(x)
print(output.shape)
