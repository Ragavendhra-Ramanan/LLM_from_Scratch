import torch.nn as nn
import torch
class SelfAttention_V1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_Query = nn.Parameter(torch.rand(d_in,d_out))
        self.W_Key = nn.Parameter(torch.rand(d_in,d_out))
        self.W_Value = nn.Parameter(torch.rand(d_in,d_out))

    def forward(self, x):
        keys = x @ self.W_Key
        queries = x @ self.W_Query
        values = x @ self.W_Value

        attn_scores = queries @keys.T 
        attn_scores = torch.softmax(attn_scores/(keys.shape[-1]**0.5), dim=-1)
        context_vector = attn_scores @ values
        return context_vector

torch.manual_seed(123)
inputs = torch.tensor(
    [
        [0.43,0.15,0.89],# Your
        [0.55,0.87,0.66],# journey
        [0.57,0.85,0.64],# starts
        [0.22,0.58,0.33],# with
        [0.77,0.25,0.10],# one
        [0.05,0.80,0.55] # step
    ]
)

d_in = inputs.shape[1]
d_out = 2
sa_v1 = SelfAttention_V1(d_in,d_out)
print(sa_v1(inputs))

import torch.nn as nn
import torch
class SelfAttention_V2(nn.Module):
    def __init__(self, d_in, d_out,qkv_bias = False):
        super().__init__()
        self.W_Query = nn.Linear(d_in,d_out, bias = qkv_bias)
        self.W_Key = nn.Linear(d_in,d_out,bias = qkv_bias)
        self.W_Value = nn.Linear(d_in,d_out,bias = qkv_bias)

    def forward(self, x):
        keys = self.W_Key(x)
        queries = self.W_Query(x)
        values = self.W_Value(x)

        attn_scores = queries @keys.T 
        attn_scores = torch.softmax(attn_scores/(keys.shape[-1]**0.5), dim=-1)
        context_vector = attn_scores @ values
        return context_vector

torch.manual_seed(789)
inputs = torch.tensor(
    [
        [0.43,0.15,0.89],# Your
        [0.55,0.87,0.66],# journey
        [0.57,0.85,0.64],# starts
        [0.22,0.58,0.33],# with
        [0.77,0.25,0.10],# one
        [0.05,0.80,0.55] # step
    ]
)

d_in = inputs.shape[1]
d_out = 2
sa_v2 = SelfAttention_V2(d_in,d_out)
print(sa_v2(inputs))