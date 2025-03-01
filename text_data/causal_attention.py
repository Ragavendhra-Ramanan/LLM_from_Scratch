import torch

class CausalAttention(torch.nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.W_query = torch.nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in,d_out,bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length,context_length),diagonal=1)
        )
    
    def forward(self,x):
        batch_size, seq_length, d_in = x.size()
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2)
        attn_scores = attn_scores.masked_fill(self.mask.bool()[:seq_length,:seq_length], -torch.inf)
        attn_weights = torch.softmax(attn_scores/(keys.shape[-1]**0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vector = attn_weights @ values
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

batch = torch.stack((inputs,inputs),dim=0)

causal_attn = CausalAttention(3, 2, batch.shape[1], 0.0)
context_vector = causal_attn(batch)
print(context_vector)
