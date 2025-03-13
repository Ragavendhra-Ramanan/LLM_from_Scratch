import torch 
# from causal_attention import CausalAttention

# class MultiHeadAttentionWrapper(torch.nn.Module):
#     def __init__(self,d_in,d_out,context_length,dropout,num_heads,qkv_bias=False):
#         super().__init__()
#         self.heads = torch.nn.ModuleList([CausalAttention(d_in,d_out,context_length,dropout,qkv_bias) for _ in range(num_heads)])
    
#     def forward(self, x):
#         return torch.cat([head(x) for head in self.heads],dim=-1)
    
# torch.manual_seed(123)
# mha = MultiHeadAttentionWrapper(
#     3,2,6,0.0,num_heads=2
#     )
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

# context_vecs = mha(batch)

# print(context_vecs)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = self.d_out//self.num_heads

        self.W_query = torch.nn.Linear(d_in, d_out,bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

        self.output_proj = torch.nn.Linear(d_out, d_out)
        self.dropout = torch.nn.Dropout(dropout)

        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length,context_length),diagonal=1)
        )

    def forward (self,x):
            b,num_tokens,d_in = x.shape
            queries = self.W_query(x)
            keys = self.W_key(x)
            values = self.W_value(x)

            queries = queries.view(b,num_tokens,self.num_heads,self.head_dim)
            keys = keys.view(b,num_tokens,self.num_heads,self.head_dim)
            values = values.view(b,num_tokens,self.num_heads,self.head_dim)

            keys = keys.transpose(1,2)
            queries = queries.transpose(1,2)
            values = values.transpose(1,2)

            attn_scores = queries @ keys.transpose(2,3)
            attn_scores = attn_scores.masked_fill(self.mask.bool()[:num_tokens,:num_tokens],-torch.inf)
            attn_weights = torch.softmax(attn_scores/(keys.shape[-1]**0.5),dim=-1)
            attn_weights = self.dropout(attn_weights)
            context_vector = (attn_weights @ values).transpose(1,2)
            context_vector = context_vector.contiguous().view(
                b, num_tokens, self.d_out
            )
            return self.output_proj(context_vector)
torch.manual_seed(123)
mha = MultiHeadAttention(
        3,2,6,0.0,num_heads=2
    )
context_vecs = mha(batch)
print(context_vecs)        