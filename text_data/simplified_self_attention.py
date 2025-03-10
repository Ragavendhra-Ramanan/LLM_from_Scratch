import torch 
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

query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])

for i,x_i in enumerate(inputs):
    attn_scores_2[i]= torch.dot(x_i,query)
print("attn_scores_2",attn_scores_2)

attn_weights_2_temp = attn_scores_2 / attn_scores_2.sum()

print("attn_weights_2_temp",attn_weights_2_temp)

def softmax_naive(x):
    return torch.exp(x)/torch.exp(x).sum(dim=0)

attn_weights_2 = softmax_naive(attn_scores_2)
attn_weights_2 = torch.softmax(attn_scores_2,dim=0)

print("attn_weights_2",attn_weights_2)
context_vec_2 = torch.zeros(query.shape[0])

for i,x_i in enumerate(inputs):
    context_vec_2+= (attn_weights_2[i]*x_i)
print("context_ vector 2",context_vec_2)

attn_scores = torch.empty(6,6)
for i,x_i in enumerate(inputs):
    for j,x_j in enumerate(inputs):
        attn_scores[i][j]= torch.dot(x_i,x_j)
print(attn_scores)

attn_weights = torch.softmax(attn_scores,dim=1)

print("attn_weights",attn_weights)


all_context_vec = attn_weights@inputs

print("all_context_vec",all_context_vec)