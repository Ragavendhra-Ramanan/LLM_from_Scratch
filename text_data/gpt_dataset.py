import torch
from torch.utils.data import Dataset,DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self,text, tokenizer,max_length,stride):
        self.input_ids=[]
        self.target_ids=[]
        token_ids = tokenizer.encode(text)
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i+max_length]))
            self.target_ids.append(torch.tensor(token_ids[i+1:i+max_length+1]))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

def create_dataloader_v1(txt,batch_size=4,max_length=256,stride=128,
                         shuffle=True,drop_last=True,num_workers=0):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, batch_size=8 ,max_length=4, stride=4,shuffle=False) 
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

context_length = 4
position_embedding_layer = torch.nn.Embedding(context_length,output_dim)
pos_embeddings = position_embedding_layer(torch.arange(context_length))

print(pos_embeddings.shape)

input_embeddings = pos_embeddings+token_embeddings
print(input_embeddings.shape)