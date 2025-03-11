import tiktoken
import sys,os
import torch
from gpt_model import GPTModel
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length":256,
    "emb_dim":768,
    "n_heads":12,
    "n_layers":12,
    "drop_rate":0.1,
    "qkv_bias":False
}

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from text_data.gpt_dataset import create_dataloader_v1

file_path ="the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = tiktoken.get_encoding('gpt2')
total_characters = len(text)
total_tokens = len(tokenizer.encode(text))

print(f"Total characters in the file: {total_characters}")

print(f"Total tokens in the file: {total_tokens}")

train_ratio = 0.9
split_idx = int(train_ratio* len(text))
train_data = text[:split_idx]
valid_data = text[split_idx:]

torch.manual_seed(123)

train_loader = create_dataloader_v1(train_data,
                                    batch_size=2,
                                    max_length=GPT_CONFIG_124M['context_length'],
                                    stride=GPT_CONFIG_124M['context_length'],
                                    drop_last=True,
                                    shuffle=True,
                                    num_workers=0)

valid_loader = create_dataloader_v1(valid_data,
                                    batch_size=2,
                                    max_length=GPT_CONFIG_124M['context_length'],
                                    stride=GPT_CONFIG_124M['context_length'],
                                    drop_last=False,
                                    shuffle=False,
                                    num_workers=0)
print("train_loader")
for x,y in train_loader:
    print(x.shape, y.shape)

print("valid_loader")

for x,y in valid_loader:
    print(x.shape, y.shape)

def calc_loss_batch(input_batch, target_batch, model, device):
    input_ids = input_batch.to(device)
    target_ids = target_batch.to(device)
    outputs = model(input_ids)
    loss = torch.nn.functional.cross_entropy(outputs.flatten(0,1), target_ids.flatten())
    return loss

def calc_loss_loader(data_loader, model, device,num_batches=None):
    total_loss = 0
    if (len(data_loader)==0):
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches,len(data_loader))
    for i, (input_batch,target_batch) in enumerate(data_loader):
        if i<num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(123)
model = GPTModel(config=GPT_CONFIG_124M)

model.to(device)

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader,model=model, device=device)
    valid_loss = calc_loss_loader(valid_loader,model=model, device=device)

print(f"Train Loss: {train_loss:.4f}")

print(f"Valid Loss: {valid_loss:.4f}")