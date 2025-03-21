import tiktoken
import sys,os
import torch
from gpt_model import GPTModel,generate_text_simple,text_to_token,token_to_text
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

def train_model_simple(model,train_loader,val_loader,optimizer,device,num_epochs,eval_freq,eval_iter,start_context,tokenizer):
    train_losses, val_losses, track_tokens_seen = [],[],[]
    tokens_seen,global_step = 0,-1

    for epoch in range(num_epochs):
        model.train()
        for i, (input_ids, target_ids) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = calc_loss_batch(input_ids,target_ids,model,device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_ids.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss,val_loss = evaluate_model(model,train_loader,val_loader,device,eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch: {epoch+1}/{num_epochs}, Step: {global_step}, Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}, Tokens Seen: {tokens_seen}")
        generate_and_print_sample(model,tokenizer,device,start_context)
    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model,train_loader, val_loader, device, num_batches=None):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token(start_context,tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model,input_ids=encoded,max_new_tokens=25,context_size=context_size)
    text = token_to_text(token_ids, tokenizer)
    print(text.replace("\n", " "))
    model.train()

torch.manual_seed(123)
model = GPTModel(config=GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004,
        weight_decay=0.1
    )
num_epochs = 10
start_context = "Every effort moves you"
train_loss,valid_loss,tokens_seen = train_model_simple(model, train_loader, valid_loader, optimizer, device, num_epochs, eval_freq=5, eval_iter=5, start_context=start_context, tokenizer=tokenizer)

model.to("cpu")
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(model=model,
                                 input_ids=text_to_token("Every effort moves you",tokenizer),
                                 max_new_tokens=25,
                                 context_size=GPT_CONFIG_124M['context_length'])

text = token_to_text(token_ids, tokenizer)

print(text.replace("\n", " "))

def generate(model,idx,max_new_tokens,context_size,temperature=0.0,
             top_k=None,eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:,-1,:]
        if top_k is not None:
            top_logits,_ = torch.topk(logits, top_k)
            min_val = top_logits[:,-1]
            logits = torch.where(
                logits<min_val,torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature>0:
            logits /= temperature
            logits = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(logits, 1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=-1)
    return idx


torch.manual_seed(123)
token_ids = generate(model=model,idx=text_to_token("Every effort moves you",tokenizer),max_new_tokens=35,
                     context_size=GPT_CONFIG_124M['context_length'],top_k=25,temperature=1.4)
print(token_to_text(token_ids,tokenizer).replace("\n"," "))

#load the model parameters
torch.save(model.state_dict(),"model.pth")
model = GPTModel(GPT_CONFIG_124M)

model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

#generate save
torch.save({
    "model_state_dict":model.state_dict(),
    "optimizer_state_dict":optimizer.state_dict(),
},
"model_and_optimizer.pth"
)

#checkpoints
checkpoint = torch.load("model_and_optimizer.pth",map_location=device)
model = GPTModel(GPT_CONFIG_124M)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=0.1)

optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

#training
model.train()

import urllib.request
url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch05/"
       "01_main-chapter-code/gpt_download.py")
filename = url.split('/')[-1]
urllib.request.urlretrieve(url,filename)
print("Filename",filename)
from gpt_download import download_and_load_gpt2

settings,params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
print("Settings",settings)
print("parameters",params.keys())