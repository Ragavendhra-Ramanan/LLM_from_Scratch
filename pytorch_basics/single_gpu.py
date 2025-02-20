import torch 
import torch.nn.functional as F 

class NeuralNetwork(torch.nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            #1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            #2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            #linear layer takes number of input and output nodes as arguments, bias true by default
            #output layer
            torch.nn.Linear(20, num_outputs)
        )
    def forward(self,x):
        logits = self.layers(x)
        return logits
torch.manual_seed(123)    
model = NeuralNetwork(50,3)

#model summary
print(model)

#total parameters
print(sum(p.numel() for p in model.parameters()))
#total trainable parameters
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

#see first layer weights

print(model.layers[0].weight.shape)

#see first layer bias

print(model.layers[0].bias.shape)

torch.manual_seed(123)
X = torch.rand((1,50))
out = model(X)
print(out)

#Addmm - Matrix multiplcation followed by addition says , last used function to compute computational graph

#  for inference no computation graph with softmax

with torch.no_grad():
    out = torch.softmax(model(X),dim=1)

print(out)

#creating toy dataset

X_train = torch.tensor([
    [-1.2,3.1],
    [-0.9,2.9],
    [-0.5,2.6],
    [2.3,-1.1],
    [2.7,-1.5]
])

y_train = torch.tensor([0,0,0,1,1])

X_test = torch.tensor([
    [-0.8,2.8],
    [2.6,-1.6]
])

y_test = torch.tensor([0,1])

#custom dataset class

from torch.utils.data import Dataset

class ToyDataset(Dataset):
    def __init__(self,X,y):
        self.features = X
        self.labels = y
    
    #retrieve one  record from dataset
    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y
    
    def __len__(self):
        return self.labels.shape[0]
    
train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test,y_test)

print(len(train_ds))

#dataloader
from torch.utils.data import DataLoader

torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0)

# batch size =2 but last sample is less than that , this can disturb convergence in training hence drop last batch

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True
)


torch.manual_seed(123)

# Initialize model
model = NeuralNetwork(num_inputs=2,num_outputs=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device

model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(features)
        loss = F.cross_entropy(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
model.eval()