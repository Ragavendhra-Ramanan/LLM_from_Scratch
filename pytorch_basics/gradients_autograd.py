#calculating logistic regression gradient

import torch.nn.functional as F 
from torch.autograd import grad
import torch 

y= torch.tensor([1.0])
x1 = torch.tensor([1.1])

w1 = torch.tensor([2.2],requires_grad=True)
b = torch.tensor([0.0],requires_grad=True)

z = x1*w1 +b
a= torch.sigmoid(z)

loss = F.binary_cross_entropy(a,y)

grad_w1 = grad(loss,w1,retain_graph=True)
grad_b = grad(loss,b,retain_graph=True)

print("Gradient of w1: ",grad_w1[0])
print("Gradient of b: ",grad_b[0])

#more direct way

loss.backward()
print(w1.grad)
print(b.grad)
