import torch
tensor1d = torch.tensor([1,2,3]) 
print(tensor1d.dtype) #int64

#for float -float 32 is automatically generated

#type conversion
floatvec = tensor1d.to(torch.float32)
print(floatvec.dtype) #float 32

#most operations are similar to numpy
print(floatvec.shape) # torch.Size([3])

print(floatvec.reshape(3,1)) #2d array 