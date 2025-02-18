import torch
print(torch.__version__)  # check python version
print(torch.version.cuda) # check cuda version
print(torch.cuda.is_available()) # check cuda available
print(torch.cuda.device_count()) #check cuda device count