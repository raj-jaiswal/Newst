import torch
print(torch.cuda.is_available())   # should be True
print(torch.version.cuda)          # shows CUDA version used by torch
print(torch.cuda.get_device_name(0))
