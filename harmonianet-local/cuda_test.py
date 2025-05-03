import torch
print(torch.cuda.is_available())  # should return True
print(torch.cuda.get_device_name(0))  # should return RTX 4090
