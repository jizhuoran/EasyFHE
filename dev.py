import torch
import torch.fhe.functional as F 

a = torch.tensor([1, 2, 3], dtype=torch.uint64, device='cuda')
b = torch.tensor([4, 5, 6], dtype=torch.uint64, device='cuda')

c = F.add_mod(a, b, 7)

print(c)