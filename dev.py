import torch
import torch.fhe.functional as F 

a = torch.tensor([1, 2, 6], dtype=torch.uint64, device='cuda')
b = torch.tensor([4, 5, 6], dtype=torch.uint64, device='cuda')

mu = torch.tensor([14347467612885206812, 2049638230412172401], dtype=torch.uint64, device='cuda')

c = F.mul_mod(a, b, 9, mu)

print(c)