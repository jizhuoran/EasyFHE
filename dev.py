import torch
import torch.fhe.functional as F 

a = torch.tensor([6] * (2**15), dtype=torch.uint64, device='cuda')
b = torch.tensor([4] * (2**15), dtype=torch.uint64, device='cuda')

mu = torch.tensor([14347467612885206812, 2049638230412172401], dtype=torch.uint64, device='cuda')

c = F.mul_scalar_mod(a, 7, 9, mu)

print(c)