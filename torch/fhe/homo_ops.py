import torch
import numpy as np
from .Ciphertext import Ciphertext
from .context import Context
from . import functional as F
Tensor = torch.Tensor

def polys_add_mod(a, b, MOD):
    assert a.shape == b.shape and a.shape[0] == len(MOD)
    c_ = torch.zeros(a.shape, dtype=torch.uint64, device='cuda')
    a_ = torch.from_numpy(a).cuda()
    b_ = torch.from_numpy(b).cuda()
    for i, val in enumerate(MOD):
        c_[i] = F.vec_add_mod(a_[i], b_[i], val)
    c = c_.cpu().numpy()
    return c

def polys_sub_mod(a, b, MOD):
    assert a.shape == b.shape and a.shape[0] == len(MOD)
    c_ = torch.zeros(a.shape, dtype=torch.uint64, device='cuda')
    a_ = torch.from_numpy(a).cuda()
    b_ = torch.from_numpy(b).cuda()
    for i, val in enumerate(MOD):
        c_[i] = F.vec_sub_mod(a_[i], b_[i], val)
    c = c_.cpu().numpy()
    return c

def polys_mul_mod(a, b, MOD, mu):
    assert a.shape == b.shape and a.shape[0] == len(MOD)
    c_ = torch.zeros(a.shape, dtype=torch.uint64, device='cuda')
    a_ = torch.from_numpy(a).cuda()
    b_ = torch.from_numpy(b).cuda()
    mu_ = torch.from_numpy(mu).cuda()
    for i, val in enumerate(MOD):
        c_[i] = F.vec_mul_mod(a_[i], b_[i], val, mu_[i])
    c = c_.cpu().numpy()
    return c

def homo_add(in0, in1, cryptoContext):
    assert in0.curr_limbs == in1.curr_limbs
    curr_limbs = in0.curr_limbs
    MOD = cryptoContext.moduliQ[:curr_limbs]
    res = np.zeros(in0.cv.shape, dtype=np.uint64)
    res[0] = polys_add_mod(in0.cv[0], in1.cv[0], MOD)  # res.ax
    res[1] = polys_add_mod(in0.cv[1], in1.cv[1], MOD)  # res.bx

    return Ciphertext(res, curr_limbs)

def homo_mult_core(in0, in1, moduliQ, q_mu):
    assert in0.curr_limbs == in1.curr_limbs
    curr_limbs = in0.curr_limbs
    MOD = moduliQ[:curr_limbs]
    MU = q_mu[:curr_limbs]
    res = np.zeros((3, in0.cv.shape[1], in0.cv.shape[2]), dtype=np.uint64)

    res[0] = polys_add_mod(in0.cv[0], in0.cv[1], MOD)  # res.ax
    axbx2 = polys_add_mod(in1.cv[0], in1.cv[1], MOD)
    res[0] = polys_mul_mod(res[0], axbx2, MOD, MU)
    res[2] = polys_mul_mod(in0.cv[0], in1.cv[0], MOD, MU)  # axax, use in the next KS step
    res[1] = polys_mul_mod(in0.cv[1], in1.cv[1], MOD, MU)  # res.bx
    tmp = polys_add_mod(res[1], res[2], MOD)
    res[0] = polys_sub_mod(res[0], tmp, MOD)

    return res