import torch
import numpy as np
import torch

class Cipher:
    def __init__(self, ax, bx, cur_limbs):
        self.cv = [ax, bx]
        self.cur_limbs = cur_limbs
    
    def __init__(self, cv, cur_limbs):
        self.cv = cv
        self.cur_limbs = cur_limbs
    
    def drop_axax(self):
        assert len(self.cv) == 3
        res, self.cv = self.cv[-1], self.cv[:-1]
        return res

    def __repr__(self):
        return (
            "Cipher(\n"
            f"    ax={self.cv[0][:self.cur_limbs]},\n"
            f"    bx={self.cv[1][:self.cur_limbs]},\n"
            f"    cx={self.cv[2][:self.cur_limbs]},\n" if len(self.cv) > 2 else ""
            f"    cur_limbs={self.cur_limbs}\n"
            ")"
        )
    
    def __eq__(self, other):
        if not isinstance(other, Cipher):
            return False
        if self.cur_limbs != other.cur_limbs:
            return False
        if len(self.cv) != len(other.cv):
            return False
        for i in range(len(self.cv)):
            if not torch.equal(self.cv[i], other.cv[i]):
                return False
        return True