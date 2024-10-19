import torch
import numpy as np
import torch

class Ciphertext:
    def __init__(self, cv, curr_limbs):  # cv for ciphertext vector
        # 确保polynomial是一个numpy数组
        if not isinstance(cv, np.ndarray):
            raise ValueError("Polynomial must be a numpy array")
        # 确保index是一个uint64类型
        if not isinstance(curr_limbs, (int, np.uint64)):
            raise ValueError("curr_limbs must be an integer or uint64")
        if curr_limbs < 0 or curr_limbs > np.iinfo(np.uint64).max:
            raise ValueError("curr_limbs must be in the range of uint64")

        self.cv = cv
        self.curr_limbs = int(curr_limbs)

    def __repr__(self):
        return f"Ciphertext(cv={self.cv}, curr_limbs={self.curr_limbs})"

class Cipher:
    def __init__(self, ax, bx, cur_limbs):
        self.cv = [ax, bx]
        self.cur_limbs = cur_limbs
    
    def __init__(self, cv, cur_limbs):
        self.cv = cv
        self.cur_limbs = cur_limbs
    
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

class Cipher:
    def __init__(self, ax, bx, cur_limbs):
        self.cv = [ax, bx]
        self.cur_limbs = cur_limbs
    
    def __init__(self, cv, cur_limbs):
        self.cv = cv
        self.cur_limbs = cur_limbs
    
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