import torch
import numpy as np
import torch

class Cipher:
    def __init__(self, cv0, cv1, cur_limbs, scaling_factor=0, noise_deg=1): #todo: remove the default value of scaling_factor and noise_deg
        self.cv = [cv0, cv1]
        self.cur_limbs = cur_limbs
        self.scaling_factor = scaling_factor
        self.noise_deg = noise_deg

    def __init__(self, cv, cur_limbs, scaling_factor=0, noise_deg = 1): #todo: remove the default value of scaling_factor and noise_deg
        self.cv = cv
        self.cur_limbs = cur_limbs
        self.scaling_factor = scaling_factor
        self.noise_deg = noise_deg
    
    def clone(self):
        return Cipher([x.clone() for x in self.cv], self.cur_limbs, self. scaling_factor, self.noise_deg)

    def drop_axax(self):
        assert len(self.cv) == 3
        res, self.cv = self.cv[-1], self.cv[:-1]
        return res

    def __repr__(self):

        s = "Cipher(\n"
        for i, cv in enumerate(self.cv):
            s += f"cv{i}={cv[:self.cur_limbs]},\n"
        s += f"cur_limbs={self.cur_limbs}\n"
        s += f"scaling_factor={self.scaling_factor}\n"
        s += f"noise_deg={self.noise_deg}\n"
        s += ")"
        return s
    
    def __eq__(self, other):
        if not isinstance(other, Cipher):
            return False
        if self.noise_deg != other.noise_deg:
            return False
        if self.scaling_factor != other.scaling_factor:
            return False
        if self.cur_limbs != other.cur_limbs:
            return False
        if len(self.cv) != len(other.cv):
            return False
        for i in range(len(self.cv)):
            if not torch.equal(self.cv[i], other.cv[i]):
                return False
        return True