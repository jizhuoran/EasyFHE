import torch
class Cipher:
    def __init__(self, cv, cur_limbs, scaling_factor, noise_deg, slots):
        self.cv = cv
        self.cur_limbs = cur_limbs
        self.scaling_factor = scaling_factor
        self.noise_deg = noise_deg
        self.slots = slots
    
    def deep_copy(self):
        return Cipher([x.clone() for x in self.cv], self.cur_limbs, self.scaling_factor, self.noise_deg, self.slots)

    def shallow_copy(self):
        return Cipher(self.cv, self.cur_limbs, self.scaling_factor, self.noise_deg, self.slots)

    def clone(self):
        return Cipher([x.clone() for x in self.cv], self.cur_limbs, self.scaling_factor, self.noise_deg, self.slots)

    def drop_last_elements(self, num_levels):
        assert num_levels <= self.cur_limbs and num_levels >= 0
        self.cur_limbs -= num_levels

    def try_drop_last_elements(self, num_levels):
        if num_levels > 0:
            self.drop_last_elements(num_levels)

    def __repr__(self):

        s = "Cipher(\n"
        for i, cv in enumerate(self.cv):
            s += f"cv{i}={cv[:self.cur_limbs]},\n"
        s += f"cur_limbs={self.cur_limbs}\n"
        s += f"scaling_factor={self.scaling_factor}\n"
        s += f"noise_deg={self.noise_deg}\n"
        s += f"slots={self.slots}\n"
        s += ")"
        return s
    
    def __eq__(self, other):
        if not isinstance(other, Cipher):
            return False
        if self.slots != other.slots:
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

#todo: remove default values of noise_deg and scaling_factor, should be set after encoding
#todo: remove N
#todo: rename l to cur_limbs
#todo: put `slots` to the end
class Plaintext:
    def __init__(self, mx, N, slots, l, scaling_factor, noise_deg):
        self.mx = mx
        self.N = N
        self.slots = slots
        self.l = l
        self.noise_deg = noise_deg
        self.scaling_factor = scaling_factor


    def __eq__(self, other):
        if not isinstance(other, Plaintext):
            return False
        if self.N != other.N:
            return False
        if len(self.mx) != len(other.mx):
            return False
        if not torch.equal(self.mx, other.mx):
            return False
        if self.slots != other.slots:
            return False
        if self.l != other.l:
            return False
        if self.noise_deg != other.noise_deg:
            return False
        if self.scaling_factor != other.scaling_factor:
            return False
        return True