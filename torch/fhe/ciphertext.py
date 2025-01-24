import torch


class Cipher:
    def __init__(self, cv, cur_limbs, scaling_factor, noise_deg, slots, is_ext):
        self.cv = cv
        self.cur_limbs = cur_limbs
        self.scaling_factor = scaling_factor
        self.noise_deg = noise_deg
        self.slots = slots
        self.is_ext = is_ext

    def cipher_like(
        self,
        cv,
        cur_limbs=None,
        scaling_factor=None,
        noise_deg=None,
        slots=None,
        is_ext=None,
    ):
        return Cipher(
            cv,
            self.cur_limbs if cur_limbs == None else cur_limbs,
            self.scaling_factor if scaling_factor == None else scaling_factor,
            self.noise_deg if noise_deg == None else noise_deg,
            self.slots if slots == None else slots,
            self.is_ext if is_ext == None else is_ext,
        )

    def deep_copy(self):
        return self.cipher_like([x.clone() for x in self.cv])

    def shallow_copy(self):
        return self.cipher_like(self.cv)

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


# todo: remove default values of noise_deg and scaling_factor, should be set after encoding
# todo: remove N
# todo: rename l to cur_limbs
# todo: put `slots` to the end


class Plaintext:
    def __init__(self, mv, cur_limbs, scaling_factor, noise_deg, slots, is_ext):
        self.mv = mv
        self.cur_limbs = cur_limbs
        self.noise_deg = noise_deg
        self.scaling_factor = scaling_factor
        self.slots = slots
        self.is_ext = is_ext

    def shallow_copy(self):
        return Plaintext(self.mv, self.cur_limbs, self.scaling_factor, self.noise_deg, self.slots, self.is_ext)

    def __eq__(self, other):
        if not isinstance(other, Plaintext):
            return False
        if len(self.mv) != len(other.mv):
            return False
        if not torch.equal(self.mv, other.mv):
            return False
        if self.slots != other.slots:
            return False
        if self.cur_limbs != other.cur_limbs:
            return False
        if self.noise_deg != other.noise_deg:
            return False
        if self.scaling_factor != other.scaling_factor:
            return False
        if self.is_ext != other.is_ext:
            return False
        return True
