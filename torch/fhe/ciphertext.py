class Cipher:
    def __init__(self, cv, cur_limbs, noise_deg, slots, is_ext):
        self.cv = cv
        self.cur_limbs = cur_limbs
        self.noise_deg = noise_deg
        self.slots = slots
        self.is_ext = is_ext

    def cipher_like(self, cv, cur_limbs=None, noise_deg=None, slots=None, is_ext=None):
        return Cipher(
            cv,
            self.cur_limbs if cur_limbs == None else cur_limbs,
            self.noise_deg if noise_deg == None else noise_deg,
            self.slots if slots == None else slots,
            self.is_ext if is_ext == None else is_ext,
        )

    def deep_copy(self):
        return self.cipher_like([x.clone() for x in self.cv])

    def __repr__(self):
        s = "Cipher(\n"
        for i, cv in enumerate(self.cv):
            s += f"cv{i}={cv[:self.cur_limbs]},\n"
        s += f"cur_limbs={self.cur_limbs}\n"
        s += f"noise_deg={self.noise_deg}\n"
        s += f"slots={self.slots}\n"
        s += ")"
        return s


Plaintext = Cipher
