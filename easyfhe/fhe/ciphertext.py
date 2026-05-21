class Cipher:
    def __init__(self, cv, cur_limbs, scaling_factor, noise_deg, slots, is_ext, batch_size=1):
        self.cv = cv
        self.cur_limbs = cur_limbs
        self.scaling_factor = scaling_factor
        self.noise_deg = noise_deg
        self.slots = slots
        self.is_ext = is_ext
        self.batch_size = int(batch_size)

    def cipher_like(
        self,
        cv,
        cur_limbs=None,
        scaling_factor=None,
        noise_deg=None,
        slots=None,
        is_ext=None,
        batch_size=None,
    ):
        res = Cipher(
            cv,
            self.cur_limbs if cur_limbs is None else cur_limbs,
            self.scaling_factor if scaling_factor is None else scaling_factor,
            self.noise_deg if noise_deg is None else noise_deg,
            self.slots if slots is None else slots,
            self.is_ext if is_ext is None else is_ext,
            self.batch_size if batch_size is None else batch_size,
        )
        if "ptx_twin" in self.__dict__:
            res.ptx_twin = np.copy(self.ptx_twin)
        return res

    def deep_copy(self):
        return self.cipher_like([x.clone() for x in self.cv])

    def replace_with(self, other):
        # Cipher-level mutation: callers may pass preallocated component tensors,
        # but not every higher-level path guarantees tensor-level zero allocation.
        self.cv = other.cv
        self.cur_limbs = other.cur_limbs
        self.scaling_factor = other.scaling_factor
        self.noise_deg = other.noise_deg
        self.slots = other.slots
        self.is_ext = other.is_ext
        self.batch_size = int(other.batch_size)
        if "ptx_twin" in other.__dict__:
            self.ptx_twin = np.copy(other.ptx_twin)
        elif "ptx_twin" in self.__dict__:
            del self.ptx_twin
        return self

    def shallow_copy(self):
        return self.cipher_like(self.cv)

    def cuda(self):
        cv = [x.cuda() for x in self.cv]
        return self.cipher_like(cv)

    def cpu(self):
        cv = [x.cpu() for x in self.cv]
        return self.cipher_like(cv)

    def __repr__(self):
        s = "Cipher(\n"
        for i, cv in enumerate(self.cv):
            s += f"cv{i}={cv[:self.cur_limbs]},\n"
        s += f"cur_limbs={self.cur_limbs}\n"
        s += f"scaling_factor={self.scaling_factor}\n"
        s += f"noise_deg={self.noise_deg}\n"
        s += f"slots={self.slots}\n"
        s += f"batch_size={self.batch_size}\n"
        s += ")"
        return s

Plaintext = Cipher
