import easyfhe as torch
import numpy as np

class Cipher:
    _id_counter = 0

    def get_next_id():
        Cipher._id_counter += 1
        return Cipher._id_counter

    def __init__(self, cv, cur_limbs, scaling_factor, noise_deg, slots, is_ext, cipher_id = "assign"):
        self.cv = cv
        self.cur_limbs = cur_limbs
        self.scaling_factor = scaling_factor
        self.noise_deg = noise_deg
        self.slots = slots
        self.is_ext = is_ext
        if cipher_id == "assign":
            self.cipher_id = Cipher.get_next_id()
        else:
            self.cipher_id = cipher_id

    def cipher_like(
        self,
        cv,
        cur_limbs=None,
        scaling_factor=None,
        noise_deg=None,
        slots=None,
        is_ext=None,
        cipher_id="copy",
    ):
        res = Cipher(
            cv,
            self.cur_limbs if cur_limbs == None else cur_limbs,
            self.scaling_factor if scaling_factor == None else scaling_factor,
            self.noise_deg if noise_deg == None else noise_deg,
            self.slots if slots == None else slots,
            self.is_ext if is_ext == None else is_ext,
            self.cipher_id if cipher_id == "copy" else cipher_id,
        )
        if "ptx_twin" in self.__dict__:
            res.ptx_twin = np.copy(self.ptx_twin)
        return res

    def deep_copy(self):
        return self.cipher_like([x.clone() for x in self.cv], cipher_id="assign")

    def shallow_copy(self):
        return self.cipher_like(self.cv, cipher_id="copy")

    def cuda(self):
        cv = [x.cuda() for x in self.cv]
        return self.cipher_like(cv, cipher_id="to_cuda")

    def cpu(self):
        cv = [x.cpu() for x in self.cv]
        return self.cipher_like(cv, cipher_id="to_cpu")

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

Plaintext = Cipher

class PreparedPlaintext:
    def __init__(self, values, slots, encoded_values, max_encoded_value):
        self.values = values
        self.slots = slots
        self.encoded_values = encoded_values
        self.max_encoded_value = max_encoded_value
    
    def deep_copy(self):
        if torch.is_tensor(self.encoded_values):
            encoded_values = self.encoded_values.clone()
        else:
            encoded_values = np.array(self.encoded_values, copy=True)
        return PreparedPlaintext(
            self.values.copy(),
            self.slots,
            encoded_values,
            self.max_encoded_value,
        )


PreEncodeValues = PreparedPlaintext
