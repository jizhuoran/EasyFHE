from pathlib import Path

import numpy as np

import easyfhe.fhe as fhe


class WeightPack:
    def __init__(self, arrays):
        self.arrays = arrays

    @classmethod
    def from_npz(cls, path):
        weight_path = Path(path)
        if not weight_path.exists():
            raise ValueError(f"Weight npz {weight_path} does not exist!")
        with np.load(weight_path) as weights:
            arrays = {name: np.asarray(weights[name], dtype=np.double) for name in weights.files}
        return cls(arrays)

    def __len__(self):
        return len(self.arrays)

    def values(self, name, slots, scale=1.0):
        if name not in self.arrays:
            raise KeyError(f"weight {name!r} is missing")

        values = np.asarray(self.arrays[name], dtype=np.double).reshape(-1)
        if values.size < slots:
            values = np.pad(values, (0, slots - values.size))
        elif values.size > slots:
            values = values[:slots]
        if scale != 1.0:
            values = values * scale
        return values

    def encode(self, name, level, slots, cryptoContext, scale=1.0):
        print(name)
        return fhe.encode(self.values(name, slots, scale), name, level, slots, False, cryptoContext)

    def encode_for_cipher(self, name, cipher, cryptoContext, scale=1.0):
        return self.encode(name, cryptoContext.L - cipher.cur_limbs, cipher.slots, cryptoContext, scale)
