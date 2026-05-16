from pathlib import Path

import numpy as np

import easyfhe.fhe as fhe


class WeightPack(fhe.ConstantBundle):
    def __init__(self, arrays, cache_mode="plain"):
        super().__init__(
            info={"kind": "resnet20_aespa.weights"},
            vectors=arrays,
            cache_mode=cache_mode,
        )
        self.arrays = self.vectors

    @classmethod
    def from_npz(cls, path, cache_mode="plain"):
        weight_path = Path(path)
        if not weight_path.exists():
            raise ValueError(f"Weight npz {weight_path} does not exist!")
        with np.load(weight_path) as weights:
            arrays = {name: np.asarray(weights[name], dtype=np.double) for name in weights.files}
        return cls(arrays, cache_mode=cache_mode)
