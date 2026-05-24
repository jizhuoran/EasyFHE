from pathlib import Path

import numpy as np

import easyfhe.fhe as fhe


DEFAULT_SCALARS = {
    "scale.one": 1.0,
    "scale.aespa": 0.015625,
    "scale.aespa.sqrt": 0.125,
}

FOLD_SLOT_SHRINKS = (
    (16384, 4096),
    (32768, 8192),
)


class WeightPack(fhe.ConstantBundle):
    def __init__(
        self,
        arrays,
        scalars=None,
        cache_mode="plain",
        plain_cache_limit_gb=None,
        plain_cache_policy="first_fit",
    ):
        merged_scalars = dict(DEFAULT_SCALARS)
        merged_scalars.update(scalars or {})
        self.scalars = merged_scalars
        super().__init__(
            scalars=merged_scalars,
            vectors=arrays,
            cache_mode=cache_mode,
            plain_cache_limit_gb=plain_cache_limit_gb,
            plain_cache_policy=plain_cache_policy,
        )
        self.arrays = self._vectors

    @classmethod
    def from_npz(cls, path, cache_mode="plain", plain_cache_limit_gb=None, plain_cache_policy="first_fit"):
        weight_path = Path(path)
        if not weight_path.exists():
            raise ValueError(f"Weight npz {weight_path} does not exist!")
        with np.load(weight_path) as weights:
            arrays = {name: np.asarray(weights[name], dtype=np.double) for name in weights.files}
        _add_fold_slot_masks(arrays)
        return cls(
            arrays,
            cache_mode=cache_mode,
            plain_cache_limit_gb=plain_cache_limit_gb,
            plain_cache_policy=plain_cache_policy,
        )

    def scalar_value(self, name):
        return self.scalars[name]


def fold_slots_mask_name(source_slots, target_slots):
    return f"fold_slots_mask_{int(source_slots)}to{int(target_slots)}"


def _add_fold_slot_masks(arrays):
    for source_slots, target_slots in FOLD_SLOT_SHRINKS:
        arrays.setdefault(
            fold_slots_mask_name(source_slots, target_slots),
            np.ones(int(target_slots), dtype=np.double),
        )
