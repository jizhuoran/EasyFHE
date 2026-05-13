from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .sample_arithmetic import CkksParams, as_uint64_matrix


UIntArray = np.ndarray


@dataclass(frozen=True)
class ContextKeyMaterial:
    secret_key: UIntArray
    public_key_b: UIntArray
    public_key_a: UIntArray
    params: Optional[CkksParams] = None
    secret_key_coeff: Optional[UIntArray] = None

    def __post_init__(self):
        object.__setattr__(self, "secret_key", as_uint64_matrix("secret_key", self.secret_key))
        object.__setattr__(self, "public_key_b", as_uint64_matrix("public_key_b", self.public_key_b))
        object.__setattr__(self, "public_key_a", as_uint64_matrix("public_key_a", self.public_key_a))
        if self.secret_key_coeff is not None:
            object.__setattr__(self, "secret_key_coeff", np.asarray(self.secret_key_coeff, dtype=np.int64))
        if self.public_key_b.shape != self.public_key_a.shape:
            raise ValueError(
                f"public_key_b/public_key_a shape mismatch: {self.public_key_b.shape} vs {self.public_key_a.shape}"
            )
        if self.secret_key.shape != self.public_key_b.shape:
            raise ValueError(
                f"secret/public key shape mismatch: {self.secret_key.shape} vs {self.public_key_b.shape}"
            )

    def to_dict(self) -> dict:
        return {
            "secret_key": self.secret_key,
            "public_key_b": self.public_key_b,
            "public_key_a": self.public_key_a,
            "params": self.params,
            "secret_key_coeff": self.secret_key_coeff,
        }
