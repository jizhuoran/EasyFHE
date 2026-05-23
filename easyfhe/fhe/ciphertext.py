from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CipherState:
    cur_limbs: int
    noise_deg: int
    scaling_factor: Optional[float] = None

    def replace(self, *, cur_limbs=None, noise_deg=None, scaling_factor=None):
        return CipherState(
            self.cur_limbs if cur_limbs is None else cur_limbs,
            self.noise_deg if noise_deg is None else noise_deg,
            self.scaling_factor if scaling_factor is None else scaling_factor,
        )


class Cipher:
    def __init__(self, cv, state: CipherState, slots, is_ext, batch_size=1):
        self.cv = _normalize_components(cv)
        self.state = state
        self.slots = slots
        self.is_ext = is_ext
        self.batch_size = int(batch_size)

    def cipher_like(
        self,
        cv,
        state=None,
        slots=None,
        is_ext=None,
        batch_size=None,
    ):
        res = Cipher(
            cv,
            self.state if state is None else state,
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
        self.state = other.state
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
            if hasattr(cv, "dim") and cv.dim() >= 2:
                cv_repr = cv[..., : self.state.cur_limbs, :]
            else:
                cv_repr = cv
            s += f"cv{i}={cv_repr},\n"
        s += f"state={self.state}\n"
        s += f"slots={self.slots}\n"
        s += f"batch_size={self.batch_size}\n"
        s += ")"
        return s

Plaintext = Cipher


def _normalize_components(cv):
    if hasattr(cv, "dim"):
        cv = [cv]
    components = list(cv)
    normalized = []
    for component in components:
        if hasattr(component, "dim"):
            if component.dim() == 2:
                component = component.unsqueeze(0)
            elif component.dim() != 3:
                raise ValueError(
                    "cipher/plaintext components must be [batch, limbs, N]"
                )
        normalized.append(component)
    return normalized
