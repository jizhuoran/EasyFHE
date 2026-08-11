from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CipherState:
    cur_limbs: int
    scale_degree: int
    scaling_factor: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(self, "cur_limbs", _positive_int("cur_limbs", self.cur_limbs))
        object.__setattr__(self, "scale_degree", _positive_int("scale_degree", self.scale_degree))
        if self.scaling_factor is not None:
            scaling_factor = float(self.scaling_factor)
            if scaling_factor <= 0.0:
                raise ValueError(f"scaling_factor must be positive, got {scaling_factor}")
            object.__setattr__(self, "scaling_factor", scaling_factor)

    def replace(self, *, cur_limbs=None, scale_degree=None, scaling_factor=None):
        return CipherState(
            self.cur_limbs if cur_limbs is None else cur_limbs,
            self.scale_degree if scale_degree is None else scale_degree,
            self.scaling_factor if scaling_factor is None else scaling_factor,
        )


@dataclass(frozen=True)
class EncodedScalar:
    """CRT-encoded scalar values together with their CKKS scale metadata."""

    residues: object
    cur_limbs: int
    scale_degree: int = 1
    scaling_factor: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "cur_limbs", _positive_int("cur_limbs", self.cur_limbs))
        object.__setattr__(self, "scale_degree", _nonnegative_int("scale_degree", self.scale_degree))
        scaling_factor = float(self.scaling_factor)
        if scaling_factor <= 0.0:
            raise ValueError(f"scaling_factor must be positive, got {scaling_factor}")
        object.__setattr__(self, "scaling_factor", scaling_factor)

    def to(self, device):
        return EncodedScalar(
            self.residues.to(device),
            self.cur_limbs,
            self.scale_degree,
            self.scaling_factor,
        )

    def clone(self):
        return EncodedScalar(
            self.residues.clone(),
            self.cur_limbs,
            self.scale_degree,
            self.scaling_factor,
        )

    def __getitem__(self, item):
        return EncodedScalar(
            self.residues[item],
            self.cur_limbs,
            self.scale_degree,
            self.scaling_factor,
        )

    @property
    def shape(self):
        return self.residues.shape

    def tolist(self):
        return self.residues.tolist()


class Cipher:
    def __init__(self, cv, state: CipherState, slots, is_ext, batch_size=1):
        self.cv = _normalize_components(cv)
        if not isinstance(state, CipherState):
            raise TypeError(f"state must be CipherState, got {type(state)}")
        self.state = state
        self.slots = _positive_int("slots", slots)
        self.is_ext = bool(is_ext)
        self.batch_size = _positive_int("batch_size", batch_size)
        _validate_component_batch_size(self.cv, self.batch_size)

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
    if not components:
        raise ValueError("cipher/plaintext components must not be empty")
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


def _validate_component_batch_size(components, batch_size):
    for component in components:
        if not hasattr(component, "dim"):
            continue
        component_batch = int(component.shape[0])
        if component_batch != int(batch_size):
            raise ValueError(
                "cipher/plaintext component batch size mismatch: "
                f"component={component_batch}, metadata={batch_size}"
            )


def _positive_int(name, value):
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _nonnegative_int(name, value):
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value
