from __future__ import annotations

from dataclasses import dataclass

from ..context import Context


@dataclass(frozen=True)
class CKKSContextSpec:
    depth: int
    log_n: int
    dnum: int
    dcrt_bits: int
    first_mod: int
    secret_key_dist: str = "SPARSE_TERNARY"
    rescale_tech: str = "FIXEDMANUAL"
    rotations: tuple[int, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "depth", int(self.depth))
        object.__setattr__(self, "log_n", int(self.log_n))
        object.__setattr__(self, "dnum", int(self.dnum))
        object.__setattr__(self, "dcrt_bits", int(self.dcrt_bits))
        object.__setattr__(self, "first_mod", int(self.first_mod))
        object.__setattr__(self, "secret_key_dist", str(self.secret_key_dist))
        object.__setattr__(self, "rescale_tech", str(self.rescale_tech))
        object.__setattr__(self, "rotations", tuple(int(rotation) for rotation in (self.rotations or ())))


def generate_context(spec: CKKSContextSpec, device="cpu", options=None):
    if not isinstance(spec, CKKSContextSpec):
        raise TypeError("generate_context expects a CKKSContextSpec")
    return Context.build(spec, device, options)
