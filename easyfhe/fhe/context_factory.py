from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

from .client import Client
from ._keygen.context_material_builder import ContextMaterialBuilder
from ._keygen.native_sampler import CkksSamplerConfig, NativeServerMaterial, sample_native_client_server


@dataclass(frozen=True)
class CKKSContextSpec:
    depth: int
    log_n: int
    dnum: int
    dcrt_bits: int
    first_mod: int
    secret_key_dist: str = "SPARSE_TERNARY"
    scale_mode: str = "fixed"
    rescale_policy: str = "manual"
    rotations: tuple[int, ...] = ()
    auto_load_keys: Optional[bool] = None
    rotation_random_mode: str = "fresh"
    rotation_key_limb_limits: Mapping[int, int] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "depth", int(self.depth))
        object.__setattr__(self, "log_n", int(self.log_n))
        object.__setattr__(self, "dnum", int(self.dnum))
        object.__setattr__(self, "dcrt_bits", int(self.dcrt_bits))
        object.__setattr__(self, "first_mod", int(self.first_mod))
        object.__setattr__(self, "secret_key_dist", str(self.secret_key_dist))
        object.__setattr__(self, "scale_mode", _normalize_scale_mode(self.scale_mode))
        object.__setattr__(self, "rescale_policy", _normalize_rescale_policy(self.rescale_policy))
        object.__setattr__(self, "rotations", tuple(int(rotation) for rotation in (self.rotations or ())))
        object.__setattr__(self, "rotation_random_mode", str(self.rotation_random_mode))
        object.__setattr__(
            self,
            "rotation_key_limb_limits",
            {int(rotation): int(limbs) for rotation, limbs in (self.rotation_key_limb_limits or {}).items()},
        )


def generate_client_context(spec: CKKSContextSpec, device="cpu"):
    client_material, server_material = _sample_material(spec)
    client = Client(
        client_material,
        auto_load_keys=spec.auto_load_keys,
        rotation_key_limb_limits=spec.rotation_key_limb_limits,
    )
    context = _build_context(server_material, spec, device=device)
    return client, context


def _sample_material(spec: CKKSContextSpec):
    if not isinstance(spec, CKKSContextSpec):
        raise TypeError("generate_client_context expects a CKKSContextSpec")

    depth = int(spec.depth)

    rotation_indices = [] if spec.rotations is None else [int(rotation) for rotation in spec.rotations]
    sampler_config = CkksSamplerConfig(
        log_n=spec.log_n,
        depth=depth,
        dcrt_bits=spec.dcrt_bits,
        first_mod=spec.first_mod,
        dnum=spec.dnum,
        secret_key_dist=spec.secret_key_dist,
        rotation_key_limb_limits=dict(spec.rotation_key_limb_limits),
        random_mode="parallel_deterministic",
        rotation_random_mode=spec.rotation_random_mode,
    )
    client_material, server_material = sample_native_client_server(
        sampler_config,
        rotation_indices,
        slots=max(1, 1 << (int(spec.log_n) - 1)),
        scale_mode=spec.scale_mode,
        rescale_policy=spec.rescale_policy,
    )
    return client_material, server_material


def _build_context(server_material: NativeServerMaterial, spec: CKKSContextSpec, device="cpu"):
    if not isinstance(server_material, NativeServerMaterial):
        raise TypeError("_build_context expects native server material")
    from .context import Context

    builder = ContextMaterialBuilder.from_server(server_material)

    context = Context(
        builder.to_runtime_material(),
        device,
        auto_load_keys=spec.auto_load_keys,
        rotation_key_limb_limits=spec.rotation_key_limb_limits,
        native_context_gen=True,
        generation_metadata={
            "depth": int(server_material.depth),
            "logN": int(server_material.log_n),
            "dnum": int(server_material.dnum),
            "dcrtBits": int(server_material.dcrt_bits),
            "firstMod": int(server_material.special_mod),
            "secretKeyDist": server_material.secret_key_dist,
            "scaleMode": server_material.scale_mode,
            "rescalePolicy": server_material.rescale_policy,
        },
        roots_q=server_material.roots_q,
        roots_p=server_material.roots_p,
    )
    print(
        "Generated EasyFHE native context "
        f"(depth={server_material.depth}, rotation keys={len(server_material.rotation_keys)})."
    )
    return context


def _normalize_scale_mode(value):
    value = str(value).lower()
    if value not in {"fixed", "flexible"}:
        raise ValueError(f"scale_mode must be 'fixed' or 'flexible', got {value!r}")
    return value


def _normalize_rescale_policy(value):
    value = str(value).lower()
    if value not in {"manual", "auto"}:
        raise ValueError(f"rescale_policy must be 'manual' or 'auto', got {value!r}")
    return value
