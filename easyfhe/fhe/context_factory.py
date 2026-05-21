from __future__ import annotations

import copy
from dataclasses import dataclass

from .runtime.options import RuntimeOptions
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


def generate_client_context(spec: CKKSContextSpec, device="cpu", options=None):
    client_material, server_material = _sample_material(spec, options)
    client = Client(client_material, copy.copy(options))
    context = _build_context(server_material, device=device, options=options)
    return client, context


def _sample_material(spec: CKKSContextSpec, options=None):
    if not isinstance(spec, CKKSContextSpec):
        raise TypeError("generate_client_context expects a CKKSContextSpec")
    if options is None:
        options = RuntimeOptions()

    depth = int(spec.depth)

    rotation_indices = [] if spec.rotations is None else [int(rotation) for rotation in spec.rotations]
    sampler_config = CkksSamplerConfig(
        log_n=spec.log_n,
        depth=depth,
        dcrt_bits=spec.dcrt_bits,
        first_mod=spec.first_mod,
        dnum=spec.dnum,
        secret_key_dist=spec.secret_key_dist,
        scale_mode=spec.scale_mode,
        rescale_policy=spec.rescale_policy,
        rotation_key_limb_limits=dict(getattr(options, "rotation_key_limb_limits", None) or {}),
        random_mode="parallel_deterministic",
        rotation_random_mode=str(getattr(options, "rotation_random_mode", "fresh")),
    )
    client_material, server_material = sample_native_client_server(
        sampler_config,
        rotation_indices,
        slots=max(1, 1 << (int(spec.log_n) - 1)),
    )
    return client_material, server_material


def _build_context(server_material: NativeServerMaterial, device="cpu", options=None):
    if not isinstance(server_material, NativeServerMaterial):
        raise TypeError("_build_context expects native server material")
    if options is None:
        options = RuntimeOptions()
    from .context import Context

    builder = ContextMaterialBuilder.from_server(server_material, copy.copy(options))

    context = Context(
        builder.to_runtime_material(),
        device,
        options,
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
    if value != "fixed":
        raise ValueError(f"scale_mode must be 'fixed', got {value!r}")
    return value


def _normalize_rescale_policy(value):
    value = str(value).lower()
    if value not in {"manual", "auto"}:
        raise ValueError(f"rescale_policy must be 'manual' or 'auto', got {value!r}")
    return value
