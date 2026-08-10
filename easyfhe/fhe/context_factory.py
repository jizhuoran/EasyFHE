from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from .client import Client
from ._keygen.context_material_builder import ContextMaterialBuilder
from ._keygen.native_sampler import CkksSamplerConfig, NativeServerMaterial, sample_native_client_server


@dataclass(frozen=True, init=False)
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
    exact_q_primes: Optional[Sequence[int]] = None
    limb_specs: Optional[Sequence[int]] = None

    def __init__(
        self,
        depth=None,
        log_n=None,
        dnum=None,
        dcrt_bits=None,
        first_mod=None,
        secret_key_dist="SPARSE_TERNARY",
        scale_mode="fixed",
        rescale_policy="manual",
        rotations=(),
        auto_load_keys=None,
        rotation_random_mode="fresh",
        rotation_key_limb_limits=None,
        exact_q_primes=None,
        limb_specs=None,
    ):
        """Describe a u64 CKKS context.

        The legacy ``depth``/``dcrt_bits``/``first_mod`` form remains the
        shortest way to request a regular chain.  ``limb_specs`` is the
        explicit form: one integer bit-size per physical u64 Q prime, ordered
        from the first modulus to the last modulus.
        """
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "log_n", log_n)
        object.__setattr__(self, "dnum", dnum)
        object.__setattr__(self, "dcrt_bits", dcrt_bits)
        object.__setattr__(self, "first_mod", first_mod)
        object.__setattr__(self, "secret_key_dist", secret_key_dist)
        object.__setattr__(self, "scale_mode", scale_mode)
        object.__setattr__(self, "rescale_policy", rescale_policy)
        object.__setattr__(self, "rotations", rotations)
        object.__setattr__(self, "auto_load_keys", auto_load_keys)
        object.__setattr__(self, "rotation_random_mode", rotation_random_mode)
        object.__setattr__(self, "rotation_key_limb_limits", rotation_key_limb_limits or {})
        object.__setattr__(self, "exact_q_primes", exact_q_primes)
        object.__setattr__(self, "limb_specs", limb_specs)
        self.__post_init__()

    def __post_init__(self):
        if self.log_n is None:
            raise TypeError("CKKSContextSpec requires log_n")
        if self.dnum is None:
            raise TypeError("CKKSContextSpec requires dnum")
        object.__setattr__(self, "log_n", int(self.log_n))
        object.__setattr__(self, "dnum", int(self.dnum))

        limb_specs = None if self.limb_specs is None else tuple(self.limb_specs)
        if limb_specs is None:
            if self.depth is None:
                raise TypeError("CKKSContextSpec requires depth when limb_specs is not provided")
            if self.dcrt_bits is None:
                raise TypeError("CKKSContextSpec requires dcrt_bits when limb_specs is not provided")
            if self.first_mod is None:
                raise TypeError("CKKSContextSpec requires first_mod when limb_specs is not provided")
            depth = int(self.depth)
            dcrt_bits = int(self.dcrt_bits)
            first_mod = int(self.first_mod)
            plan = plan_prime_chain(
                limb_specs=(first_mod, *(dcrt_bits for _ in range(depth))),
                exact_q_primes=self.exact_q_primes,
            )
        else:
            if self.dcrt_bits is not None or self.first_mod is not None:
                raise ValueError("dcrt_bits and first_mod must not be provided together with limb_specs")
            plan = plan_prime_chain(limb_specs=limb_specs, exact_q_primes=self.exact_q_primes)
            if self.depth is not None and int(self.depth) != plan.depth:
                raise ValueError(
                    "depth does not match limb_specs: "
                    f"depth={int(self.depth)}, planned_depth={plan.depth}"
                )
            depth = plan.depth
            dcrt_bits = plan.dcrt_bit
            first_mod = plan.first_mod

        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "dcrt_bits", dcrt_bits)
        object.__setattr__(self, "first_mod", first_mod)
        object.__setattr__(self, "limb_specs", limb_specs)
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
        object.__setattr__(self, "exact_q_primes", plan.exact_q_primes)

        if self.scale_mode == "fixed" and len(set(plan.dcrt_bits[1:])) > 1:
            raise ValueError(
                "fixed scale_mode requires every rescale Q prime to use the same bit-size; "
                "use scale_mode='flexible' for a heterogeneous limb_specs chain"
            )

    @property
    def q_prime_bits(self):
        if self.limb_specs is not None:
            return tuple(int(bit) for bit in self.limb_specs)
        return (int(self.first_mod), *(int(self.dcrt_bits) for _ in range(int(self.depth))))


@dataclass(frozen=True)
class PrimeChainPlan:
    """An explicit physical u64 Q chain.

    EasyFHE's u64 backend always uses one physical prime per level and drops
    exactly one prime per rescale.  Composite limb specifications intentionally
    do not belong in this plan.
    """

    dcrt_bits: tuple[int, ...]
    exact_q_primes: Optional[tuple[int, ...]] = None

    @property
    def depth(self):
        return len(self.dcrt_bits) - 1

    @property
    def physical_limb_count(self):
        return len(self.dcrt_bits)

    @property
    def first_mod(self):
        return int(self.dcrt_bits[0])

    @property
    def dcrt_bit(self):
        return int(self.dcrt_bits[-1])

    @property
    def first_mod_limb_count(self):
        return 1

    @property
    def rescale_limb_count(self):
        return 1


def plan_prime_chain(*, limb_specs: Sequence[int], exact_q_primes: Optional[Sequence[int]] = None):
    """Normalize an explicit u64 Q-chain description without sampling it."""
    if not limb_specs:
        raise ValueError("plan_prime_chain requires at least one limb spec")

    bits = []
    for spec in limb_specs:
        if isinstance(spec, (tuple, list)):
            raise ValueError("u64 limb_specs entries must be scalar prime bit-sizes")
        bit = int(spec)
        if isinstance(spec, float) and not spec.is_integer():
            raise ValueError(f"u64 limb bit-size must be an integer, got {spec!r}")
        if not 1 <= bit <= 63:
            raise ValueError(f"u64 limb bit-size must be in [1, 63], got {bit}")
        bits.append(bit)

    exact = None if exact_q_primes is None else tuple(int(prime) for prime in exact_q_primes)
    if exact is not None:
        if len(exact) != len(bits):
            raise ValueError(f"exact_q_primes must contain {len(bits)} Q primes, got {len(exact)}")
        if any(prime <= 2 for prime in exact):
            raise ValueError("exact_q_primes must contain primes greater than 2")

    return PrimeChainPlan(tuple(bits), exact)


def generate_client_context(spec: CKKSContextSpec, device="cpu"):
    device = _validate_device(device)
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
        q_prime_bits=spec.q_prime_bits,
        exact_q_primes=spec.exact_q_primes,
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
            "qPrimeBits": tuple(int(bit) for bit in server_material.q_prime_bits),
            "firstMod": int(server_material.special_mod),
            "secretKeyDist": server_material.secret_key_dist,
            "scaleMode": server_material.scale_mode,
            "rescalePolicy": server_material.rescale_policy,
            "exactQPrimes": None if spec.exact_q_primes is None else tuple(spec.exact_q_primes),
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


def _validate_device(device):
    device = str(device)
    if device != "cpu" and device != "cuda" and not (
        device.startswith("cuda:") and device[5:].isdigit()
    ):
        raise ValueError(f"device must be 'cpu', 'cuda', or 'cuda:<index>', got {device!r}")
    return device
