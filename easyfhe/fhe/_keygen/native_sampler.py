from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import easyfhe as torch


def _decode_rotation_tensors(tensors):
    if not tensors:
        return []
    manifest = np.ascontiguousarray(tensors[0].cpu().numpy()).astype(np.int64, copy=False).reshape(-1)
    index = 1
    rotation_keys = []
    for rot_idx in manifest:
        if index + 3 >= len(tensors):
            raise ValueError("native sampler returned an incomplete rotation-key tensor list")
        key_b = np.ascontiguousarray(tensors[index].cpu().numpy())
        key_a = np.ascontiguousarray(tensors[index + 1].cpu().numpy())
        auto_map = np.ascontiguousarray(tensors[index + 2].cpu().numpy())
        inverse_auto_map = np.ascontiguousarray(tensors[index + 3].cpu().numpy())
        index += 4
        rotation_keys.append((int(rot_idx), key_b, key_a, auto_map, inverse_auto_map))
    if index != len(tensors):
        raise ValueError("native sampler returned trailing tensors after rotation-key material")
    return rotation_keys


@dataclass(frozen=True)
class CkksSamplerConfig:
    log_n: int = 12
    depth: int = 2
    dcrt_bits: int = 50
    first_mod: int = 60
    first_mod_limb_count: int = 1
    dnum: int = 3
    secret_key_dist: str = "SPARSE_TERNARY"
    rotation_key_limb_limits: dict[int, int] | None = None
    random_mode: str = "sequential"
    rotation_random_mode: str = "fresh"
    q_prime_bits: tuple[int, ...] | None = None
    exact_q_primes: tuple[int, ...] | None = None

    def __post_init__(self):
        object.__setattr__(self, "log_n", int(self.log_n))
        object.__setattr__(self, "depth", int(self.depth))
        object.__setattr__(self, "dcrt_bits", int(self.dcrt_bits))
        object.__setattr__(self, "first_mod", int(self.first_mod))
        object.__setattr__(self, "first_mod_limb_count", int(self.first_mod_limb_count))
        object.__setattr__(self, "dnum", int(self.dnum))
        object.__setattr__(self, "secret_key_dist", str(self.secret_key_dist))
        object.__setattr__(self, "random_mode", str(self.random_mode))
        object.__setattr__(self, "rotation_random_mode", str(self.rotation_random_mode))
        object.__setattr__(self, "q_prime_bits", _optional_int_tuple(self.q_prime_bits))
        object.__setattr__(self, "exact_q_primes", _optional_int_tuple(self.exact_q_primes))

        if self.log_n <= 0:
            raise ValueError(f"log_n must be positive, got {self.log_n}")
        if self.depth < 0:
            raise ValueError(f"depth must be nonnegative, got {self.depth}")
        if self.dnum <= 0:
            raise ValueError(f"dnum must be positive, got {self.dnum}")
        if self.first_mod_limb_count != 1:
            raise ValueError("the u64 sampler requires first_mod_limb_count=1")

        expected = self.depth + 1
        if self.q_prime_bits is not None:
            if len(self.q_prime_bits) != expected:
                raise ValueError(f"q_prime_bits must contain {expected} bit-sizes, got {len(self.q_prime_bits)}")
            if any(not 1 <= bit <= 63 for bit in self.q_prime_bits):
                raise ValueError("q_prime_bits values must be in [1, 63]")
        if self.exact_q_primes is not None:
            if len(self.exact_q_primes) != expected:
                raise ValueError(f"exact_q_primes must contain {expected} Q primes, got {len(self.exact_q_primes)}")
            if any(prime <= 2 for prime in self.exact_q_primes):
                raise ValueError("exact_q_primes must contain primes greater than 2")


def _optional_int_tuple(values):
    return None if values is None else tuple(int(value) for value in values)


@dataclass(frozen=True)
class NativeContextBundle:
    moduli_q: np.ndarray
    roots_q: np.ndarray
    moduli_p: np.ndarray
    roots_p: np.ndarray
    secret_key: np.ndarray
    secret_key_coeff: np.ndarray
    public_key_b: np.ndarray
    public_key_a: np.ndarray
    eval_mult_key_b: np.ndarray
    eval_mult_key_a: np.ndarray


@dataclass(frozen=True)
class NativeClientMaterial:
    secret_key: np.ndarray
    public_key_b: np.ndarray
    public_key_a: np.ndarray
    log_n: int
    depth: int
    dcrt_bits: int
    q_prime_bits: tuple[int, ...]
    special_mod: int
    dnum: int
    secret_key_dist: str
    scale_mode: str
    rescale_policy: str
    moduli_q: np.ndarray
    roots_q: np.ndarray
    moduli_p: np.ndarray
    roots_p: np.ndarray
    eval_mult_key: np.ndarray


@dataclass(frozen=True)
class NativeServerMaterial:
    log_n: int
    depth: int
    dcrt_bits: int
    q_prime_bits: tuple[int, ...]
    special_mod: int
    dnum: int
    secret_key_dist: str
    scale_mode: str
    rescale_policy: str
    moduli_q: np.ndarray
    roots_q: np.ndarray
    moduli_p: np.ndarray
    roots_p: np.ndarray
    eval_mult_key: np.ndarray
    rotation_keys: tuple


def sample_native_context(
    config: CkksSamplerConfig,
    slots: int = 0,
) -> NativeContextBundle:
    values = torch.as_tensor([0.0], dtype=torch.float64, device="cpu")
    try:
        tensors = torch.fhe_native_sample_ckks(
            values,
            int(config.log_n),
            int(config.depth),
            int(config.dcrt_bits),
            int(config.first_mod),
            int(config.first_mod_limb_count),
            _q_prime_bits(config),
            torch.as_tensor(_exact_q_primes(config), dtype=torch.uint64, device="cpu"),
            int(config.dnum),
            str(config.secret_key_dist),
            True,
            False,
            1,
            0,
            int(slots),
            str(config.random_mode),
            3.19,
            1,
        )
    except TypeError:
        if config.exact_q_primes is not None or config.q_prime_bits is not None or int(config.first_mod_limb_count) != 1:
            raise
        tensors = torch.fhe_native_sample_ckks(
            values,
            int(config.log_n),
            int(config.depth),
            int(config.dcrt_bits),
            int(config.first_mod),
            int(config.dnum),
            str(config.secret_key_dist),
            True,
            False,
            1,
            0,
            int(slots),
            str(config.random_mode),
            3.19,
            1,
        )
    (
        moduli_q,
        roots_q,
        moduli_p,
        roots_p,
        secret_key,
        secret_key_coeff,
        public_key_b,
        public_key_a,
        eval_mult_key_b,
        eval_mult_key_a,
    ) = list(tensors)

    return NativeContextBundle(
        moduli_q=np.ascontiguousarray(moduli_q.cpu().numpy()),
        roots_q=np.ascontiguousarray(roots_q.cpu().numpy()),
        moduli_p=np.ascontiguousarray(moduli_p.cpu().numpy()),
        roots_p=np.ascontiguousarray(roots_p.cpu().numpy()),
        secret_key=np.ascontiguousarray(secret_key.cpu().numpy()),
        secret_key_coeff=np.ascontiguousarray(secret_key_coeff.cpu().numpy()),
        public_key_b=np.ascontiguousarray(public_key_b.cpu().numpy()),
        public_key_a=np.ascontiguousarray(public_key_a.cpu().numpy()),
        eval_mult_key_b=np.ascontiguousarray(eval_mult_key_b.cpu().numpy()),
        eval_mult_key_a=np.ascontiguousarray(eval_mult_key_a.cpu().numpy()),
    )


def split_native_client_server(
    config: CkksSamplerConfig,
    bundle: NativeContextBundle,
    rotation_keys,
    *,
    scale_mode,
    rescale_policy,
):
    eval_mult_key = np.asarray([bundle.eval_mult_key_b, bundle.eval_mult_key_a], dtype=np.uint64)
    q_prime_bits = _resolved_q_prime_bits(config)
    client = NativeClientMaterial(
        secret_key=bundle.secret_key,
        public_key_b=bundle.public_key_b,
        public_key_a=bundle.public_key_a,
        log_n=int(config.log_n),
        depth=int(config.depth),
        dcrt_bits=int(config.dcrt_bits),
        q_prime_bits=q_prime_bits,
        special_mod=int(config.first_mod),
        dnum=int(config.dnum),
        secret_key_dist=str(config.secret_key_dist),
        scale_mode=str(scale_mode),
        rescale_policy=str(rescale_policy),
        moduli_q=np.array(bundle.moduli_q, copy=True),
        roots_q=np.array(bundle.roots_q, copy=True),
        moduli_p=np.array(bundle.moduli_p, copy=True),
        roots_p=np.array(bundle.roots_p, copy=True),
        eval_mult_key=np.array(eval_mult_key, copy=True),
    )
    server = NativeServerMaterial(
        log_n=int(config.log_n),
        depth=int(config.depth),
        dcrt_bits=int(config.dcrt_bits),
        q_prime_bits=q_prime_bits,
        special_mod=int(config.first_mod),
        dnum=int(config.dnum),
        secret_key_dist=str(config.secret_key_dist),
        scale_mode=str(scale_mode),
        rescale_policy=str(rescale_policy),
        moduli_q=bundle.moduli_q,
        roots_q=bundle.roots_q,
        moduli_p=bundle.moduli_p,
        roots_p=bundle.roots_p,
        eval_mult_key=eval_mult_key,
        rotation_keys=tuple(rotation_keys),
    )
    return client, server


def _q_prime_bits(config):
    if config.q_prime_bits is None:
        return ()
    return tuple(int(bit) for bit in config.q_prime_bits)


def _resolved_q_prime_bits(config):
    bits = _q_prime_bits(config)
    if bits:
        return bits
    return (int(config.first_mod), *(int(config.dcrt_bits) for _ in range(int(config.depth))))


def _exact_q_primes(config):
    if config.exact_q_primes is None:
        return ()
    primes = tuple(int(prime) for prime in config.exact_q_primes)
    expected = int(config.depth) + 1
    if len(primes) != expected:
        raise ValueError(f"exact_q_primes must contain {expected} Q primes, got {len(primes)}")
    if any(prime <= 2 for prime in primes):
        raise ValueError("exact_q_primes must contain primes greater than 2")
    return primes


def sample_native_client_server(
    config: CkksSamplerConfig,
    rotation_indices=(),
    slots=0,
    *,
    scale_mode,
    rescale_policy,
):
    bundle = sample_native_context(config, slots=slots)
    rotation_keys = (
        sample_native_rotation_keys(
            config,
            bundle.secret_key,
            bundle.secret_key_coeff,
            rotation_indices,
        )
        if rotation_indices
        else ()
    )
    return split_native_client_server(
        config,
        bundle,
        rotation_keys,
        scale_mode=scale_mode,
        rescale_policy=rescale_policy,
    )


def sample_native_rotation_keys(
    config: CkksSamplerConfig,
    secret_key: object,
    secret_key_coeff: object,
    rotation_indices: object,
):
    rotation_indices = [int(index) for index in rotation_indices]
    rotation_offsets = [] if not rotation_indices else [0, len(rotation_indices)]
    trim_map = _rotation_trim_limbs_by_auto_index(
        rotation_indices,
        1 << int(config.log_n),
        config.rotation_key_limb_limits or {},
    )
    trim_auto_indices = [int(key) for key in trim_map]
    trim_limbs = [int(trim_map[key]) for key in trim_map]
    secret_key_tensor = torch.as_tensor(np.asarray(secret_key, dtype=np.uint64), dtype=torch.uint64, device="cpu")
    secret_key_coeff_tensor = torch.as_tensor(
        np.asarray(secret_key_coeff, dtype=np.int64),
        dtype=torch.int64,
        device="cpu",
    )
    try:
        tensors = torch.fhe_native_sample_rotation_keys(
            secret_key_tensor,
            secret_key_coeff_tensor,
            int(config.log_n),
            int(config.depth),
            int(config.dcrt_bits),
            int(config.first_mod),
            int(config.first_mod_limb_count),
            _q_prime_bits(config),
            torch.as_tensor(_exact_q_primes(config), dtype=torch.uint64, device="cpu"),
            int(config.dnum),
            str(config.secret_key_dist),
            rotation_indices,
            rotation_offsets,
            trim_auto_indices,
            trim_limbs,
            str(config.rotation_random_mode),
            str(config.random_mode),
            3.19,
            1,
        )
    except TypeError:
        if config.exact_q_primes is not None or config.q_prime_bits is not None or int(config.first_mod_limb_count) != 1:
            raise
        tensors = torch.fhe_native_sample_rotation_keys(
            secret_key_tensor,
            secret_key_coeff_tensor,
            int(config.log_n),
            int(config.depth),
            int(config.dcrt_bits),
            int(config.first_mod),
            int(config.dnum),
            str(config.secret_key_dist),
            rotation_indices,
            rotation_offsets,
            trim_auto_indices,
            trim_limbs,
            str(config.rotation_random_mode),
            str(config.random_mode),
            3.19,
            1,
        )
    return _decode_rotation_tensors(list(tensors))


def _rotation_trim_limbs_by_auto_index(rotation_indices, ring_dim, max_limbs_by_rotation):
    if not max_limbs_by_rotation:
        return {}
    cycl_order = int(ring_dim) << 1
    trim_map = {}
    for rotation in rotation_indices:
        normalized_rotation = _normalize_rotation_index(rotation, ring_dim)
        if normalized_rotation in max_limbs_by_rotation:
            auto_index = _find_auto_index_2n_complex(rotation, cycl_order)
            trim_map[int(auto_index)] = int(max_limbs_by_rotation[normalized_rotation])
    return trim_map


def _normalize_rotation_index(rotation, ring_dim):
    rotation = int(rotation)
    if rotation < 0:
        return int((int(ring_dim) // 2) + rotation)
    return rotation


def _find_auto_index_2n_complex(rotation, cycl_order):
    rotation = int(rotation)
    cycl_order = int(cycl_order)
    if rotation == 0:
        return 1
    if rotation == cycl_order - 1:
        return cycl_order - 1

    generator = _mod_inverse(5, cycl_order) if rotation < 0 else 5
    result = generator
    for _ in range(1, abs(rotation)):
        result = (result * generator) % cycl_order
    return int(result)


def _mod_inverse(value, modulus):
    old_r, r = int(value), int(modulus)
    old_s, s = 1, 0
    while r:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
    if old_r != 1:
        raise ValueError(f"{value} is not invertible modulo {modulus}")
    return old_s % modulus
