from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


UIntArray = np.ndarray


def as_uint64_matrix(name: str, value: object) -> UIntArray:
    arr = np.asarray(value, dtype=np.uint64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a [limbs, N] uint64 matrix, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def as_uint64_tensor3(name: str, value: object) -> UIntArray:
    arr = np.asarray(value, dtype=np.uint64)
    if arr.ndim != 3:
        raise ValueError(f"{name} must be a rank-3 uint64 tensor, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def as_moduli_q(value: object, limbs: Optional[int] = None) -> UIntArray:
    arr = np.asarray(value, dtype=np.uint64).reshape(-1)
    if limbs is not None and arr.shape[0] < limbs:
        raise ValueError(f"Need at least {limbs} Q moduli, got {arr.shape[0]}")
    return np.ascontiguousarray(arr[:limbs] if limbs is not None else arr)


def mod_add(a: object, b: object, moduli_q: object) -> UIntArray:
    a = as_uint64_matrix("a", a)
    b = as_uint64_matrix("b", b)
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got {a.shape} and {b.shape}")

    moduli = as_moduli_q(moduli_q, a.shape[0])
    out = np.empty_like(a)
    for limb, modulus in enumerate(moduli):
        q = int(modulus)
        for idx in range(a.shape[1]):
            out[limb, idx] = (int(a[limb, idx]) + int(b[limb, idx])) % q
    return out


def mod_mul_add(a: object, b: object, c: object, moduli_q: object) -> UIntArray:
    a = as_uint64_matrix("a", a)
    b = as_uint64_matrix("b", b)
    c = as_uint64_matrix("c", c)
    if a.shape != b.shape or a.shape != c.shape:
        raise ValueError(f"a, b, c must have the same shape, got {a.shape}, {b.shape}, {c.shape}")

    moduli = as_moduli_q(moduli_q, a.shape[0])
    out = np.empty_like(a)
    for limb, modulus in enumerate(moduli):
        q = int(modulus)
        for idx in range(a.shape[1]):
            out[limb, idx] = (int(a[limb, idx]) * int(b[limb, idx]) + int(c[limb, idx])) % q
    return out


def _get_msb(value: int) -> int:
    return int(value).bit_length()


def _reverse_bits(value: int, width: int) -> int:
    result = 0
    for _ in range(width):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


def _root_table(root: int, ring_dim: int, modulus: int) -> list[int]:
    table = [0] * ring_dim
    x = 1
    msb = _get_msb(ring_dim - 1)
    for i in range(ring_dim):
        table[_reverse_bits(i, msb)] = x
        x = (x * root) % modulus
    return table


def inverse_ntt_from_eval(values: object, root: int, modulus: int) -> UIntArray:
    """Inverse NTT from the bit-reversed evaluation format used by EasyFHE."""

    out = np.asarray(values, dtype=np.uint64).reshape(-1).astype(object).tolist()
    n = len(out)
    if n == 0 or n & (n - 1):
        raise ValueError(f"iNTT requires a non-empty power-of-two length, got {n}")

    root_inv_table = _root_table(pow(int(root), int(modulus) - 2, int(modulus)), n, int(modulus))
    t = 1
    logt1 = 1
    m = n >> 1
    while m >= 1:
        for i in range(m):
            j1 = i << logt1
            j2 = j1 + t
            omega = root_inv_table[m + i]
            for index_lo in range(j1, j2):
                index_hi = index_lo + t
                lo_val = int(out[index_lo])
                hi_val = int(out[index_hi])
                omega_factor = lo_val
                if omega_factor < hi_val:
                    omega_factor += int(modulus)
                omega_factor -= hi_val
                out[index_lo] = (lo_val + hi_val) % int(modulus)
                out[index_hi] = (omega_factor * omega) % int(modulus)
        if m == 1:
            break
        t <<= 1
        logt1 += 1
        m >>= 1

    ring_dim_inv = pow(n, -1, int(modulus))
    return np.asarray([(int(x) * ring_dim_inv) % int(modulus) for x in out], dtype=np.uint64)


def _crt_interpolate_coeffs(eval_matrix: object, moduli_q: object, roots_q: object) -> tuple[list[int], int]:
    matrix = as_uint64_matrix("eval_matrix", eval_matrix)
    moduli = as_moduli_q(moduli_q, matrix.shape[0])
    roots = np.asarray(roots_q, dtype=np.uint64).reshape(-1)
    if roots.shape[0] < matrix.shape[0]:
        raise ValueError(f"Need at least {matrix.shape[0]} roots, got {roots.shape[0]}")

    coeff_limbs = [
        inverse_ntt_from_eval(matrix[limb], int(roots[limb]), int(moduli[limb]))
        for limb in range(matrix.shape[0])
    ]
    q = 1
    for modulus in moduli:
        q *= int(modulus)

    multipliers = []
    for modulus in moduli:
        qi = int(modulus)
        q_hat = q // qi
        multipliers.append(pow(q_hat % qi, -1, qi) * q_hat)

    coeffs: list[int] = []
    for coeff_idx in range(matrix.shape[1]):
        accum = 0
        for limb, multiplier in enumerate(multipliers):
            accum += int(coeff_limbs[limb][coeff_idx]) * multiplier
        coeffs.append(accum % q)
    return coeffs, q


def _bit_reverse_complex(vals: list[complex]) -> None:
    size = len(vals)
    j = 0
    for i in range(1, size):
        bit = size >> 1
        while j >= bit:
            j -= bit
            bit >>= 1
        j += bit
        if i < j:
            vals[i], vals[j] = vals[j], vals[i]


def _fft_special(vals: list[complex], cycl_order: int) -> None:
    vals_size = len(vals)
    rot_group = [0] * vals_size
    five_pows = 1
    for i in range(vals_size):
        rot_group[i] = five_pows
        five_pows = (five_pows * 5) % cycl_order
    ksi_pows = [complex(math.cos(2.0 * math.pi * j / cycl_order), math.sin(2.0 * math.pi * j / cycl_order))
                for j in range(cycl_order)]
    ksi_pows.append(ksi_pows[0])

    _bit_reverse_complex(vals)
    length = 2
    while length <= vals_size:
        lenh = length >> 1
        lenq = length << 2
        gap = cycl_order // lenq
        for i in range(0, vals_size, length):
            for j in range(lenh):
                idx = (rot_group[j] % lenq) * gap
                u = vals[i + j]
                v = vals[i + j + lenh] * ksi_pows[idx]
                vals[i + j] = u + v
                vals[i + j + lenh] = u - v
        length <<= 1


def _conjugate_slots(vals: list[complex]) -> list[complex]:
    result = [0j] * len(vals)
    if vals:
        result[0] = complex(vals[0].real, -vals[0].imag)
    for i in range(1, len(vals)):
        result[i] = complex(-vals[len(vals) - i].imag, -vals[len(vals) - i].real)
    return result


def decode_ckks_phase(
    phase: object,
    params: "CkksParams",
    *,
    plaintext_modulus_bits: int,
    noise_scale_deg: int = 1,
    scaling_factor: float | None = None,
    slots: int = 0,
) -> UIntArray:
    """Decode phase = ct0 + ct1*s into real CKKS slots."""

    del scaling_factor
    if params.roots_q is None:
        raise ValueError("CKKS phase decode requires roots_q in CkksParams")
    phase = as_uint64_matrix("phase", phase)
    ring_dim = phase.shape[1]
    nh = ring_dim // 2
    slots = slots or nh
    if slots > nh or nh % slots != 0:
        raise ValueError(f"invalid CKKS slots={slots} for ring_dim={ring_dim}")

    coeffs, q = _crt_interpolate_coeffs(phase, params.moduli_q, params.roots_q)
    q_half = q >> 1
    gap = nh // slots
    scaling_pre = 2.0 ** (-plaintext_modulus_bits * max(noise_scale_deg - 1, 0))
    cur_values: list[complex] = []
    for idx in range(0, slots * gap, gap):
        re = coeffs[idx]
        im = coeffs[idx + nh]
        real = -float(q - re) if re > q_half else float(re)
        imag = -float(q - im) if im > q_half else float(im)
        cur_values.append(complex(real * scaling_pre, imag * scaling_pre))

    conjugate = _conjugate_slots(cur_values)
    pow_p = 2.0 ** (-plaintext_modulus_bits)
    scale = 0.5 * pow_p
    real_values = [
        complex(scale * (cur_values[i].real + conjugate[i].real),
                scale * (cur_values[i].imag + conjugate[i].imag))
        for i in range(slots)
    ]
    _fft_special(real_values, ring_dim * 2)
    return np.asarray([value.real for value in real_values], dtype=np.float64)


def mod_sub_mul(e: object, a: object, s: object, moduli_q: object) -> UIntArray:
    e = as_uint64_matrix("e", e)
    a = as_uint64_matrix("a", a)
    s = as_uint64_matrix("s", s)
    if e.shape != a.shape or e.shape != s.shape:
        raise ValueError(f"e, a, s must have the same shape, got {e.shape}, {a.shape}, {s.shape}")

    moduli = as_moduli_q(moduli_q, e.shape[0])
    out = np.empty_like(e)
    for limb, modulus in enumerate(moduli):
        q = int(modulus)
        for idx in range(e.shape[1]):
            out[limb, idx] = (int(e[limb, idx]) - int(a[limb, idx]) * int(s[limb, idx])) % q
    return out


@dataclass(frozen=True)
class CkksParams:
    moduli_q: UIntArray
    roots_q: Optional[UIntArray] = None
    moduli_p: Optional[UIntArray] = None
    roots_p: Optional[UIntArray] = None
    scaling_factors: Optional[UIntArray] = None
    depth: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(self, "moduli_q", as_moduli_q(self.moduli_q))
        if self.roots_q is not None:
            object.__setattr__(self, "roots_q", np.asarray(self.roots_q, dtype=np.uint64))
        if self.moduli_p is not None:
            object.__setattr__(self, "moduli_p", np.asarray(self.moduli_p, dtype=np.uint64))
        if self.roots_p is not None:
            object.__setattr__(self, "roots_p", np.asarray(self.roots_p, dtype=np.uint64))
        if self.scaling_factors is not None:
            object.__setattr__(self, "scaling_factors", np.asarray(self.scaling_factors, dtype=np.float64))


@dataclass(frozen=True)
class KeygenSamples:
    sk: UIntArray
    pk_a: UIntArray
    pk_e: UIntArray

    def __post_init__(self):
        object.__setattr__(self, "sk", as_uint64_matrix("sk", self.sk))
        object.__setattr__(self, "pk_a", as_uint64_matrix("pk_a", self.pk_a))
        object.__setattr__(self, "pk_e", as_uint64_matrix("pk_e", self.pk_e))
        if self.sk.shape != self.pk_a.shape or self.sk.shape != self.pk_e.shape:
            raise ValueError("sk, pk_a, pk_e must have the same [limbs, N] shape")


@dataclass(frozen=True)
class KeyMaterial:
    sk: UIntArray
    pk_b: UIntArray
    pk_a: UIntArray

    def __post_init__(self):
        object.__setattr__(self, "sk", as_uint64_matrix("sk", self.sk))
        object.__setattr__(self, "pk_b", as_uint64_matrix("pk_b", self.pk_b))
        object.__setattr__(self, "pk_a", as_uint64_matrix("pk_a", self.pk_a))
        if self.sk.shape != self.pk_b.shape or self.sk.shape != self.pk_a.shape:
            raise ValueError("sk, pk_b, pk_a must have the same [limbs, N] shape")


@dataclass(frozen=True)
class EncryptSamples:
    v: UIntArray
    e0: UIntArray
    e1: UIntArray

    def __post_init__(self):
        object.__setattr__(self, "v", as_uint64_matrix("v", self.v))
        object.__setattr__(self, "e0", as_uint64_matrix("e0", self.e0))
        object.__setattr__(self, "e1", as_uint64_matrix("e1", self.e1))
        if self.v.shape != self.e0.shape or self.v.shape != self.e1.shape:
            raise ValueError("v, e0, e1 must have the same [limbs, N] shape")


@dataclass(frozen=True)
class CipherArrays:
    ct0: UIntArray
    ct1: UIntArray

    def __post_init__(self):
        object.__setattr__(self, "ct0", as_uint64_matrix("ct0", self.ct0))
        object.__setattr__(self, "ct1", as_uint64_matrix("ct1", self.ct1))
        if self.ct0.shape != self.ct1.shape:
            raise ValueError(f"ct0/ct1 shape mismatch: {self.ct0.shape} vs {self.ct1.shape}")

    def as_tuple(self) -> Tuple[UIntArray, UIntArray]:
        return self.ct0, self.ct1


@dataclass(frozen=True)
class SampleBundle:
    params: CkksParams
    keygen: KeygenSamples
    encrypt: Optional[EncryptSamples] = None
    ptx: Optional[UIntArray] = None
    expected_key: Optional[KeyMaterial] = None
    expected_zero: Optional[CipherArrays] = None
    expected_cipher: Optional[CipherArrays] = None
    expected_phase: Optional[UIntArray] = None
    secret_key_coeff: Optional[UIntArray] = None
    eval_mult_key_b: Optional[UIntArray] = None
    eval_mult_key_a: Optional[UIntArray] = None
    rotation_keys: Optional[object] = None
    decoded_real: Optional[UIntArray] = None

    def __post_init__(self):
        if self.ptx is not None:
            object.__setattr__(self, "ptx", as_uint64_matrix("ptx", self.ptx))
        if self.expected_phase is not None:
            object.__setattr__(self, "expected_phase", as_uint64_matrix("expected_phase", self.expected_phase))
        if self.secret_key_coeff is not None:
            object.__setattr__(self, "secret_key_coeff", np.asarray(self.secret_key_coeff, dtype=np.int64))
        if self.eval_mult_key_b is not None:
            object.__setattr__(self, "eval_mult_key_b", as_uint64_tensor3("eval_mult_key_b", self.eval_mult_key_b))
        if self.eval_mult_key_a is not None:
            object.__setattr__(self, "eval_mult_key_a", as_uint64_tensor3("eval_mult_key_a", self.eval_mult_key_a))
        if self.rotation_keys is not None:
            normalized = []
            for item in self.rotation_keys:
                auto_idx, key_b, key_a = item[:3]
                normalized_item = (
                    int(auto_idx),
                    np.ascontiguousarray(np.asarray(key_b, dtype=np.uint64)),
                    np.ascontiguousarray(np.asarray(key_a, dtype=np.uint64)),
                )
                if len(item) > 3:
                    normalized_item = normalized_item + tuple(item[3:])
                normalized.append(normalized_item)
            object.__setattr__(self, "rotation_keys", normalized)
        if self.decoded_real is not None:
            object.__setattr__(self, "decoded_real", np.asarray(self.decoded_real, dtype=np.float64))


@dataclass(frozen=True)
class FixtureSampleBundle(SampleBundle):
    ptx: UIntArray = None

    @classmethod
    def from_npz(cls, path: str | Path) -> "FixtureSampleBundle":
        fixture = np.load(path, allow_pickle=False)
        required = {"moduli_q", "sk", "pk_a", "pk_e", "ptx", "v", "e0", "e1"}
        missing = sorted(required - set(fixture.files))
        if missing:
            raise ValueError(f"{path} is missing required fixture arrays: {missing}")

        expected_key = None
        if {"pk_b"}.issubset(fixture.files):
            expected_key = KeyMaterial(sk=fixture["sk"], pk_b=fixture["pk_b"], pk_a=fixture["pk_a"])

        expected_zero = None
        if {"ct0_zero", "ct1_zero"}.issubset(fixture.files):
            expected_zero = CipherArrays(ct0=fixture["ct0_zero"], ct1=fixture["ct1_zero"])

        expected_cipher = None
        if "cipher" in fixture.files:
            expected_cipher = CipherArrays(ct0=fixture["cipher"][0], ct1=fixture["cipher"][1])

        expected_phase = fixture["phase"] if "phase" in fixture.files else None
        eval_mult_key_b = fixture["evalmult_key_b"] if "evalmult_key_b" in fixture.files else None
        eval_mult_key_a = fixture["evalmult_key_a"] if "evalmult_key_a" in fixture.files else None
        decoded_real = fixture["decoded_real"] if "decoded_real" in fixture.files else None

        return cls(
            params=CkksParams(
                moduli_q=fixture["moduli_q"],
                roots_q=fixture["roots_q"] if "roots_q" in fixture.files else None,
                moduli_p=fixture["moduli_p"] if "moduli_p" in fixture.files else None,
                roots_p=fixture["roots_p"] if "roots_p" in fixture.files else None,
            ),
            keygen=KeygenSamples(sk=fixture["sk"], pk_a=fixture["pk_a"], pk_e=fixture["pk_e"]),
            encrypt=EncryptSamples(v=fixture["v"], e0=fixture["e0"], e1=fixture["e1"]),
            ptx=fixture["ptx"],
            expected_key=expected_key,
            expected_zero=expected_zero,
            expected_cipher=expected_cipher,
            expected_phase=expected_phase,
            eval_mult_key_b=eval_mult_key_b,
            eval_mult_key_a=eval_mult_key_a,
            decoded_real=decoded_real,
        )


def keygen_from_samples(samples: KeygenSamples, params: CkksParams) -> KeyMaterial:
    pk_b = mod_sub_mul(samples.pk_e, samples.pk_a, samples.sk, params.moduli_q)
    return KeyMaterial(sk=samples.sk, pk_b=pk_b, pk_a=samples.pk_a)


def encrypt_zero_from_samples(key: KeyMaterial, samples: EncryptSamples, params: CkksParams) -> CipherArrays:
    ct0_zero = mod_mul_add(key.pk_b, samples.v, samples.e0, params.moduli_q)
    ct1_zero = mod_mul_add(key.pk_a, samples.v, samples.e1, params.moduli_q)
    return CipherArrays(ct0=ct0_zero, ct1=ct1_zero)


def encrypt_from_samples(ptx: object, key: KeyMaterial, samples: EncryptSamples, params: CkksParams) -> CipherArrays:
    ptx = as_uint64_matrix("ptx", ptx)
    zero = encrypt_zero_from_samples(key, samples, params)
    if ptx.shape != zero.ct0.shape:
        raise ValueError(f"ptx/cipher shape mismatch: {ptx.shape} vs {zero.ct0.shape}")
    return CipherArrays(ct0=mod_add(zero.ct0, ptx, params.moduli_q), ct1=zero.ct1)


def decrypt_phase_from_arrays(cipher: CipherArrays, key: KeyMaterial, params: CkksParams) -> UIntArray:
    sk = key.sk[: cipher.ct0.shape[0]]
    return mod_mul_add(cipher.ct1, sk, cipher.ct0, params.moduli_q)
