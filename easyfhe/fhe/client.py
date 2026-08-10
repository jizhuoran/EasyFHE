from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
import easyfhe as torch

from .ciphertext import Cipher, CipherState
from .ops import kernels as F
from .ops.encoding import encode_stage1, encode_stage2


class Client:
    def __init__(self, material, *, auto_load_keys=None, rotation_key_limb_limits=None):
        self.auto_load_keys = auto_load_keys
        self.rotation_key_limb_limits = dict(rotation_key_limb_limits or {})
        self._contexts = {}

        self.log_n = int(material.log_n)
        self.depth = int(material.depth)
        self.dcrt_bits = int(material.dcrt_bits)
        self.q_prime_bits = tuple(int(bit) for bit in material.q_prime_bits)
        self.special_mod = int(material.special_mod)
        self.dnum = int(material.dnum)
        self.secret_key_dist = str(material.secret_key_dist)
        self.scale_mode = str(material.scale_mode)
        self.rescale_policy = str(material.rescale_policy)
        self.moduli_q = _as_uint64_vector("moduli_q", material.moduli_q)
        self.roots_q = _as_uint64_vector("roots_q", material.roots_q)
        self.moduli_p = _as_uint64_vector("moduli_p", material.moduli_p)
        self.roots_p = _as_uint64_vector("roots_p", material.roots_p)
        self._validate_public_params()
        self.eval_mult_key = _as_eval_mult_key(
            material.eval_mult_key,
            dnum=self.dnum,
            limbs=len(self.moduli_q) + len(self.moduli_p),
            ring_dim=self.N,
        )

        self._secret_key = _as_uint64_matrix("secret_key", material.secret_key, shape=(len(self.moduli_q), self.N))
        public_key_b = _as_uint64_matrix("public_key_b", material.public_key_b, shape=(len(self.moduli_q), self.N))
        public_key_a = _as_uint64_matrix("public_key_a", material.public_key_a, shape=(len(self.moduli_q), self.N))
        self._public_key_b = _uint64_cpu_tensor(public_key_b)
        self._public_key_a = _uint64_cpu_tensor(public_key_a)
        self._public_key_device_cache = {"cpu": (self._public_key_b, self._public_key_a)}
        self._secret_key_cpu = _uint64_cpu_tensor(self._secret_key)
        self._secret_key_device_cache = {"cpu": self._secret_key_cpu}
        self._moduli_p_cpu = _uint64_cpu_tensor(self.moduli_p)
        self._moduli_q_cpu = _uint64_cpu_tensor(self.moduli_q)
        self._roots_q_cpu = _uint64_cpu_tensor(self.roots_q)
        self._crt_inv_moduli_cpu = _uint64_cpu_tensor(_crt_inverse_matrix(self.moduli_q))
        self._crt_inv_moduli_device_cache = {"cpu": self._crt_inv_moduli_cpu}
        self._params = _CkksParams(
            moduli_q=self.moduli_q,
            roots_q=self.roots_q,
        )

    def encrypt(self, x, *, device=None, scale_deg=1, level=0, slots=0, scaling_factor=None, cur_limbs=None):
        device = device or "cpu"
        context = self._context_for(device)
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        ptx = encode_stage2(
            encode_stage1(x, slots, cryptoContext=context),
            level=level,
            slots=slots,
            is_ext=False,
            cryptoContext=context,
            scaling_factor=scaling_factor,
            cur_limbs=cur_limbs,
        )
        ptx = _raise_plaintext_scale_degree(ptx, scale_deg, context)
        cur_limbs = ptx.state.cur_limbs
        pk0, pk1 = self._public_key_for_device(ptx.cv[0].device, cur_limbs)
        return _encrypt(
            pk0,
            pk1,
            ptx,
            device,
            context,
            self._moduli_p_cpu,
            self._moduli_q_cpu,
        )

    def decrypt(self, cipher, *, complex_output=False):
        if cipher.cv[0].is_cuda and not complex_output and _cuda_decode_supports_state(
            cipher,
            self.dcrt_bits,
        ):
            native_cuda_decoded = _decrypt_decode_cuda(
                cipher,
                self._secret_key_for_device(cipher.cv[0].device),
                self._crt_inv_moduli_for_device(cipher.cv[0].device),
                self._context_for(cipher.cv[0].device),
                plaintext_modulus_bits=self.dcrt_bits,
            )
            if native_cuda_decoded is not None:
                return native_cuda_decoded
        native_decoded = _decrypt_decode_native(
            cipher,
            self._secret_key_cpu,
            self._moduli_q_cpu,
            self._roots_q_cpu,
            plaintext_modulus_bits=self.dcrt_bits,
            complex_output=complex_output,
        )
        if native_decoded is not None:
            return native_decoded.to(cipher.cv[0].device)
        phase = _decrypt_phase(cipher, self._secret_key, self.moduli_q)
        decoded = _decode_ckks_phase(
            phase,
            self._params,
            plaintext_modulus_bits=self.dcrt_bits,
            noise_scale_deg=cipher.state.noise_deg,
            scaling_factor=cipher.state.scaling_factor,
            slots=getattr(cipher, "slots", 0),
        )
        return torch.tensor(decoded, device=cipher.cv[0].device, dtype=torch.float64)

    def _secret_key_for_device(self, device):
        key = str(device)
        cached = self._secret_key_device_cache.get(key)
        if cached is None:
            cached = self._secret_key_cpu.to(device).view(1, len(self.moduli_q), self.N)
            self._secret_key_device_cache[key] = cached
        return cached

    def _public_key_for_device(self, device, cur_limbs):
        key = str(device)
        cached = self._public_key_device_cache.get(key)
        if cached is None:
            cached = (self._public_key_b.to(device), self._public_key_a.to(device))
            self._public_key_device_cache[key] = cached
        return cached[0][:cur_limbs], cached[1][:cur_limbs]

    def _crt_inv_moduli_for_device(self, device):
        key = str(device)
        cached = self._crt_inv_moduli_device_cache.get(key)
        if cached is None:
            cached = self._crt_inv_moduli_cpu.to(device)
            self._crt_inv_moduli_device_cache[key] = cached
        return cached

    def _context_for(self, device):
        device = str(device)
        cached = self._contexts.get(device)
        if cached is not None:
            return cached

        from ._keygen.context_material_builder import ContextMaterialBuilder
        from .context import Context

        builder = ContextMaterialBuilder.from_public_params(
            log_n=self.log_n,
            depth=self.depth,
            dcrt_bits=self.q_prime_bits,
            special_mod=self.special_mod,
            dnum=self.dnum,
            secret_key_dist=self.secret_key_dist,
            scale_mode=self.scale_mode,
            rescale_policy=self.rescale_policy,
            moduli_q=self.moduli_q,
            roots_q=self.roots_q,
            moduli_p=self.moduli_p,
            roots_p=self.roots_p,
            eval_mult_key=self.eval_mult_key,
        )
        context = Context(
            builder.to_runtime_material(),
            device,
            auto_load_keys=self.auto_load_keys,
            rotation_key_limb_limits=self.rotation_key_limb_limits,
            native_context_gen=True,
            generation_metadata=self._generation_metadata(),
            roots_q=self.roots_q,
            roots_p=self.roots_p,
        )
        self._contexts[device] = context
        return context

    @property
    def N(self):
        return 1 << self.log_n

    def _validate_public_params(self):
        if self.log_n <= 0:
            raise ValueError(f"log_n must be positive, got {self.log_n}")
        if self.dnum <= 0:
            raise ValueError(f"dnum must be positive, got {self.dnum}")
        if self.moduli_q.size == 0:
            raise ValueError("moduli_q must not be empty")
        if self.moduli_p.size == 0:
            raise ValueError("moduli_p must not be empty")
        if self.roots_q.shape != self.moduli_q.shape:
            raise ValueError(f"roots_q shape must match moduli_q, got {self.roots_q.shape} vs {self.moduli_q.shape}")
        if self.roots_p.shape != self.moduli_p.shape:
            raise ValueError(f"roots_p shape must match moduli_p, got {self.roots_p.shape} vs {self.moduli_p.shape}")

    def _generation_metadata(self):
        return {
            "depth": self.depth,
            "logN": self.log_n,
            "dnum": self.dnum,
            "dcrtBits": self.dcrt_bits,
            "qPrimeBits": self.q_prime_bits,
            "firstMod": self.special_mod,
            "secretKeyDist": self.secret_key_dist,
            "scaleMode": self.scale_mode,
            "rescalePolicy": self.rescale_policy,
        }

    def __repr__(self):
        return (
            "<Client "
            f"log_n={self.log_n} depth={self.depth} "
            f"dnum={self.dnum} dcrt_bits={self.dcrt_bits} "
            f"scale_mode={self.scale_mode} "
            f"rescale_policy={self.rescale_policy}>"
        )


def _uint64_cpu_tensor(value):
    return torch.as_tensor(value, dtype=torch.uint64, device="cpu").contiguous()


def _as_uint64_vector(name, value):
    arr = np.asarray(value, dtype=np.uint64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D uint64 vector, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def _as_uint64_matrix(name, value, *, shape=None):
    arr = np.asarray(value, dtype=np.uint64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a [limbs, N] uint64 matrix, got shape {arr.shape}")
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return np.ascontiguousarray(arr)


def _as_eval_mult_key(value, *, dnum, limbs, ring_dim):
    arr = np.asarray(value, dtype=np.uint64)
    expected = (2, int(dnum), int(limbs), int(ring_dim))
    if arr.size != np.prod(expected):
        raise ValueError(f"eval_mult_key must contain {np.prod(expected)} uint64 values for shape {expected}, got {arr.shape}")
    return np.ascontiguousarray(arr.reshape(expected))


def _as_moduli_q(value, limbs=None):
    arr = np.asarray(value, dtype=np.uint64).reshape(-1)
    if limbs is not None and arr.shape[0] < limbs:
        raise ValueError(f"Need at least {limbs} Q moduli, got {arr.shape[0]}")
    return np.ascontiguousarray(arr[:limbs] if limbs is not None else arr)


def _crt_inverse_matrix(moduli_q):
    moduli = _as_moduli_q(moduli_q)
    result = np.zeros((len(moduli), len(moduli)), dtype=np.uint64)
    for i, modulus_i in enumerate(moduli):
        qi = int(modulus_i)
        for j in range(i):
            result[i, j] = pow(int(moduli[j]) % qi, -1, qi)
    return np.ascontiguousarray(result)


def _encrypt(pk0, pk1, ptx, device, context, moduli_p, moduli_q):
    logn = context.logN
    cur_limbs = ptx.state.cur_limbs
    l = cur_limbs
    nh = context.N // 2

    target_device = device
    cv = F.cv_encrypt(
        ptx.cv[0],
        pk0,
        pk1,
        l,
        logn,
        nh,
        moduli_p,
        moduli_q,
        context,
    )
    bx, ax = cv
    n = 1 << logn
    if bx.numel() != l * n:
        raise RuntimeError(f"Unexpected encrypted tensor size: got {bx.numel()}, expected {l * n}")
    return Cipher(
        [
            bx.view(1, l, n).to(target_device),
            ax.view(1, l, n).to(target_device),
        ],
        ptx.state,
        ptx.slots,
        is_ext=False,
    )


def _raise_plaintext_scale_degree(ptx, scale_deg, context):
    if scale_deg == ptx.state.noise_deg:
        return ptx
    if scale_deg < 1 or ptx.state.noise_deg != 1:
        raise ValueError(f"unsupported plaintext scale degree transition: {ptx.state.noise_deg} -> {scale_deg}")

    cur_limbs = ptx.state.cur_limbs
    base_scale = context.scale_at(cur_limbs)
    base_scale_int = round(base_scale)
    scale_multiplier = [
        pow(base_scale_int, scale_deg - 1, int(context.moduliQ_scalar[i]))
        for i in range(cur_limbs)
    ]
    scale_multiplier = F.gen_scalar_tensor(
        scale_multiplier,
        context.moduliQ_scalar,
        cur_limbs,
    ).to(ptx.cv[0].device)
    cv = [
        F.cv_mul_scalar(
            ptx.cv[0],
            scale_multiplier,
            context.moduliQ,
            context.q_mu,
            cur_limbs,
        )
    ]
    return ptx.cipher_like(
        cv,
        state=CipherState(cur_limbs, scale_deg, base_scale ** scale_deg),
    )


def _decrypt_decode_native(
    cipher,
    secret_key,
    moduli_q,
    roots_q,
    *,
    plaintext_modulus_bits,
    complex_output=False,
):
    if len(cipher.cv) != 2:
        raise ValueError(f"Expected a degree-1 ciphertext with two components, got {len(cipher.cv)}")
    ct0_tensor = cipher.cv[0]
    ct1_tensor = cipher.cv[1]
    if ct0_tensor.dim() == 3:
        if int(ct0_tensor.shape[0]) != 1:
            raise ValueError("decrypt currently expects batch_size=1")
        ct0_tensor = ct0_tensor[0]
        ct1_tensor = ct1_tensor[0]
    return F.cv_decrypt_decode(
        ct0_tensor[: cipher.state.cur_limbs],
        ct1_tensor[: cipher.state.cur_limbs],
        secret_key[: cipher.state.cur_limbs],
        moduli_q,
        roots_q,
        cipher.state.cur_limbs,
        plaintext_modulus_bits,
        cipher.state.noise_deg,
        getattr(cipher, "slots", 0),
        0.0 if cipher.state.scaling_factor is None else float(cipher.state.scaling_factor),
        complex_output=complex_output,
    )


def _cuda_decode_supports_state(cipher, plaintext_modulus_bits):
    scaling_factor = cipher.state.scaling_factor
    if scaling_factor is None:
        return True
    nominal = 2.0 ** (int(plaintext_modulus_bits) * int(cipher.state.noise_deg))
    return math.isclose(float(scaling_factor), nominal, rel_tol=1e-12, abs_tol=0.0)


def _decrypt_decode_cuda(cipher, secret_key, crt_inv_moduli, context, *, plaintext_modulus_bits):
    if len(cipher.cv) != 2:
        raise ValueError(f"Expected a degree-1 ciphertext with two components, got {len(cipher.cv)}")
    ct0_tensor = cipher.cv[0]
    ct1_tensor = cipher.cv[1]
    if ct0_tensor.dim() != 3 or ct1_tensor.dim() != 3:
        raise ValueError("CUDA decrypt currently expects [batch, limbs, N] ciphertext components")
    if int(ct0_tensor.shape[0]) != 1:
        raise ValueError("decrypt currently expects batch_size=1")

    cur_limbs = cipher.state.cur_limbs
    product = F.cv_mul(
        ct1_tensor[:, :cur_limbs, :],
        secret_key[:, :cur_limbs, :],
        context.moduliQ,
        context.q_mu,
        cur_limbs,
    )
    phase = F.cv_add(
        product,
        ct0_tensor[:, :cur_limbs, :],
        context.moduliQ,
        cur_limbs,
    )[0]
    return F.cv_decode_phase_cuda(
        phase,
        context.moduliQ,
        crt_inv_moduli[:cur_limbs, :cur_limbs].contiguous(),
        cur_limbs,
        plaintext_modulus_bits,
        cipher.state.noise_deg,
        getattr(cipher, "slots", 0) or context.Nh,
        context,
    )


def _decrypt_phase(cipher, secret_key, moduli_q):
    if len(cipher.cv) != 2:
        raise ValueError(f"Expected a degree-1 ciphertext with two components, got {len(cipher.cv)}")
    ct0_tensor = cipher.cv[0]
    ct1_tensor = cipher.cv[1]
    if ct0_tensor.dim() == 3:
        if int(ct0_tensor.shape[0]) != 1:
            raise ValueError("decrypt currently expects batch_size=1")
        ct0_tensor = ct0_tensor[0]
        ct1_tensor = ct1_tensor[0]
    ct0 = _as_uint64_matrix("ct0", ct0_tensor.detach().cpu().numpy())[: cipher.state.cur_limbs]
    ct1 = _as_uint64_matrix("ct1", ct1_tensor.detach().cpu().numpy())[: cipher.state.cur_limbs]
    if ct0.shape != ct1.shape:
        raise ValueError(f"ct0/ct1 shape mismatch: {ct0.shape} vs {ct1.shape}")

    sk = _as_uint64_matrix("secret_key", secret_key)[: cipher.state.cur_limbs]
    if sk.shape != ct0.shape:
        raise ValueError(f"secret key/cipher shape mismatch: {sk.shape} vs {ct0.shape}")

    moduli = _as_moduli_q(moduli_q, ct0.shape[0])
    phase = np.empty_like(ct0)
    for limb, modulus in enumerate(moduli):
        q = int(modulus)
        for idx in range(ct0.shape[1]):
            phase[limb, idx] = (int(ct1[limb, idx]) * int(sk[limb, idx]) + int(ct0[limb, idx])) % q
    return phase


@dataclass(frozen=True)
class _CkksParams:
    moduli_q: np.ndarray
    roots_q: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "moduli_q", _as_moduli_q(self.moduli_q))
        object.__setattr__(self, "roots_q", np.asarray(self.roots_q, dtype=np.uint64).reshape(-1))


def _decode_ckks_phase(
    phase,
    params,
    *,
    plaintext_modulus_bits,
    noise_scale_deg=1,
    scaling_factor=None,
    slots=0,
):
    del scaling_factor
    phase = _as_uint64_matrix("phase", phase)
    ring_dim = phase.shape[1]
    nh = ring_dim // 2
    slots = slots or nh
    if slots > nh or nh % slots != 0:
        raise ValueError(f"invalid CKKS slots={slots} for ring_dim={ring_dim}")

    coeffs, q = _crt_interpolate_coeffs(phase, params.moduli_q, params.roots_q)
    q_half = q >> 1
    gap = nh // slots
    scaling_pre = 2.0 ** (-plaintext_modulus_bits * max(noise_scale_deg - 1, 0))
    cur_values = []
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
        complex(
            scale * (cur_values[i].real + conjugate[i].real),
            scale * (cur_values[i].imag + conjugate[i].imag),
        )
        for i in range(slots)
    ]
    _fft_special(real_values, ring_dim * 2)
    return np.asarray([value.real for value in real_values], dtype=np.float64)


def _crt_interpolate_coeffs(eval_matrix, moduli_q, roots_q):
    matrix = _as_uint64_matrix("eval_matrix", eval_matrix)
    moduli = _as_moduli_q(moduli_q, matrix.shape[0])
    roots = np.asarray(roots_q, dtype=np.uint64).reshape(-1)
    if roots.shape[0] < matrix.shape[0]:
        raise ValueError(f"Need at least {matrix.shape[0]} roots, got {roots.shape[0]}")

    coeff_limbs = [
        _inverse_ntt_from_eval(matrix[limb], int(roots[limb]), int(moduli[limb]))
        for limb in range(matrix.shape[0])
    ]
    q, multipliers = _crt_multipliers(tuple(int(modulus) for modulus in moduli))

    coeffs = []
    for coeff_idx in range(matrix.shape[1]):
        accum = 0
        for limb, multiplier in enumerate(multipliers):
            accum += int(coeff_limbs[limb][coeff_idx]) * multiplier
        coeffs.append(accum % q)
    return coeffs, q


@lru_cache(maxsize=None)
def _crt_multipliers(moduli):
    q = 1
    for modulus in moduli:
        q *= int(modulus)

    multipliers = []
    for modulus in moduli:
        qi = int(modulus)
        q_hat = q // qi
        multipliers.append(pow(q_hat % qi, -1, qi) * q_hat)
    return q, tuple(multipliers)


def _inverse_ntt_from_eval(values, root, modulus):
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


@lru_cache(maxsize=None)
def _root_table(root, ring_dim, modulus):
    table = [0] * ring_dim
    x = 1
    msb = int(ring_dim - 1).bit_length()
    for i in range(ring_dim):
        table[_reverse_bits(i, msb)] = x
        x = (x * root) % modulus
    return tuple(table)


def _reverse_bits(value, width):
    result = 0
    for _ in range(width):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


def _bit_reverse_complex(vals):
    for i, j in _bit_reverse_swaps(len(vals)):
        vals[i], vals[j] = vals[j], vals[i]


@lru_cache(maxsize=None)
def _bit_reverse_swaps(size):
    swaps = []
    j = 0
    for i in range(1, int(size)):
        bit = int(size) >> 1
        while j >= bit:
            j -= bit
            bit >>= 1
        j += bit
        if i < j:
            swaps.append((i, j))
    return tuple(swaps)


def _fft_special(vals, cycl_order):
    vals_size = len(vals)
    rot_group, ksi_pows = _fft_special_tables(vals_size, cycl_order)

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


@lru_cache(maxsize=None)
def _fft_special_tables(vals_size, cycl_order):
    rot_group = [0] * vals_size
    five_pows = 1
    for i in range(vals_size):
        rot_group[i] = five_pows
        five_pows = (five_pows * 5) % cycl_order
    ksi_pows = [
        complex(math.cos(2.0 * math.pi * j / cycl_order), math.sin(2.0 * math.pi * j / cycl_order))
        for j in range(cycl_order)
    ]
    ksi_pows.append(ksi_pows[0])
    return tuple(rot_group), tuple(ksi_pows)


def _conjugate_slots(vals):
    result = [0j] * len(vals)
    if vals:
        result[0] = complex(vals[0].real, -vals[0].imag)
    for i in range(1, len(vals)):
        result[i] = complex(-vals[len(vals) - i].imag, -vals[len(vals) - i].real)
    return result
