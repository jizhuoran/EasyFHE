import math
from functools import lru_cache

import numpy as np
import easyfhe as torch

from ..ciphertext import Plaintext, PreparedPlaintext
from ..runtime.instrumentation import run_instrumented_op

MAX_ENCODED_BITS = 61
ZERO_THRESHOLD = 1e-20


@lru_cache(maxsize=None)
def _prepare_plaintext_params(ring_dim):
    import cmath

    ring_dim = int(ring_dim)
    cycl_order = ring_dim << 1
    half_ring_dim = ring_dim >> 1

    five_pows = 1
    rot_group = np.empty(half_ring_dim, dtype=np.int32)
    for i in range(half_ring_dim):
        rot_group[i] = five_pows
        five_pows = (five_pows * 5) % cycl_order

    ksi_pows = np.asarray(
        [cmath.exp(1j * (2.0 * math.pi * j / cycl_order)) for j in range(cycl_order)],
        dtype=np.complex128,
    )
    ksi_pows = np.concatenate((ksi_pows, ksi_pows[:1]))
    rot_group.setflags(write=False)
    ksi_pows.setflags(write=False)
    return rot_group, ksi_pows


@lru_cache(maxsize=None)
def _bit_reverse_permutation(size):
    size = int(size)
    indices = np.arange(size)
    j = 0
    for i in range(1, size):
        bit = size >> 1
        while j >= bit:
            j -= bit
            bit >>= 1
        j += bit
        if i < j:
            indices[i], indices[j] = indices[j], indices[i]
    indices.setflags(write=False)
    return indices


def _as_complex_array(values):
    values = np.asarray(values)
    if np.iscomplexobj(values):
        return values.astype(np.complex128, copy=False)
    return values.astype(np.float64, copy=False).astype(np.complex128)


def _pad_1d(values, target_size, fill_value):
    if target_size < len(values):
        raise ValueError(f"The number of slots [{target_size}] is less than the size of data [{len(values)}]")
    return np.pad(
        values,
        pad_width=(0, target_size - len(values)),
        mode="constant",
        constant_values=fill_value,
    )


def _fft_special_inv(values, cycl_order, rot_group, ksi_pows):
    values_size = len(values)
    values = np.asarray(values, dtype=np.complex128).copy()

    len_size = values_size
    while len_size >= 1:
        len_h = len_size >> 1
        if len_h == 0:
            break
        len_q = len_size << 2
        gap = cycl_order // len_q
        indices = (len_q - (rot_group[:len_h] % len_q)) * gap
        roots = ksi_pows[indices]

        blocks = values.reshape(-1, len_size)
        left = blocks[:, :len_h].copy()
        right = blocks[:, len_h:].copy()
        blocks[:, :len_h] = left + right
        blocks[:, len_h:] = (left - right) * roots
        len_size >>= 1

    values = values[_bit_reverse_permutation(values_size)]
    return values / values_size


def _validate_prepared_slots(prepared, slots):
    if prepared.slots != slots:
        raise ValueError(f"Prepared plaintext slots [{prepared.slots}] do not match requested slots [{slots}]")


def _validate_scaled_range(prepared, scaling_factor):
    if prepared.max_encoded_value < ZERO_THRESHOLD:
        return
    scaled = int(prepared.max_encoded_value * scaling_factor)
    if scaled <= 0:
        return
    if math.log2(scaled) >= MAX_ENCODED_BITS:
        raise ValueError(
            f"Prepared plaintext is too large for encoding: "
            f"max_encoded_value={prepared.max_encoded_value}, scaling_factor={scaling_factor}"
        )


def _encoded_values_tensor(prepared, slots, cryptoContext):
    return torch.as_tensor(
        prepared.encoded_values,
        dtype=torch.float64,
        device=cryptoContext.device,
    ).reshape(-1, 2 * slots)


def prepare_plaintext(x, slots, ring_dim):
    if ring_dim is None:
        raise ValueError("prepare_plaintext requires an explicit ring_dim")
    if not isinstance(x, np.ndarray):
        raise TypeError(f"prepare_plaintext input must be np.ndarray, got {type(x)}")
    ring_dim = int(ring_dim)
    cycl_order = ring_dim << 1
    rot_group, ksi_pows = _prepare_plaintext_params(ring_dim)

    values = x
    inverse_complex = _pad_1d(_as_complex_array(values), slots, complex(0.0, 0.0))
    inverse_complex = _fft_special_inv(inverse_complex, cycl_order, rot_group, ksi_pows)
    encoded_values = np.ascontiguousarray(inverse_complex, dtype=np.complex128).view(np.float64)

    return PreparedPlaintext(
        _pad_1d(values, slots, 0.0),
        slots,
        encoded_values,
        np.max(np.abs(encoded_values)),
    )


def make_plaintext(prepared, level, slots, is_ext, cryptoContext):
    return run_instrumented_op(
        cryptoContext,
        "make_plaintext",
        _make_plaintext,
        prepared,
        level,
        slots,
        is_ext,
        cryptoContext,
    )


def encode(x, name, level, slots, is_ext, cryptoContext):
    return run_instrumented_op(
        cryptoContext,
        "encode",
        _encode,
        x,
        name,
        level,
        slots,
        is_ext,
        cryptoContext,
    )


def _encode(x, name, level, slots, is_ext, cryptoContext):
    if isinstance(x, np.ndarray):
        middle_value = prepare_plaintext(x, slots, cryptoContext.N)
    elif isinstance(x, PreparedPlaintext):
        _validate_prepared_slots(x, slots)
        middle_value = x
    else:
        raise TypeError(f"Invalid plaintext source type: {type(x)}")

    return _make_plaintext(middle_value, level, slots, is_ext, cryptoContext)


def _make_plaintext(middle_value, level, slots, is_ext, cryptoContext):
    if not isinstance(middle_value, PreparedPlaintext):
        raise TypeError(f"Invalid prepared plaintext type: {type(middle_value)}")
    _validate_prepared_slots(middle_value, slots)
    cur_limbs = cryptoContext.L - level
    scaling_factor = cryptoContext.scale_at(cur_limbs)
    _validate_scaled_range(middle_value, scaling_factor)

    pt_encode = torch.encode(
        input=_encoded_values_tensor(middle_value, slots, cryptoContext),
        N=cryptoContext.N,
        cur_limbs=cur_limbs,
        slots=slots,
        scaling_factor=scaling_factor,
        is_ext=is_ext,
        sizeP=cryptoContext.primes.shape[0] - cryptoContext.L,
        primes=cryptoContext.QplusP_map[cur_limbs],
        max_int_diffs=cryptoContext.QmaxdiffplusPmaxdiff_map[cur_limbs],
        barret_ratio=cryptoContext.QbarretRatioplusPbarretRatio_map[cur_limbs],
        barret_k=cryptoContext.QbarretKplusPbarretK_map[cur_limbs],
        power_of_roots_shoup=cryptoContext.power_of_roots_shoup,
        power_of_roots=cryptoContext.power_of_roots,
    )
    gpufhe_cipher = Plaintext(pt_encode.unsqueeze(0), cur_limbs, scaling_factor, 1, slots, is_ext)
    return gpufhe_cipher
