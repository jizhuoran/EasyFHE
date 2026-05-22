import math
from functools import lru_cache

import numpy as np
import easyfhe as torch

from ..ciphertext import Plaintext
from . import kernels as F

MAX_ENCODED_BITS = 61
ZERO_THRESHOLD = 1e-20


class PreparedPlaintext:
    def __init__(self, values, slots, encoded_values, max_encoded_value):
        self.values = values
        self.slots = slots
        self.encoded_values = encoded_values
        self.max_encoded_value = max_encoded_value

    def deep_copy(self):
        if torch.is_tensor(self.encoded_values):
            encoded_values = self.encoded_values.clone()
        else:
            encoded_values = np.array(self.encoded_values, copy=True)
        return PreparedPlaintext(
            self.values.copy(),
            self.slots,
            encoded_values,
            self.max_encoded_value,
        )


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


def encode_stage1(raw_values, slots, ring_dim):
    """Encode raw slot values into a contiguous middle representation."""

    if ring_dim is None:
        raise ValueError("encode_stage1 requires an explicit ring_dim")
    slots = int(slots)
    ring_dim = int(ring_dim)
    cycl_order = ring_dim << 1
    rot_group, ksi_pows = _prepare_plaintext_params(ring_dim)

    rows, is_batch = _raw_rows(raw_values)
    values = []
    encoded_values = []
    max_encoded_value = 0.0
    for row in rows:
        padded = _pad_1d(row, slots, 0.0)
        encoded = _fft_special_inv(
            _pad_1d(_as_complex_array(row), slots, complex(0.0, 0.0)),
            cycl_order,
            rot_group,
            ksi_pows,
        )
        encoded = np.ascontiguousarray(encoded, dtype=np.complex128).view(np.float64)
        values.append(padded)
        encoded_values.append(encoded)
        if encoded.size:
            max_encoded_value = max(max_encoded_value, float(np.max(np.abs(encoded))))

    if is_batch:
        return PreparedPlaintext(
            np.stack(values, axis=0),
            slots,
            np.stack(encoded_values, axis=0),
            max_encoded_value,
        )
    return PreparedPlaintext(values[0], slots, encoded_values[0], max_encoded_value)


def encode_stage2(middle, level, slots, is_ext, cryptoContext):
    """Materialize a middle representation as an FHE plaintext."""

    if not isinstance(middle, PreparedPlaintext):
        raise TypeError(f"encode_stage2 expected PreparedPlaintext, got {type(middle)}")
    level = int(level)
    slots = int(slots)
    is_ext = bool(is_ext)
    _validate_prepared_slots(middle, slots)

    cur_limbs = cryptoContext.L - level
    scaling_factor = cryptoContext.scale_at(cur_limbs)
    _validate_scaled_range(middle, scaling_factor)

    encoded_values = np.asarray(middle.encoded_values)
    batch_size = 1 if encoded_values.ndim == 1 else int(encoded_values.shape[0])
    encoded_values = encoded_values.reshape(batch_size, 2 * slots)

    pt_encode = F.cv_encode(
        torch.as_tensor(encoded_values, dtype=torch.float64, device=cryptoContext.device),
        cryptoContext.N,
        cur_limbs,
        slots,
        scaling_factor,
        is_ext,
        cryptoContext,
    )
    return Plaintext([pt_encode], cur_limbs, scaling_factor, 1, slots, is_ext, batch_size=batch_size)


def _raw_rows(raw_values):
    if isinstance(raw_values, PreparedPlaintext):
        raise TypeError("encode_stage1 expects raw values, not PreparedPlaintext")
    if isinstance(raw_values, (list, tuple)):
        if not raw_values:
            raise ValueError("encode_stage1 requires at least one row")
        if np.asarray(raw_values).ndim == 1 and not isinstance(raw_values[0], (list, tuple, np.ndarray)):
            return [_raw_values(raw_values)], False
        rows = [_raw_values(row) for row in raw_values]
        return rows, True

    values = np.asarray(raw_values)
    if values.ndim >= 2:
        flat = values.reshape(values.shape[0], -1)
        return [_raw_values(row) for row in flat], True
    return [_raw_values(values)], False


def _raw_values(values):
    values = np.asarray(values).reshape(-1)
    if np.iscomplexobj(values):
        return np.asarray(values, dtype=np.complex128).reshape(-1)
    return np.asarray(values, dtype=np.double).reshape(-1)


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
    if int(prepared.slots) != int(slots):
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
