import math
from functools import lru_cache

import numpy as np
import easyfhe as torch

from ..ciphertext import CipherState, Plaintext
from . import kernels as F

MAX_ENCODED_BITS = 61
ZERO_THRESHOLD = 1e-20


class PreparedPlaintext:
    def __init__(self, values, slots, encoded_values, max_encoded_value, packed=False):
        self.values = values
        self.slots = slots
        self.encoded_values = encoded_values
        self.max_encoded_value = max_encoded_value
        self.packed = bool(packed)

    def deep_copy(self):
        if torch.is_tensor(self.values):
            values = self.values.clone()
        else:
            values = self.values.copy()
        if torch.is_tensor(self.encoded_values):
            encoded_values = self.encoded_values.clone()
        else:
            encoded_values = np.array(self.encoded_values, copy=True)
        return PreparedPlaintext(
            values,
            self.slots,
            encoded_values,
            self.max_encoded_value,
            packed=self.packed,
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


def encode_stage1(raw_values, slots, ring_dim=None, device=None, cryptoContext=None):
    """Encode raw slot values into a contiguous middle representation."""

    slots = int(slots)
    params = _resolve_stage1_params(slots, ring_dim, device, cryptoContext)
    device = params["device"]

    if _use_torch_stage1(device):
        output_values, stage1_values, is_batch = _raw_cuda_matrices(raw_values, slots)
        encoded_values, max_encoded_value = _native_pre_encode_stage1_matrix(
            stage1_values,
            slots,
            params["cryptoContext"],
            device,
        )
    else:
        rows, is_batch = _raw_rows(raw_values)
        values = []
        for row in rows:
            padded = _pad_1d(row, slots, 0.0)
            values.append(padded)

        complex_rows = [
            _pad_1d(_as_complex_array(row), slots, complex(0.0, 0.0))
            for row in rows
        ]
        encoded_values = []
        max_encoded_value = 0.0
        for row in complex_rows:
            encoded = _fft_special_inv(
                row,
                params["cycl_order"],
                params["rot_group_np"],
                params["ksi_pows_np"],
                params["bitrev_np"],
            )
            encoded = np.ascontiguousarray(encoded, dtype=np.complex128).view(np.float64)
            encoded_values.append(encoded)
            if encoded.size:
                max_encoded_value = max(max_encoded_value, float(np.max(np.abs(encoded))))
        encoded_values = _as_encoded_tensor(np.stack(encoded_values, axis=0), device)
        output_values = np.stack(values, axis=0)

    if is_batch:
        return PreparedPlaintext(output_values, slots, encoded_values, max_encoded_value)
    return PreparedPlaintext(output_values[0], slots, encoded_values[0], max_encoded_value)


def encode_stage1_packed(packed_values, slots=None, cryptoContext=None):
    """Encode CUDA-packed complex slot values without Python-side padding."""

    if cryptoContext is None:
        raise ValueError("encode_stage1_packed requires cryptoContext")
    if not torch.is_tensor(packed_values):
        raise TypeError("encode_stage1_packed expects a CUDA tensor")
    if not packed_values.is_cuda:
        raise ValueError("encode_stage1_packed expects a CUDA tensor")
    if packed_values.dim() not in (1, 2):
        raise ValueError("encode_stage1_packed expects a rank-1 or rank-2 tensor")
    if not _is_complex_tensor(packed_values):
        raise TypeError("encode_stage1_packed expects complex32, complex64, or complex128 input")

    inferred_slots = int(packed_values.shape[-1])
    slots = inferred_slots if slots is None else int(slots)
    if slots != inferred_slots:
        raise ValueError(f"Packed values last dimension [{inferred_slots}] does not match slots [{slots}]")

    encoded_values, max_encoded_value = F.cv_pre_encode_stage1(
        packed_values,
        slots,
        cryptoContext,
    )
    if packed_values.dim() == 1:
        encoded_values = encoded_values[0]
    return PreparedPlaintext(
        packed_values,
        slots,
        encoded_values,
        max_encoded_value,
        packed=True,
    )


def encode_stage2(middle, level, slots, is_ext, cryptoContext, *, scaling_factor=None, cur_limbs=None):
    """Materialize a middle representation as an FHE plaintext."""

    if not isinstance(middle, PreparedPlaintext):
        raise TypeError(f"encode_stage2 expected PreparedPlaintext, got {type(middle)}")
    slots = int(slots)
    is_ext = bool(is_ext)
    _validate_prepared_slots(middle, slots)

    cur_limbs = _resolve_cur_limbs(cryptoContext, level, cur_limbs, "encode_stage2")
    scaling_factor = _resolve_stage2_scale(
        cryptoContext,
        cur_limbs,
        scaling_factor,
        "encode_stage2",
    )
    _validate_scaled_range(middle, scaling_factor)

    encoded_values = middle.encoded_values
    batch_size = 1 if encoded_values.dim() == 1 else int(encoded_values.shape[0])
    encoded_values = encoded_values.reshape(batch_size, 2 * slots)

    pt_encode = F.cv_encode(
        encoded_values,
        cryptoContext.N,
        cur_limbs,
        slots,
        scaling_factor,
        is_ext,
        cryptoContext,
    )
    return Plaintext(
        [pt_encode],
        CipherState(cur_limbs, 1, scaling_factor),
        slots,
        is_ext,
        batch_size=batch_size,
    )


def _as_encoded_tensor(encoded_values, device):
    return torch.as_tensor(encoded_values, dtype=torch.float64, device=device or "cpu")


def _resolve_stage1_params(slots, ring_dim, device, cryptoContext):
    log_slots = int(math.log2(int(slots))) if int(slots) > 0 else 0
    if cryptoContext is not None:
        ring_dim = int(cryptoContext.N if ring_dim is None else ring_dim)
        device = getattr(cryptoContext, "device", device)
        if not _has_context_encode_tables(cryptoContext):
            rot_group, ksi_pows = _prepare_plaintext_params(ring_dim)
            return {
                "device": device,
                "cycl_order": ring_dim << 1,
                "rot_group_np": rot_group,
                "ksi_pows_np": ksi_pows,
                "bitrev_np": None,
                "cryptoContext": None,
            }
        if ring_dim != int(cryptoContext.N):
            raise ValueError(
                f"encode_stage1 ring_dim [{ring_dim}] does not match context N [{cryptoContext.N}]"
            )
        return {
            "device": device,
            "cycl_order": int(cryptoContext.M),
            "rot_group_np": _table_numpy(cryptoContext.encode_params_rotGroup),
            "ksi_pows_np": _table_numpy(cryptoContext.encode_params_ksiPows),
            "bitrev_np": _bitrev_numpy_from_context(cryptoContext, log_slots),
            "cryptoContext": cryptoContext,
        }

    if ring_dim is None:
        raise ValueError("encode_stage1 requires ring_dim or cryptoContext")
    if _use_torch_stage1(device):
        raise ValueError("CUDA encode_stage1 requires cryptoContext")
    ring_dim = int(ring_dim)
    rot_group, ksi_pows = _prepare_plaintext_params(ring_dim)
    return {
        "device": device,
        "cycl_order": ring_dim << 1,
        "rot_group_np": rot_group,
        "ksi_pows_np": ksi_pows,
        "bitrev_np": None,
        "cryptoContext": None,
    }


def _has_context_encode_tables(cryptoContext):
    return all(
        hasattr(cryptoContext, name)
        for name in (
            "M",
            "encode_params_rotGroup",
            "encode_params_ksiPows",
            "encode_bitrev_indices",
        )
    )


def _table_numpy(value):
    if torch.is_tensor(value):
        return value.cpu().numpy()
    return np.asarray(value)


def _bitrev_numpy_from_context(cryptoContext, log_slots):
    indices = cryptoContext.encode_bitrev_indices.get(log_slots)
    if indices is None:
        return None
    return _table_numpy(indices).astype(np.intp, copy=False)


def _use_torch_stage1(device):
    return device is not None and str(device) != "cpu"


def _is_complex_tensor(value):
    return value.dtype in tuple(
        dtype
        for dtype in (
            getattr(torch, "complex32", None),
            torch.complex64,
            torch.complex128,
        )
        if dtype is not None
    )


def _native_pre_encode_stage1_matrix(values, slots, cryptoContext, device):
    values = torch.as_tensor(
        values,
        device=device,
    )
    return F.cv_pre_encode_stage1(values, slots, cryptoContext)


def _raw_matrix(raw_values, slots):
    if isinstance(raw_values, PreparedPlaintext):
        raise TypeError("encode_stage1 expects raw values, not PreparedPlaintext")
    if isinstance(raw_values, (list, tuple)) and not raw_values:
        raise ValueError("encode_stage1 requires at least one row")

    values = np.asarray(raw_values)
    if values.dtype == object:
        rows, is_batch = _raw_rows(raw_values)
        return np.stack([_pad_1d(row, slots, 0.0) for row in rows], axis=0), is_batch

    is_batch = values.ndim >= 2
    if is_batch:
        matrix = values.reshape(values.shape[0], -1)
    else:
        matrix = values.reshape(1, -1)

    if slots < matrix.shape[1]:
        raise ValueError(f"The number of slots [{slots}] is less than the size of data [{matrix.shape[1]}]")

    dtype = np.complex128 if np.iscomplexobj(matrix) else np.double
    output = np.zeros((matrix.shape[0], slots), dtype=dtype)
    output[:, : matrix.shape[1]] = matrix.astype(dtype, copy=False)
    return output, is_batch


def _raw_cuda_matrices(raw_values, slots):
    if isinstance(raw_values, PreparedPlaintext):
        raise TypeError("encode_stage1 expects raw values, not PreparedPlaintext")
    if isinstance(raw_values, (list, tuple)) and not raw_values:
        raise ValueError("encode_stage1 requires at least one row")

    values = np.asarray(raw_values)
    if values.dtype == object:
        output, is_batch = _raw_matrix(raw_values, slots)
        return output, np.ascontiguousarray(output, dtype=np.complex128), is_batch

    is_batch = values.ndim >= 2
    if is_batch:
        matrix = values.reshape(values.shape[0], -1)
    else:
        matrix = values.reshape(1, -1)

    if slots < matrix.shape[1]:
        raise ValueError(f"The number of slots [{slots}] is less than the size of data [{matrix.shape[1]}]")

    output_dtype = np.complex128 if np.iscomplexobj(matrix) else np.double
    output = np.zeros((matrix.shape[0], slots), dtype=output_dtype)
    stage1 = np.zeros((matrix.shape[0], slots), dtype=np.complex128)
    output[:, : matrix.shape[1]] = matrix.astype(output_dtype, copy=False)
    stage1[:, : matrix.shape[1]] = matrix.astype(np.complex128, copy=False)
    return output, stage1, is_batch


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


def _fft_special_inv(values, cycl_order, rot_group, ksi_pows, bitrev_indices=None):
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

    if bitrev_indices is None:
        bitrev_indices = _bit_reverse_permutation(values_size)
    values = values[bitrev_indices]
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


def _resolve_stage2_scale(cryptoContext, cur_limbs, scaling_factor, op_name):
    if scaling_factor is None:
        if _is_flexible_context(cryptoContext):
            raise ValueError(f"{op_name} requires scaling_factor in flexible scale mode")
        scaling_factor = cryptoContext.scale_at(cur_limbs)
    scaling_factor = float(scaling_factor)
    if scaling_factor <= 0:
        raise ValueError(f"{op_name} scaling_factor must be positive, got {scaling_factor}")
    return scaling_factor


def _is_flexible_context(cryptoContext):
    return str(getattr(cryptoContext, "scale_mode", "")).lower() == "flexible"


def _resolve_cur_limbs(cryptoContext, level, cur_limbs, op_name):
    if cur_limbs is None:
        if level is None:
            raise ValueError(f"{op_name} requires either level or cur_limbs")
        return int(cryptoContext.L) - int(level)
    cur_limbs = int(cur_limbs)
    if cur_limbs <= 0:
        raise ValueError(f"{op_name} cur_limbs must be positive, got {cur_limbs}")
    if level is not None:
        expected = int(cryptoContext.L) - int(level)
        if expected != cur_limbs:
            raise ValueError(
                f"{op_name} received inconsistent level and cur_limbs: "
                f"level={level} implies cur_limbs={expected}, got cur_limbs={cur_limbs}"
            )
    return cur_limbs
