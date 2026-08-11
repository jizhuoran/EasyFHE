from __future__ import annotations

import math
import numpy as np
import easyfhe as torch

from .ciphertext import CipherState, EncodedScalar
from .ops.encoding import PreparedPlaintext, encode_stage1, encode_stage1_packed, encode_stage2


_CACHE_MODES = {"none", "middle", "plain", "both", "mix_of_middle_plain"}
_PLAIN_CACHE_POLICIES = {"first_fit", "small_first"}
_MAX_ENCODED_BITS = 61
_MAX_SCALAR_LOG_STEP = 60


class PackedRaw:
    """Slot-ready tensor source for ConstantBundle."""

    def __init__(self, tensor):
        if not torch.is_tensor(tensor):
            raise TypeError(f"PackedRaw tensor must be an easyfhe tensor, got {type(tensor)}")
        self.tensor = tensor

    def packed_tensor(self, slots, context=None):
        _validate_tensor_slots(self.tensor, slots, "PackedRaw")
        return self.tensor


def encode_scalar(
    value,
    *,
    cur_limbs,
    scale_degree,
    context,
    mode="scaled",
    absolute=False,
    scaling_factor=None,
):
    """Encode one scalar with explicit CKKS scale metadata."""

    bundle = ConstantBundle(scalars={"value": value}, cache_mode="none")
    return bundle.encoded_scalars(
        "value",
        cur_limbs=cur_limbs,
        scale_degree=scale_degree,
        context=context,
        mode=mode,
        absolute=absolute,
        scaling_factor=scaling_factor,
        cache=False,
    )[0]


class ConstantBundle:
    """Named scalar/vector constants with shared plaintext caching."""

    def __init__(
        self,
        *,
        scalars=None,
        vectors=None,
        cache_mode="plain",
        plain_cache_limit_gb=None,
        plain_cache_policy="first_fit",
    ):
        self._scalars = dict(scalars or {})
        self._vectors = _validate_vectors(vectors or {})
        self.cache_mode = _validate_cache_mode(cache_mode)
        self._plain_cache_limit_bytes = _cache_limit_gb_to_bytes(plain_cache_limit_gb)
        _validate_plain_cache_limit_mode(self.cache_mode, self._plain_cache_limit_bytes)
        self.plain_cache_policy = _validate_plain_cache_policy(plain_cache_policy)
        self._middle_cache = {}
        self._plain_cache = {}
        self._plain_cache_bytes_by_key = {}
        self._plain_middle_key_by_plain_key = {}
        self._plain_middle_key_counts = {}
        self._middle_crypto_context_by_key = {}
        self._plain_cache_bytes = 0
        self._scalar_cache = {}
        self._cache_stats = {
            "middle_hits": 0,
            "middle_misses": 0,
            "plain_hits": 0,
            "plain_misses": 0,
            "plain_cache_evictions": 0,
            "plain_cache_skips": 0,
            "scalar_hits": 0,
            "scalar_misses": 0,
        }

    def __len__(self):
        return len(self._vectors)

    def encoded_scalars(
        self,
        names,
        *,
        cur_limbs,
        scale_degree,
        context,
        mode="scaled",
        absolute=False,
        cache=True,
        scaling_factor=None,
    ):
        if isinstance(names, str):
            names = (names,)
        else:
            names = tuple(names)
        if not names:
            raise ValueError("ConstantBundle.encoded_scalars requires at least one scalar name")

        cur_limbs = int(cur_limbs)
        scale_degree = int(scale_degree)
        mode = _normalize_scalar_mode(mode)
        if mode == "scaled" and scale_degree < 1:
            raise ValueError("scaled scalar encoding requires scale_degree >= 1")
        if mode == "integer" and scale_degree != 0:
            raise ValueError("integer scalar encoding requires scale_degree=0")
        if mode == "scaled":
            scaling_factor = _resolve_scaled_scalar_scale(
                context,
                cur_limbs,
                scale_degree,
                scaling_factor,
            )
        else:
            scaling_factor = 1.0
        key = _scalar_key(names, cur_limbs, scale_degree, context, mode, absolute, scaling_factor)
        if cache and self.cache_mode != "none" and key in self._scalar_cache:
            self._cache_stats["scalar_hits"] += 1
            return self._scalar_cache[key]

        self._cache_stats["scalar_misses"] += 1
        encoded = self._encode_scalar_values(
            [self._scalar_value(name) for name in names],
            cur_limbs,
            scale_degree,
            context,
            mode,
            absolute,
            scaling_factor,
        )
        encoded = EncodedScalar(encoded, cur_limbs, scale_degree, scaling_factor)
        if cache and _cache_plain(self.cache_mode):
            self._scalar_cache[key] = encoded
        return encoded

    def _scalar_value(self, name):
        return self._scalars[name]

    def set_cache_mode(self, cache_mode, clear=True):
        cache_mode = _validate_cache_mode(cache_mode)
        _validate_plain_cache_limit_mode(cache_mode, self._plain_cache_limit_bytes)
        self.cache_mode = cache_mode
        if clear:
            self.clear_cache()

    def set_plain_cache_limit_gb(self, limit_gb, clear=False):
        limit_bytes = _cache_limit_gb_to_bytes(limit_gb)
        _validate_plain_cache_limit_mode(self.cache_mode, limit_bytes)
        self._plain_cache_limit_bytes = limit_bytes
        if clear:
            self._clear_plain_cache()

    def set_plain_cache_limit_bytes(self, limit_bytes, clear=False):
        limit_bytes = _cache_limit_bytes(limit_bytes)
        _validate_plain_cache_limit_mode(self.cache_mode, limit_bytes)
        self._plain_cache_limit_bytes = limit_bytes
        if clear:
            self._clear_plain_cache()

    def set_plain_cache_policy(self, policy):
        self.plain_cache_policy = _validate_plain_cache_policy(policy)

    def clear_cache(self):
        self._middle_cache.clear()
        self._clear_plain_cache()
        self._scalar_cache.clear()

    def _clear_plain_cache(self):
        self._plain_cache.clear()
        self._plain_cache_bytes_by_key.clear()
        self._plain_middle_key_by_plain_key.clear()
        self._plain_middle_key_counts.clear()
        self._middle_crypto_context_by_key.clear()
        self._plain_cache_bytes = 0

    def cache_info(self):
        middle_bytes = _cache_nbytes(self._middle_cache)
        plain_bytes = self._plain_cache_bytes
        scalar_bytes = _cache_nbytes(self._scalar_cache)
        plain_remaining = _cache_remaining_bytes(self._plain_cache_limit_bytes, plain_bytes)
        return {
            "mode": self.cache_mode,
            "plain_cache_policy": self.plain_cache_policy,
            "middle_entries": len(self._middle_cache),
            "plain_entries": len(self._plain_cache),
            "scalar_entries": len(self._scalar_cache),
            "middle_bytes": middle_bytes,
            "plain_bytes": plain_bytes,
            "scalar_bytes": scalar_bytes,
            "total_bytes": middle_bytes + plain_bytes + scalar_bytes,
            "plain_cache_limit_bytes": self._plain_cache_limit_bytes,
            "plain_cache_remaining_bytes": plain_remaining,
            "plain_cache_full": plain_remaining == 0 if self._plain_cache_limit_bytes is not None else False,
            **self._cache_stats,
        }

    def memory_info(self):
        info = self.cache_info()
        return {
            key: info[key]
            for key in (
                "middle_bytes",
                "plain_bytes",
                "scalar_bytes",
                "total_bytes",
                "plain_cache_limit_bytes",
                "plain_cache_remaining_bytes",
            )
        }

    def _encode_scalar_values(self, values, cur_limbs, scale_degree, context, mode, absolute, scaling_factor):
        rows = []
        for value in values:
            value = abs(value) if absolute else value
            if mode == "scaled":
                rows.append(_encode_scaled_scalar_value(value, cur_limbs, scale_degree, context, scaling_factor))
            else:
                encoded = int(value)
                rows.append(
                    [
                        int(encoded) % int(context.moduliQ_scalar[level])
                        for level in range(cur_limbs)
                    ]
                )
        return torch.from_numpy(np.asarray(rows, dtype=np.uint64)).to(context.device)

    def _encoded_middle(self, name, slots=None, context=None, ring_dim=None, cache_on_miss=True):
        ring_dim = _resolve_ring_dim(context, ring_dim)
        if name not in self._vectors:
            raise KeyError(f"constant vector {name!r} is missing")
        vector = self._vectors[name]
        slots = _resolve_slots(vector, slots)

        key = _middle_key(name, slots, ring_dim, _device_key(context))
        if self.cache_mode != "none" and key in self._middle_cache:
            self._cache_stats["middle_hits"] += 1
            return key, self._middle_cache[key]

        self._cache_stats["middle_misses"] += 1
        middle = _prepare_vector(
            vector,
            slots,
            ring_dim,
            _device_key(context),
            context=context,
        )
        if cache_on_miss and _cache_middle(self.cache_mode):
            self._middle_cache[key] = middle
        return key, middle

    def plaintext(
        self,
        name,
        *,
        state,
        slots,
        context,
        is_ext=False,
        cache=True,
    ):
        if not isinstance(name, str):
            raise TypeError(f"ConstantBundle.plaintext name must be str, got {type(name)}")
        if not isinstance(state, CipherState):
            raise TypeError(f"ConstantBundle.plaintext state must be CipherState, got {type(state)}")
        if state.scaling_factor is None:
            raise ValueError("ConstantBundle.plaintext state.scaling_factor is required")
        return self._plaintext_single(name, state, slots, context, is_ext, cache)

    def _plaintext_single(self, name, state, slots, context, is_ext, cache):
        plain_key = _plain_key(name, state, slots, context, is_ext)
        if cache and self.cache_mode != "none" and plain_key in self._plain_cache:
            self._cache_stats["plain_hits"] += 1
            return self._plain_cache[plain_key]

        self._cache_stats["plain_misses"] += 1
        middle_key, middle = self._encoded_middle(
            name,
            slots,
            context,
            cache_on_miss=self.cache_mode in {"middle", "both", "mix_of_middle_plain"},
        )
        had_sibling_plain = self._has_plain_for_middle(middle_key)
        plaintext = encode_stage2(
            middle,
            level=None,
            slots=slots,
            is_ext=is_ext,
            context=context,
            cur_limbs=state.cur_limbs,
            scale_degree=state.scale_degree,
            scaling_factor=state.scaling_factor,
        )
        cached_plain = self._maybe_cache_plain(plain_key, middle_key, plaintext, cache, context)
        if cache and self.cache_mode == "both":
            self._middle_cache.setdefault(middle_key, middle)
        elif cache and self.cache_mode == "mix_of_middle_plain":
            if cached_plain and not had_sibling_plain:
                self._middle_cache.pop(middle_key, None)
            elif cached_plain:
                self._middle_cache.setdefault(middle_key, middle)
            elif not self._has_plain_for_middle(middle_key):
                self._middle_cache.setdefault(middle_key, middle)
        return plaintext

    def _maybe_cache_plain(self, plain_key, middle_key, plaintext, cache, context):
        if not (cache and _cache_plain(self.cache_mode)):
            return False
        plain_bytes = _object_nbytes(plaintext)
        if self.cache_mode == "mix_of_middle_plain" and self._plain_cache_limit_bytes is not None:
            if plain_bytes > self._plain_cache_limit_bytes:
                self._cache_stats["plain_cache_skips"] += 1
                return False
            if self._plain_cache_bytes + plain_bytes > self._plain_cache_limit_bytes:
                if self.plain_cache_policy != "small_first" or not self._evict_for_smaller_plain(plain_bytes):
                    self._cache_stats["plain_cache_skips"] += 1
                    return False
        self._store_plain(plain_key, middle_key, plaintext, plain_bytes, context)
        return True

    def _evict_for_smaller_plain(self, plain_bytes):
        if not self._plain_cache_bytes_by_key:
            return False
        evict_key, evict_bytes = max(self._plain_cache_bytes_by_key.items(), key=lambda item: item[1])
        if plain_bytes >= evict_bytes:
            return False
        self._evict_plain(evict_key)
        return True

    def _store_plain(self, plain_key, middle_key, plaintext, plain_bytes, context):
        self._plain_cache[plain_key] = plaintext
        self._plain_cache_bytes_by_key[plain_key] = plain_bytes
        self._plain_middle_key_by_plain_key[plain_key] = middle_key
        self._plain_middle_key_counts[middle_key] = self._plain_middle_key_counts.get(middle_key, 0) + 1
        self._middle_crypto_context_by_key[middle_key] = context
        self._plain_cache_bytes += plain_bytes

    def _evict_plain(self, plain_key):
        plain_bytes = self._plain_cache_bytes_by_key.pop(plain_key)
        middle_key = self._plain_middle_key_by_plain_key.pop(plain_key)
        count = self._plain_middle_key_counts[middle_key] - 1
        if count:
            self._plain_middle_key_counts[middle_key] = count
            restore_context = self._middle_crypto_context_by_key.get(middle_key)
        else:
            del self._plain_middle_key_counts[middle_key]
            restore_context = self._middle_crypto_context_by_key.pop(middle_key, None)
        del self._plain_cache[plain_key]
        self._plain_cache_bytes -= plain_bytes
        self._cache_stats["plain_cache_evictions"] += 1
        if _cache_middle(self.cache_mode) and not self._has_plain_for_middle(middle_key):
            self._restore_middle(middle_key, restore_context)

    def _has_plain_for_middle(self, middle_key):
        return self._plain_middle_key_counts.get(middle_key, 0) > 0

    def _restore_middle(self, middle_key, context=None):
        if middle_key in self._middle_cache:
            return
        name, slots, ring_dim, device = middle_key
        vector = self._vectors[name]
        self._middle_cache[middle_key] = _prepare_vector(
            vector,
            slots,
            ring_dim,
            device,
            context=context,
        )


def _validate_cache_mode(cache_mode):
    if cache_mode not in _CACHE_MODES:
        raise ValueError(f"cache_mode must be one of {sorted(_CACHE_MODES)}, got {cache_mode!r}")
    return cache_mode


def _validate_vectors(vectors):
    result = dict(vectors)
    for name, vector in result.items():
        if not isinstance(vector, (PackedRaw, PreparedPlaintext)):
            raise TypeError(
                f"ConstantBundle vector {name!r} must be PackedRaw or PreparedPlaintext, got {type(vector)}"
            )
    return result


def _validate_plain_cache_limit_mode(cache_mode, limit_bytes):
    if limit_bytes is not None and cache_mode != "mix_of_middle_plain":
        raise ValueError("plain_cache_limit is only supported with cache_mode='mix_of_middle_plain'")


def _validate_plain_cache_policy(policy):
    if policy not in _PLAIN_CACHE_POLICIES:
        raise ValueError(f"plain_cache_policy must be one of {sorted(_PLAIN_CACHE_POLICIES)}, got {policy!r}")
    return policy


def _cache_limit_gb_to_bytes(limit_gb):
    if limit_gb is None:
        return None
    return _cache_limit_bytes(float(limit_gb) * (1024 ** 3))


def _cache_limit_bytes(limit_bytes):
    if limit_bytes is None:
        return None
    limit_bytes = int(limit_bytes)
    if limit_bytes < 0:
        raise ValueError(f"cache limit must be non-negative, got {limit_bytes}")
    return limit_bytes


def _cache_remaining_bytes(limit_bytes, used_bytes):
    if limit_bytes is None:
        return None
    return max(0, int(limit_bytes) - int(used_bytes))


def _cache_middle(cache_mode):
    return cache_mode in {"middle", "both", "mix_of_middle_plain"}


def _cache_plain(cache_mode):
    return cache_mode in {"plain", "both", "mix_of_middle_plain"}


def _cache_nbytes(cache):
    return sum(_object_nbytes(value) for value in cache.values())


def _object_nbytes(value):
    if isinstance(value, EncodedScalar):
        return _array_nbytes(value.residues)
    if hasattr(value, "numel") or hasattr(value, "nbytes"):
        return _array_nbytes(value)
    total = 0
    for attr in ("values", "encoded_values"):
        array = getattr(value, attr, None)
        if array is None:
            continue
        total += _array_nbytes(array)
    for cv in getattr(value, "cv", ()):
        total += _array_nbytes(cv)
    return total


def _array_nbytes(array):
    if hasattr(array, "numel") and hasattr(array, "element_size"):
        return array.numel() * array.element_size()
    if hasattr(array, "nbytes"):
        return array.nbytes
    return 0


def _context_key(context):
    return (
        id(context),
        getattr(context, "device", None),
        getattr(context, "N", None),
        getattr(context, "L", None),
        getattr(context, "scale_mode", None),
        getattr(context, "rescale_policy", None),
    )


def _device_key(context):
    if context is None:
        return None
    device = getattr(context, "device", None)
    return None if device is None else str(device)


def _middle_key(name, slots, ring_dim, device):
    return (name, slots, ring_dim, device)


def _plain_key(name, state, slots, context, is_ext):
    return (
        name,
        slots,
        is_ext,
        _context_key(context),
        int(state.cur_limbs),
        int(state.scale_degree),
        float(state.scaling_factor),
    )


def _scalar_key(names, cur_limbs, scale_degree, context, mode, absolute, scaling_factor):
    return (
        names,
        cur_limbs,
        scale_degree,
        mode,
        bool(absolute),
        _context_key(context),
        None if mode != "scaled" else float(scaling_factor),
    )


def _normalize_scalar_mode(mode):
    mode = str(mode)
    if mode not in {"scaled", "integer"}:
        raise ValueError(f"scalar mode must be 'scaled' or 'integer', got {mode!r}")
    return mode


def _encode_scaled_scalar_value(value, cur_limbs, scale_degree, context, scaling_factor):
    sc_factor = float(scaling_factor)

    log_approx = 0
    magnitude = math.fabs(value * sc_factor)
    if magnitude > 0:
        log_sf = int(math.ceil(math.log2(magnitude)))
        log_valid = min(log_sf, _MAX_ENCODED_BITS)
        log_approx = log_sf - log_valid

    approx_factor = float(2**log_approx)
    sc_constant = int(value * sc_factor / approx_factor + 0.5)
    crt_constant = cur_limbs * [sc_constant]

    if log_approx > 0:
        log_step = min(log_approx, _MAX_SCALAR_LOG_STEP)
        int_step = 2**log_step
        crt_approx = cur_limbs * [int_step]
        log_approx -= log_step

        while log_approx > 0:
            log_step = min(log_approx, _MAX_SCALAR_LOG_STEP)
            int_step = 2**log_step
            crt_sf = cur_limbs * [int_step]
            crt_approx = _crt_mult(crt_approx, crt_sf, context.moduliQ_scalar)
            log_approx -= log_step

        crt_constant = _crt_mult(crt_constant, crt_approx, context.moduliQ_scalar)

    return [
        int(item) % int(context.moduliQ_scalar[level])
        for level, item in enumerate(crt_constant)
    ]


def _crt_mult(xs, ys, mods):
    return [(int(x) * int(y)) % int(mod) for x, y, mod in zip(xs, ys, mods)]


def _resolve_scaled_scalar_scale(context, cur_limbs, scale_degree, scaling_factor):
    if scaling_factor is None:
        if int(scale_degree) != 1:
            raise ValueError(
                "ConstantBundle.encoded_scalars requires scaling_factor when scale_degree != 1"
            )
        if _is_flexible_context(context):
            raise ValueError("ConstantBundle.encoded_scalars requires scaling_factor in flexible scale mode")
        scaling_factor = context.scale_at(cur_limbs)
    scaling_factor = float(scaling_factor)
    if scaling_factor <= 0:
        raise ValueError(f"scalar scaling_factor must be positive, got {scaling_factor}")
    return scaling_factor


def _is_flexible_context(context):
    return str(getattr(context, "scale_mode", "")).lower() == "flexible"


def _resolve_ring_dim(context, ring_dim):
    if ring_dim is not None:
        return int(ring_dim)
    if context is not None:
        return int(context.N)
    raise ValueError("ConstantBundle requires ring_dim or context to prepare plaintext")


def _resolve_slots(vector, slots):
    if slots is not None:
        return int(slots)
    if isinstance(vector, PreparedPlaintext):
        return int(vector.slots)
    if isinstance(vector, PackedRaw):
        return _tensor_vector_slots(vector.tensor)
    raise TypeError("ConstantBundle vectors must be PackedRaw or PreparedPlaintext")


def _prepare_vector(vector, slots, ring_dim, device=None, context=None):
    if isinstance(vector, PreparedPlaintext):
        if int(vector.slots) != int(slots):
            raise ValueError(f"Prepared vector slots [{vector.slots}] do not match requested slots [{slots}]")
        return vector

    if isinstance(vector, PackedRaw):
        return _prepare_tensor_vector(vector.packed_tensor(slots, context), slots, ring_dim, device, context)

    raise TypeError("ConstantBundle vectors must be PackedRaw or PreparedPlaintext")


def _tensor_vector_slots(vector):
    _validate_tensor_rank(vector)
    return int(vector.shape[-1])


def _validate_tensor_rank(vector):
    if vector.dim() not in (1, 2):
        raise ValueError("raw constant tensors must be rank-1 or rank-2")


def _validate_tensor_slots(vector, slots, source_name):
    _validate_tensor_rank(vector)
    actual_slots = int(vector.shape[-1])
    if actual_slots != int(slots):
        raise ValueError(
            f"{source_name} tensor size [{actual_slots}] must match slots [{int(slots)}]; "
            "pack, pad, or truncate before constructing the bundle"
        )


def _prepare_tensor_vector(vector, slots, ring_dim, device=None, context=None):
    slots = int(slots)
    actual_slots = _tensor_vector_slots(vector)
    if actual_slots != slots:
        raise ValueError(
            f"ConstantBundle tensor vector size [{actual_slots}] must match slots [{slots}]; "
            "pad or truncate before constructing the bundle"
        )
    if vector.is_cuda:
        if context is None:
            raise ValueError("ConstantBundle CUDA tensor vectors require context")
        stage1_input = vector if _is_complex_tensor(vector) else vector.to(dtype=torch.complex128)
        return encode_stage1_packed(stage1_input, slots=slots, context=context)

    values = _tensor_values(vector)
    return encode_stage1(values, slots, ring_dim, device=device, context=context)


def _is_complex_tensor(vector):
    return vector.dtype in tuple(
        dtype
        for dtype in (
            getattr(torch, "complex32", None),
            torch.complex64,
            torch.complex128,
        )
        if dtype is not None
    )


def _tensor_values(vector):
    values = np.asarray(vector)
    if np.iscomplexobj(values):
        return np.asarray(values, dtype=np.complex128)
    return np.asarray(values, dtype=np.double)
