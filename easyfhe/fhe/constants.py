from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import easyfhe as torch

from .ops.encoding import PreparedPlaintext, encode_stage1, encode_stage2


_CACHE_MODES = {"none", "middle", "plain", "both", "mix"}
_PLAIN_CACHE_POLICIES = {"first_fit", "small_first"}
_MAX_ENCODED_BITS = 61
_MAX_SCALAR_LOG_STEP = 60


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
        self._vectors = dict(vectors or {})
        self.cache_mode = _validate_cache_mode(cache_mode)
        self._plain_cache_limit_bytes = _cache_limit_gb_to_bytes(plain_cache_limit_gb)
        _validate_plain_cache_limit_mode(self.cache_mode, self._plain_cache_limit_bytes)
        self.plain_cache_policy = _validate_plain_cache_policy(plain_cache_policy)
        self._middle_cache = {}
        self._plain_cache = {}
        self._plain_cache_bytes_by_key = {}
        self._plain_middle_key_by_plain_key = {}
        self._plain_middle_key_counts = {}
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

    def encoded_scalars(self, names, cur_limbs, noise_deg, cryptoContext, *, mode="double", absolute=False, cache=True):
        if isinstance(names, str):
            names = (names,)
        else:
            names = tuple(names)
        if not names:
            raise ValueError("ConstantBundle.encoded_scalars requires at least one scalar name")

        cur_limbs = int(cur_limbs)
        noise_deg = int(noise_deg)
        mode = _normalize_scalar_mode(mode)
        if mode == "double" and noise_deg < 1:
            raise ValueError("double scalar encoding requires noise_deg >= 1")
        key = _scalar_key(names, cur_limbs, noise_deg, cryptoContext, mode, absolute)
        if cache and self.cache_mode != "none" and key in self._scalar_cache:
            self._cache_stats["scalar_hits"] += 1
            return self._scalar_cache[key]

        self._cache_stats["scalar_misses"] += 1
        encoded = self._encode_scalar_values(
            [self._scalar_value(name) for name in names],
            cur_limbs,
            noise_deg,
            cryptoContext,
            mode,
            absolute,
        )
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

    def _encode_scalar_values(self, values, cur_limbs, noise_deg, cryptoContext, mode, absolute):
        rows = []
        for value in values:
            value = abs(value) if absolute else value
            if mode == "double":
                rows.append(_encode_double_scalar_value(value, cur_limbs, noise_deg, cryptoContext))
            else:
                encoded = int(value)
                rows.append(
                    [
                        int(encoded) % int(cryptoContext.moduliQ_scalar[level])
                        for level in range(cur_limbs)
                    ]
                )
        return torch.from_numpy(np.asarray(rows, dtype=np.uint64)).to(cryptoContext.device)

    def _encoded_middle(self, name, slots=None, cryptoContext=None, scale=1.0, ring_dim=None, cache_on_miss=True):
        ring_dim = _resolve_ring_dim(cryptoContext, ring_dim)
        if name not in self._vectors:
            raise KeyError(f"constant vector {name!r} is missing")
        vector = self._vectors[name]
        slots = _resolve_slots(vector, slots)

        key = _middle_key(name, slots, ring_dim, scale, _device_key(cryptoContext))
        if self.cache_mode != "none" and key in self._middle_cache:
            self._cache_stats["middle_hits"] += 1
            return key, self._middle_cache[key]

        self._cache_stats["middle_misses"] += 1
        middle = _prepare_vector(vector, slots, ring_dim, scale, _device_key(cryptoContext))
        if cache_on_miss and _cache_middle(self.cache_mode):
            self._middle_cache[key] = middle
        return key, middle

    def plaintext(self, name, level, slots, cryptoContext, scale=1.0, is_ext=False, cache=True):
        if not isinstance(name, str):
            raise TypeError(f"ConstantBundle.plaintext name must be str, got {type(name)}")
        return self._plaintext_single(name, level, slots, cryptoContext, scale, is_ext, cache)

    def _plaintext_single(self, name, level, slots, cryptoContext, scale, is_ext, cache):
        plain_key = _plain_key(name, level, slots, cryptoContext, scale, is_ext)
        if cache and self.cache_mode != "none" and plain_key in self._plain_cache:
            self._cache_stats["plain_hits"] += 1
            return self._plain_cache[plain_key]

        self._cache_stats["plain_misses"] += 1
        middle_key, middle = self._encoded_middle(
            name,
            slots,
            cryptoContext,
            scale,
            cache_on_miss=self.cache_mode in {"middle", "both", "mix"},
        )
        had_sibling_plain = self._has_plain_for_middle(middle_key)
        plaintext = encode_stage2(middle, level, slots, is_ext, cryptoContext)
        cached_plain = self._maybe_cache_plain(plain_key, middle_key, plaintext, cache)
        if cache and self.cache_mode == "both":
            self._middle_cache.setdefault(middle_key, middle)
        elif cache and self.cache_mode == "mix":
            if cached_plain and not had_sibling_plain:
                self._middle_cache.pop(middle_key, None)
            elif cached_plain:
                self._middle_cache.setdefault(middle_key, middle)
            elif not self._has_plain_for_middle(middle_key):
                self._middle_cache.setdefault(middle_key, middle)
        return plaintext

    def _maybe_cache_plain(self, plain_key, middle_key, plaintext, cache):
        if not (cache and _cache_plain(self.cache_mode)):
            return False
        plain_bytes = _object_nbytes(plaintext)
        if self.cache_mode == "mix" and self._plain_cache_limit_bytes is not None:
            if plain_bytes > self._plain_cache_limit_bytes:
                self._cache_stats["plain_cache_skips"] += 1
                return False
            if self._plain_cache_bytes + plain_bytes > self._plain_cache_limit_bytes:
                if self.plain_cache_policy != "small_first" or not self._evict_for_smaller_plain(plain_bytes):
                    self._cache_stats["plain_cache_skips"] += 1
                    return False
        self._store_plain(plain_key, middle_key, plaintext, plain_bytes)
        return True

    def _evict_for_smaller_plain(self, plain_bytes):
        if not self._plain_cache_bytes_by_key:
            return False
        evict_key, evict_bytes = max(self._plain_cache_bytes_by_key.items(), key=lambda item: item[1])
        if plain_bytes >= evict_bytes:
            return False
        self._evict_plain(evict_key)
        return True

    def _store_plain(self, plain_key, middle_key, plaintext, plain_bytes):
        self._plain_cache[plain_key] = plaintext
        self._plain_cache_bytes_by_key[plain_key] = plain_bytes
        self._plain_middle_key_by_plain_key[plain_key] = middle_key
        self._plain_middle_key_counts[middle_key] = self._plain_middle_key_counts.get(middle_key, 0) + 1
        self._plain_cache_bytes += plain_bytes

    def _evict_plain(self, plain_key):
        plain_bytes = self._plain_cache_bytes_by_key.pop(plain_key)
        middle_key = self._plain_middle_key_by_plain_key.pop(plain_key)
        count = self._plain_middle_key_counts[middle_key] - 1
        if count:
            self._plain_middle_key_counts[middle_key] = count
        else:
            del self._plain_middle_key_counts[middle_key]
        del self._plain_cache[plain_key]
        self._plain_cache_bytes -= plain_bytes
        self._cache_stats["plain_cache_evictions"] += 1
        if _cache_middle(self.cache_mode) and not self._has_plain_for_middle(middle_key):
            self._restore_middle(middle_key)

    def _has_plain_for_middle(self, middle_key):
        return self._plain_middle_key_counts.get(middle_key, 0) > 0

    def _restore_middle(self, middle_key):
        if middle_key in self._middle_cache:
            return
        name, slots, scale, ring_dim, device = middle_key
        vector = self._vectors[name]
        self._middle_cache[middle_key] = _prepare_vector(vector, slots, ring_dim, scale, device)


def _validate_cache_mode(cache_mode):
    if cache_mode not in _CACHE_MODES:
        raise ValueError(f"cache_mode must be one of {sorted(_CACHE_MODES)}, got {cache_mode!r}")
    return cache_mode


def _validate_plain_cache_limit_mode(cache_mode, limit_bytes):
    if limit_bytes is not None and cache_mode != "mix":
        raise ValueError("plain_cache_limit is only supported with cache_mode='mix'")


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
    return cache_mode in {"middle", "both", "mix"}


def _cache_plain(cache_mode):
    return cache_mode in {"plain", "both", "mix"}


def _cache_nbytes(cache):
    return sum(_object_nbytes(value) for value in cache.values())


def _object_nbytes(value):
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


def _context_key(cryptoContext):
    return (
        id(cryptoContext),
        getattr(cryptoContext, "device", None),
        getattr(cryptoContext, "N", None),
        getattr(cryptoContext, "L", None),
        getattr(cryptoContext, "scale_mode", None),
        getattr(cryptoContext, "rescale_policy", None),
    )


def _scale_key(scale):
    if isinstance(scale, (tuple, list)):
        return tuple(float(value) for value in scale)
    return float(scale)


def _device_key(cryptoContext):
    if cryptoContext is None:
        return None
    device = getattr(cryptoContext, "device", None)
    return None if device is None else str(device)


def _middle_key(name, slots, ring_dim, scale, device):
    return (name, slots, _scale_key(scale), ring_dim, device)


def _plain_key(name, level, slots, cryptoContext, scale, is_ext):
    return (
        name,
        level,
        slots,
        _scale_key(scale),
        is_ext,
        _context_key(cryptoContext),
    )


def _scalar_key(names, cur_limbs, noise_deg, cryptoContext, mode, absolute):
    return (
        names,
        cur_limbs,
        noise_deg,
        mode,
        bool(absolute),
        _context_key(cryptoContext),
        cryptoContext.scale_at(cur_limbs) if mode == "double" else None,
    )


def _normalize_scalar_mode(mode):
    mode = str(mode)
    if mode not in {"double", "int"}:
        raise ValueError(f"scalar mode must be 'double' or 'int', got {mode!r}")
    return mode


def _encode_double_scalar_value(value, cur_limbs, noise_deg, cryptoContext):
    sc_factor = cryptoContext.scale_at(cur_limbs)

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
            crt_approx = _crt_mult(crt_approx, crt_sf, cryptoContext.moduliQ_scalar)
            log_approx -= log_step

        crt_constant = _crt_mult(crt_constant, crt_approx, cryptoContext.moduliQ_scalar)

    int_sc_factor = int(sc_factor + 0.5)
    crt_sc_factor = cur_limbs * [int_sc_factor]
    for _ in range(1, noise_deg):
        crt_constant = _crt_mult(crt_constant, crt_sc_factor, cryptoContext.moduliQ_scalar)

    return [
        int(item) % int(cryptoContext.moduliQ_scalar[level])
        for level, item in enumerate(crt_constant)
    ]


def _crt_mult(xs, ys, mods):
    return [(int(x) * int(y)) % int(mod) for x, y, mod in zip(xs, ys, mods)]


def _resolve_ring_dim(cryptoContext, ring_dim):
    if ring_dim is not None:
        return int(ring_dim)
    if cryptoContext is not None:
        return int(cryptoContext.N)
    raise ValueError("ConstantBundle requires ring_dim or cryptoContext to prepare plaintext")


def _resolve_slots(vector, slots):
    if slots is not None:
        return int(slots)
    if isinstance(vector, PreparedPlaintext):
        return int(vector.slots)
    rows = _vector_rows(vector)
    if rows is not None:
        return max(int(row.size) for row in rows)
    return int(_raw_values(vector).size)


def _prepare_vector(vector, slots, ring_dim, scale, device=None):
    if isinstance(vector, PreparedPlaintext):
        if int(vector.slots) != int(slots):
            raise ValueError(f"Prepared vector slots [{vector.slots}] do not match requested slots [{slots}]")
        if scale == 1.0:
            return vector
        return PreparedPlaintext(
            vector.values * scale,
            vector.slots,
            vector.encoded_values * scale,
            vector.max_encoded_value * abs(scale),
        )

    if isinstance(vector, Mapping):
        raise TypeError("ConstantBundle vector values must be arrays or PreparedPlaintext, not mappings")
    rows = _vector_rows(vector)
    if rows is not None:
        scales = _normalize_scales(scale, len(rows))
        values = np.stack(
            [_pad_and_scale_values(row, slots, item_scale) for row, item_scale in zip(rows, scales)],
            axis=0,
        )
    else:
        values = _pad_and_scale_values(_raw_values(vector), slots, scale)
    return encode_stage1(values, slots, ring_dim, device=device)


def _normalize_scales(scale, count):
    if isinstance(scale, (tuple, list)):
        if len(scale) != count:
            raise ValueError(f"scale length [{len(scale)}] does not match batch size [{count}]")
        return tuple(float(value) for value in scale)
    return tuple(float(scale) for _ in range(count))


def _vector_rows(vector):
    if isinstance(vector, (PreparedPlaintext, Mapping)):
        return None
    try:
        values = np.asarray(vector)
    except ValueError as exc:
        raise TypeError("ConstantBundle vector batches must be rectangular numeric arrays") from exc
    if values.dtype == object:
        raise TypeError("ConstantBundle vector batches must be rectangular numeric arrays")
    if values.ndim < 2:
        return None
    values = values.reshape(values.shape[0], -1)
    if np.iscomplexobj(values):
        return [np.asarray(row, dtype=np.complex128).reshape(-1) for row in values]
    return [np.asarray(row, dtype=np.double).reshape(-1) for row in values]


def _raw_values(vector):
    if isinstance(vector, PreparedPlaintext):
        vector = vector.values
    try:
        values = np.asarray(vector)
    except ValueError as exc:
        raise TypeError("ConstantBundle vector values must be numeric arrays") from exc
    if values.dtype == object:
        raise TypeError("ConstantBundle vector values must be numeric arrays")
    values = values.reshape(-1)
    if np.iscomplexobj(values):
        return np.asarray(values, dtype=np.complex128).reshape(-1)
    return np.asarray(values, dtype=np.double).reshape(-1)


def _pad_and_scale_values(values, slots, scale=1.0):
    values = _raw_values(values)
    slots = int(slots)
    if values.size < slots:
        values = np.pad(values, (0, slots - values.size))
    elif values.size > slots:
        values = values[:slots]
    if scale != 1.0:
        values = values * scale
    return values
