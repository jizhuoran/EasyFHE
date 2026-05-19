from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import easyfhe as torch

from .ciphertext import PreparedPlaintext
from .ops.encoding import encode_stage1, encode_stage2


_CACHE_MODES = {"none", "middle", "plain", "both"}
_MAX_ENCODED_BITS = 61
_MAX_SCALAR_LOG_STEP = 60


class ConstantBundle:
    """Named scalar/vector constants with shared plaintext caching."""

    def __init__(self, *, scalars=None, vectors=None, cache_mode="plain"):
        self.scalars = dict(scalars or {})
        self.vectors = dict(vectors or {})
        self.cache_mode = self._validate_cache_mode(cache_mode)
        self._middle_cache = {}
        self._plain_cache = {}
        self._scalar_cache = {}
        self._cache_stats = {
            "middle_hits": 0,
            "middle_misses": 0,
            "plain_hits": 0,
            "plain_misses": 0,
            "scalar_hits": 0,
            "scalar_misses": 0,
        }

    @classmethod
    def from_vectors(cls, vectors, *, scalars=None, cache_mode="plain"):
        return cls(scalars=scalars, vectors=vectors, cache_mode=cache_mode)

    def __len__(self):
        return len(self.vectors)

    def scalar(self, name):
        return self.scalars[name]

    def encoded_scalars(self, names, cur_limbs, noise_deg, cryptoContext, *, mode="double", absolute=False, cache=True):
        if isinstance(names, str):
            names = (names,)
        else:
            names = tuple(names)
        if not names:
            raise ValueError("ConstantBundle.encoded_scalars requires at least one scalar name")

        cur_limbs = int(cur_limbs)
        noise_deg = int(noise_deg)
        mode = self._normalize_scalar_mode(mode)
        if mode == "double" and noise_deg < 1:
            raise ValueError("double scalar encoding requires noise_deg >= 1")
        key = self._scalar_key(names, cur_limbs, noise_deg, cryptoContext, mode, absolute)
        if cache and self.cache_mode != "none" and key in self._scalar_cache:
            self._cache_stats["scalar_hits"] += 1
            return self._scalar_cache[key]

        self._cache_stats["scalar_misses"] += 1
        encoded = self._encode_scalar_values(
            [self.scalar(name) for name in names],
            cur_limbs,
            noise_deg,
            cryptoContext,
            mode,
            absolute,
        )
        if cache and self._cache_plain():
            self._scalar_cache[key] = encoded
        return encoded

    def _validate_cache_mode(self, cache_mode):
        if cache_mode not in _CACHE_MODES:
            raise ValueError(f"cache_mode must be one of {sorted(_CACHE_MODES)}, got {cache_mode!r}")
        return cache_mode

    def set_cache_mode(self, cache_mode, clear=True):
        self.cache_mode = self._validate_cache_mode(cache_mode)
        if clear:
            self.clear_cache()

    def clear_cache(self):
        self._middle_cache.clear()
        self._plain_cache.clear()
        self._scalar_cache.clear()

    def cache_info(self):
        return {
            "mode": self.cache_mode,
            "middle_entries": len(self._middle_cache),
            "plain_entries": len(self._plain_cache),
            "scalar_entries": len(self._scalar_cache),
            "middle_bytes": self._cache_nbytes(self._middle_cache),
            "plain_bytes": self._cache_nbytes(self._plain_cache),
            "scalar_bytes": self._cache_nbytes(self._scalar_cache),
            **self._cache_stats,
        }

    def _cache_nbytes(self, cache):
        return sum(self._object_nbytes(value) for value in cache.values())

    def _object_nbytes(self, value):
        if hasattr(value, "numel") or hasattr(value, "nbytes"):
            return self._array_nbytes(value)
        total = 0
        for attr in ("values", "encoded_values"):
            array = getattr(value, attr, None)
            if array is None:
                continue
            total += self._array_nbytes(array)
        for cv in getattr(value, "cv", ()):
            total += self._array_nbytes(cv)
        return total

    def _array_nbytes(self, array):
        if hasattr(array, "numel") and hasattr(array, "element_size"):
            return array.numel() * array.element_size()
        if hasattr(array, "nbytes"):
            return array.nbytes
        return 0

    def _cache_middle(self):
        return self.cache_mode in {"middle", "both"}

    def _cache_plain(self):
        return self.cache_mode in {"plain", "both"}

    def _context_key(self, cryptoContext):
        return (
            id(cryptoContext),
            getattr(cryptoContext, "device", None),
            getattr(cryptoContext, "N", None),
            getattr(cryptoContext, "L", None),
            getattr(cryptoContext, "rescaleTech", None),
        )

    def _scale_key(self, scale):
        if isinstance(scale, (tuple, list)):
            return tuple(float(value) for value in scale)
        return float(scale)

    def _middle_key(self, name, slots, ring_dim, scale):
        return (name, slots, self._scale_key(scale), ring_dim)

    def _plain_key(self, name, level, slots, cryptoContext, scale, is_ext):
        return (
            name,
            level,
            slots,
            self._scale_key(scale),
            is_ext,
            self._context_key(cryptoContext),
        )

    def _scalar_key(self, names, cur_limbs, noise_deg, cryptoContext, mode, absolute):
        return (
            names,
            cur_limbs,
            noise_deg,
            mode,
            bool(absolute),
            self._context_key(cryptoContext),
            cryptoContext.scale_at(cur_limbs) if mode == "double" else None,
        )

    def _normalize_scalar_mode(self, mode):
        mode = str(mode)
        if mode not in {"double", "int"}:
            raise ValueError(f"scalar mode must be 'double' or 'int', got {mode!r}")
        return mode

    def _encode_scalar_values(self, values, cur_limbs, noise_deg, cryptoContext, mode, absolute):
        rows = []
        for value in values:
            value = abs(value) if absolute else value
            if mode == "double":
                rows.append(self._encode_double_scalar_value(value, cur_limbs, noise_deg, cryptoContext))
            else:
                encoded = int(value)
                rows.append(
                    [
                        int(encoded) % int(cryptoContext.moduliQ_scalar[level])
                        for level in range(cur_limbs)
                    ]
                )
        return torch.from_numpy(np.asarray(rows, dtype=np.uint64)).to(cryptoContext.device)

    def _encode_double_scalar_value(self, value, cur_limbs, noise_deg, cryptoContext):
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
                crt_approx = self._crt_mult(crt_approx, crt_sf, cryptoContext.moduliQ_scalar)
                log_approx -= log_step

            crt_constant = self._crt_mult(crt_constant, crt_approx, cryptoContext.moduliQ_scalar)

        int_sc_factor = int(sc_factor + 0.5)
        crt_sc_factor = cur_limbs * [int_sc_factor]
        for _ in range(1, noise_deg):
            crt_constant = self._crt_mult(crt_constant, crt_sc_factor, cryptoContext.moduliQ_scalar)

        crt_constant = [
            int(item) % int(cryptoContext.moduliQ_scalar[level])
            for level, item in enumerate(crt_constant)
        ]
        return crt_constant

    def _crt_mult(self, xs, ys, mods):
        return [(int(x) * int(y)) % int(mod) for x, y, mod in zip(xs, ys, mods)]

    def _values(self, name, slots=None, scale=1.0):
        if name not in self.vectors:
            raise KeyError(f"constant vector {name!r} is missing")

        vector = self.vectors[name]
        if self._is_batched_vector(vector):
            rows = self._raw_value_rows(vector)
            target_slots = int(slots) if slots is not None else max(row.size for row in rows)
            scales = self._normalize_scales(scale, len(rows))
            return np.stack(
                [self._values_for_vector(row, target_slots, item_scale) for row, item_scale in zip(rows, scales)],
                axis=0,
            )

        values = self._raw_values(vector)

        if slots is not None:
            slots = int(slots)
            if values.size < slots:
                values = np.pad(values, (0, slots - values.size))
            elif values.size > slots:
                values = values[:slots]
        if scale != 1.0:
            values = values * scale
        return values

    def _slots(self, name, default=None):
        if name not in self.vectors:
            if default is None:
                raise KeyError(f"constant vector {name!r} is missing")
            return int(default)
        return self._resolve_slots(self.vectors[name], None)

    def _encoded_middle(self, name, slots=None, cryptoContext=None, scale=1.0, ring_dim=None):
        ring_dim = self._resolve_ring_dim(cryptoContext, ring_dim)
        vector = self._vector(name)
        slots = self._resolve_slots(vector, slots)

        key = self._middle_key(name, slots, ring_dim, scale)
        if self.cache_mode != "none" and key in self._middle_cache:
            self._cache_stats["middle_hits"] += 1
            return self._middle_cache[key]

        self._cache_stats["middle_misses"] += 1
        middle = self._prepare_vector(vector, slots, ring_dim, scale)
        if self._cache_middle():
            self._middle_cache[key] = middle
        return middle

    def plaintext(self, name, level, slots, cryptoContext, scale=1.0, is_ext=False, cache=True):
        if not isinstance(name, str):
            raise TypeError(f"ConstantBundle.plaintext name must be str, got {type(name)}")
        return self._plaintext_single(name, level, slots, cryptoContext, scale, is_ext, cache)

    def _plaintext_single(self, name, level, slots, cryptoContext, scale, is_ext, cache):
        plain_key = self._plain_key(name, level, slots, cryptoContext, scale, is_ext)
        if cache and self.cache_mode != "none" and plain_key in self._plain_cache:
            self._cache_stats["plain_hits"] += 1
            return self._plain_cache[plain_key]

        self._cache_stats["plain_misses"] += 1
        middle = self._encoded_middle(name, slots, cryptoContext, scale)
        plaintext = encode_stage2(middle, level, slots, is_ext, cryptoContext)
        if cache and self._cache_plain():
            self._plain_cache[plain_key] = plaintext
        return plaintext

    def _resolve_ring_dim(self, cryptoContext, ring_dim):
        if ring_dim is not None:
            return int(ring_dim)
        if cryptoContext is not None:
            return int(cryptoContext.N)
        raise ValueError("ConstantBundle requires ring_dim or cryptoContext to prepare plaintext")

    def _resolve_slots(self, vector, slots):
        if slots is not None:
            return int(slots)
        if isinstance(vector, PreparedPlaintext):
            return int(vector.slots)
        if self._is_batched_vector(vector):
            rows = self._raw_value_rows(vector)
            return max(int(row.size) for row in rows)
        return int(self._raw_values(vector).size)

    def _normalize_scales(self, scale, count):
        if isinstance(scale, (tuple, list)):
            if len(scale) != count:
                raise ValueError(f"scale length [{len(scale)}] does not match batch size [{count}]")
            return tuple(float(value) for value in scale)
        return tuple(float(scale) for _ in range(count))

    def _prepare_vector(self, vector, slots, ring_dim, scale):
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
        if self._is_batched_vector(vector):
            rows = self._raw_value_rows(vector)
            scales = self._normalize_scales(scale, len(rows))
            values = np.stack(
                [self._values_for_vector(row, slots, item_scale) for row, item_scale in zip(rows, scales)],
                axis=0,
            )
        else:
            values = self._values_for_vector(vector, slots, scale)
        return encode_stage1(values, slots, ring_dim)

    def _values_for_vector(self, vector, slots, scale=1.0):
        values = self._raw_values(vector)
        slots = int(slots)
        if values.size < slots:
            values = np.pad(values, (0, slots - values.size))
        elif values.size > slots:
            values = values[:slots]
        if scale != 1.0:
            values = values * scale
        return values

    def _vector(self, name):
        return self.vectors[name]

    def _is_batched_vector(self, vector):
        if isinstance(vector, (PreparedPlaintext, Mapping)):
            return False
        if isinstance(vector, (list, tuple)):
            if not vector:
                return False
            try:
                return np.asarray(vector[0]).ndim > 0
            except ValueError:
                return True
        try:
            return np.asarray(vector).ndim >= 2
        except ValueError:
            return True

    def _raw_value_rows(self, vector):
        if isinstance(vector, (list, tuple)):
            return [self._raw_values(row) for row in vector]
        values = np.asarray(vector)
        if values.ndim < 2:
            raise ValueError("batched constant vector must have at least two dimensions")
        values = values.reshape(values.shape[0], -1)
        if np.iscomplexobj(values):
            return [np.asarray(row, dtype=np.complex128).reshape(-1) for row in values]
        return [np.asarray(row, dtype=np.double).reshape(-1) for row in values]

    def _raw_values(self, vector):
        if isinstance(vector, PreparedPlaintext):
            vector = vector.values
        values = np.asarray(vector).reshape(-1)
        if np.iscomplexobj(values):
            return np.asarray(values, dtype=np.complex128).reshape(-1)
        return np.asarray(values, dtype=np.double).reshape(-1)
