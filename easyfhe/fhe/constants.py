from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .ciphertext import Cipher, PreparedPlaintext
from .ops import encode as encode_plaintext
from .ops import prepare_plaintext


_CACHE_MODES = {"none", "middle", "plain", "both"}


class ConstantBundle:
    """Named scalar/vector constants with shared plaintext caching."""

    def __init__(self, *, info=None, scalars=None, vectors=None, cache_mode="plain"):
        self.info = dict(info or {})
        self.scalars = dict(scalars or {})
        self.vectors = dict(vectors or {})
        self.cache_mode = self._validate_cache_mode(cache_mode)
        self._middle_cache = {}
        self._plain_cache = {}
        self._plain_batch_cache = {}
        self._cache_stats = {
            "middle_hits": 0,
            "middle_misses": 0,
            "plain_hits": 0,
            "plain_misses": 0,
            "plain_batch_hits": 0,
            "plain_batch_misses": 0,
        }

    @classmethod
    def from_vectors(cls, vectors, *, info=None, scalars=None, cache_mode="plain"):
        return cls(info=info, scalars=scalars, vectors=vectors, cache_mode=cache_mode)

    def __len__(self):
        return len(self.vectors)

    def has(self, name):
        return name in self.vectors

    def scalar(self, name):
        return self.scalars[name]

    def vector(self, name):
        return self.vectors[name]

    def get_info(self, name, default=None):
        return self.info.get(name, default)

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
        self._plain_batch_cache.clear()

    def cache_info(self):
        return {
            "mode": self.cache_mode,
            "middle_entries": len(self._middle_cache),
            "plain_entries": len(self._plain_cache),
            "plain_batch_entries": len(self._plain_batch_cache),
            "middle_bytes": self._cache_nbytes(self._middle_cache),
            "plain_bytes": self._cache_nbytes(self._plain_cache),
            "plain_batch_bytes": self._cache_nbytes(self._plain_batch_cache),
            **self._cache_stats,
        }

    def _cache_nbytes(self, cache):
        return sum(self._object_nbytes(value) for value in cache.values())

    def _object_nbytes(self, value):
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

    def _plain_batch_key(self, names, level, slots, cryptoContext, scales, is_ext):
        return (
            tuple(names),
            level,
            slots,
            tuple(float(scale) for scale in scales),
            is_ext,
            self._context_key(cryptoContext),
        )

    def values(self, name, slots=None, scale=1.0):
        if name not in self.vectors:
            raise KeyError(f"constant vector {name!r} is missing")

        values = self._raw_values(self.vectors[name])

        if slots is not None:
            slots = int(slots)
            if values.size < slots:
                values = np.pad(values, (0, slots - values.size))
            elif values.size > slots:
                values = values[:slots]
        if scale != 1.0:
            values = values * scale
        return values

    def slots(self, name, default=None):
        if name not in self.vectors:
            if default is None:
                raise KeyError(f"constant vector {name!r} is missing")
            return int(default)
        return self._resolve_slots(self.vectors[name], None)

    def prepared_plaintext(self, name, slots=None, cryptoContext=None, scale=1.0, ring_dim=None):
        ring_dim = self._resolve_ring_dim(cryptoContext, ring_dim)
        vector = self.vector(name)
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

    def plaintext(self, name, level, slots, cryptoContext, scale=1.0, is_ext=False):
        plain_key = self._plain_key(name, level, slots, cryptoContext, scale, is_ext)
        if self.cache_mode != "none" and plain_key in self._plain_cache:
            self._cache_stats["plain_hits"] += 1
            return self._plain_cache[plain_key]

        self._cache_stats["plain_misses"] += 1
        plaintext = self._materialize_plaintext(name, level, slots, cryptoContext, scale, is_ext)
        if self._cache_plain():
            self._plain_cache[plain_key] = plaintext
        return plaintext

    def plaintext_batch(self, names, level, slots, cryptoContext, scale=1.0, is_ext=False):
        names = tuple(names)
        if not names:
            raise ValueError("ConstantBundle.plaintext_batch requires at least one name")
        slots = int(slots)
        level = int(level)
        is_ext = bool(is_ext)
        scales = self._normalize_scales(scale, len(names))
        batch_key = self._plain_batch_key(names, level, slots, cryptoContext, scales, is_ext)
        if self.cache_mode != "none" and batch_key in self._plain_batch_cache:
            self._cache_stats["plain_batch_hits"] += 1
            return self._plain_batch_cache[batch_key]

        self._cache_stats["plain_batch_misses"] += 1
        plaintexts = [
            self._plaintext_for_batch_item(name, level, slots, cryptoContext, item_scale, is_ext)
            for name, item_scale in zip(names, scales)
        ]
        batch = self._pack_plaintexts(plaintexts)
        if self._cache_plain():
            self._plain_batch_cache[batch_key] = batch
        return batch

    def plaintext_for_cipher(self, name, cipher, cryptoContext, scale=1.0, is_ext=False):
        return self.plaintext(
            name,
            cryptoContext.L - cipher.cur_limbs,
            cipher.slots,
            cryptoContext,
            scale,
            is_ext,
        )

    def encode(self, name, level, slots, cryptoContext, scale=1.0, is_ext=False):
        return self.plaintext(name, level, slots, cryptoContext, scale, is_ext)

    def encode_for_cipher(self, name, cipher, cryptoContext, scale=1.0, is_ext=False):
        return self.plaintext_for_cipher(name, cipher, cryptoContext, scale, is_ext)

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
        return int(self._raw_values(vector).size)

    def _materialize_plaintext(self, name, level, slots, cryptoContext, scale, is_ext):
        vector = self.vector(name)
        if isinstance(vector, Cipher):
            if scale != 1.0:
                raise ValueError("ConstantBundle cannot apply scale to an already encoded plaintext")
            _, plaintext = encode_plaintext(
                vector,
                cryptoContext,
                level=level,
                slots=slots,
                is_ext=is_ext,
            )
        else:
            middle = self.prepared_plaintext(name, slots, cryptoContext, scale)
            _, plaintext = encode_plaintext(
                middle,
                cryptoContext,
                level=level,
                slots=slots,
                is_ext=is_ext,
            )
        return plaintext

    def _plaintext_for_batch_item(self, name, level, slots, cryptoContext, scale, is_ext):
        plain_key = self._plain_key(name, level, slots, cryptoContext, scale, is_ext)
        if self.cache_mode != "none" and plain_key in self._plain_cache:
            self._cache_stats["plain_hits"] += 1
            return self._plain_cache[plain_key]
        return self._materialize_plaintext(name, level, slots, cryptoContext, scale, is_ext)

    def _normalize_scales(self, scale, count):
        if isinstance(scale, (tuple, list)):
            if len(scale) != count:
                raise ValueError(f"scale length [{len(scale)}] does not match batch size [{count}]")
            return tuple(float(value) for value in scale)
        return tuple(float(scale) for _ in range(count))

    def _pack_plaintexts(self, plaintexts):
        import easyfhe as torch

        plaintexts = tuple(plaintexts)
        first = plaintexts[0]
        for idx, plaintext in enumerate(plaintexts):
            if len(plaintext.cv) != len(first.cv):
                raise ValueError(f"plaintext batch component count mismatch at index {idx}")
            for field in ("cur_limbs", "scaling_factor", "noise_deg", "slots", "is_ext"):
                if getattr(plaintext, field) != getattr(first, field):
                    raise ValueError(
                        f"plaintext batch {field} mismatch at index {idx}: "
                        f"{getattr(plaintext, field)} != {getattr(first, field)}"
                    )
        cv = [
            torch.stack([plaintext.cv[component] for plaintext in plaintexts], dim=0)
            for component in range(len(first.cv))
        ]
        return first.cipher_like(cv, batch_size=len(plaintexts), cipher_id="assign")

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
        return prepare_plaintext(self.values_for_vector(vector, slots, scale), slots, ring_dim)

    def values_for_vector(self, vector, slots, scale=1.0):
        values = self._raw_values(vector)
        slots = int(slots)
        if values.size < slots:
            values = np.pad(values, (0, slots - values.size))
        elif values.size > slots:
            values = values[:slots]
        if scale != 1.0:
            values = values * scale
        return values

    def _raw_values(self, vector):
        if isinstance(vector, PreparedPlaintext):
            vector = vector.values
        values = np.asarray(vector).reshape(-1)
        if np.iscomplexobj(values):
            return np.asarray(values, dtype=np.complex128).reshape(-1)
        return np.asarray(values, dtype=np.double).reshape(-1)
