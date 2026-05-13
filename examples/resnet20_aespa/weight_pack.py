from pathlib import Path

import numpy as np

import easyfhe.fhe as fhe


_CACHE_MODES = {"none", "middle", "plain", "both"}


class WeightPack:
    def __init__(self, arrays, cache_mode="plain"):
        self.arrays = arrays
        self.cache_mode = self._validate_cache_mode(cache_mode)
        self._middle_cache = {}
        self._plain_cache = {}
        self._cache_stats = {
            "middle_hits": 0,
            "middle_misses": 0,
            "plain_hits": 0,
            "plain_misses": 0,
        }

    @classmethod
    def from_npz(cls, path, cache_mode="plain"):
        weight_path = Path(path)
        if not weight_path.exists():
            raise ValueError(f"Weight npz {weight_path} does not exist!")
        with np.load(weight_path) as weights:
            arrays = {name: np.asarray(weights[name], dtype=np.double) for name in weights.files}
        return cls(arrays, cache_mode=cache_mode)

    def __len__(self):
        return len(self.arrays)

    def has(self, name):
        return name in self.arrays

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

    def cache_info(self):
        return {
            "mode": self.cache_mode,
            "middle_entries": len(self._middle_cache),
            "plain_entries": len(self._plain_cache),
            "middle_bytes": self._cache_nbytes(self._middle_cache),
            "plain_bytes": self._cache_nbytes(self._plain_cache),
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
            if hasattr(array, "numel") and hasattr(array, "element_size"):
                total += array.numel() * array.element_size()
            elif hasattr(array, "nbytes"):
                total += array.nbytes
        for cv in getattr(value, "cv", ()):
            if hasattr(cv, "numel") and hasattr(cv, "element_size"):
                total += cv.numel() * cv.element_size()
            elif hasattr(cv, "nbytes"):
                total += cv.nbytes
        return total

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
        return float(scale)

    def _middle_key(self, name, slots, cryptoContext, scale):
        return (name, slots, self._scale_key(scale), cryptoContext.N)

    def _plain_key(self, name, level, slots, cryptoContext, scale, is_ext):
        return (
            name,
            level,
            slots,
            self._scale_key(scale),
            is_ext,
            self._context_key(cryptoContext),
        )

    def values(self, name, slots, scale=1.0):
        if name not in self.arrays:
            raise KeyError(f"weight {name!r} is missing")

        values = np.asarray(self.arrays[name], dtype=np.double).reshape(-1)
        if values.size < slots:
            values = np.pad(values, (0, slots - values.size))
        elif values.size > slots:
            values = values[:slots]
        if scale != 1.0:
            values = values * scale
        return values

    def prepared_plaintext(self, name, slots, cryptoContext, scale=1.0):
        key = self._middle_key(name, slots, cryptoContext, scale)
        if self.cache_mode != "none" and key in self._middle_cache:
            self._cache_stats["middle_hits"] += 1
            return self._middle_cache[key]

        self._cache_stats["middle_misses"] += 1
        middle = fhe.prepare_plaintext(self.values(name, slots, scale), slots, cryptoContext.N)
        if self._cache_middle():
            self._middle_cache[key] = middle
        return middle

    def plaintext(self, name, level, slots, cryptoContext, scale=1.0, is_ext=False):
        plain_key = self._plain_key(name, level, slots, cryptoContext, scale, is_ext)
        if self.cache_mode != "none" and plain_key in self._plain_cache:
            self._cache_stats["plain_hits"] += 1
            return self._plain_cache[plain_key]

        self._cache_stats["plain_misses"] += 1
        middle = self.prepared_plaintext(name, slots, cryptoContext, scale)
        plaintext = fhe.make_plaintext(middle, level, slots, is_ext, cryptoContext)
        if self._cache_plain():
            self._plain_cache[plain_key] = plaintext
        return plaintext

    def plaintext_for_cipher(self, name, cipher, cryptoContext, scale=1.0, is_ext=False):
        return self.plaintext(
            name,
            cryptoContext.L - cipher.cur_limbs,
            cipher.slots,
            cryptoContext,
            scale,
            is_ext,
        )

    def encode(self, name, level, slots, cryptoContext, scale=1.0):
        return self.plaintext(name, level, slots, cryptoContext, scale)

    def encode_for_cipher(self, name, cipher, cryptoContext, scale=1.0):
        return self.plaintext_for_cipher(name, cipher, cryptoContext, scale)
