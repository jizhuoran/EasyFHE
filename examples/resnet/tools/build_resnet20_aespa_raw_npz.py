#!/usr/bin/env python3
"""Build a single name -> ndarray raw plaintext artifact for ResNet20 AESPA.

The existing weights_aespa_20 directory is already mostly a filesystem-backed
map of logical plaintext names to packed vectors. This script formalizes that
map as an NPZ and adds generated plaintexts that do not exist as .bin files:
fc_4096, bias_4096, masks, and slot-conversion masks used by resnet20_aespa.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RESNET_DIR = SCRIPT_DIR.parent
DEFAULT_WEIGHT_DIR = RESNET_DIR / "weights_aespa_20"
DEFAULT_OUTPUT = RESNET_DIR / "resnet20_aespa_weights.npz"


def read_bin_vector(path: Path) -> np.ndarray:
    text = path.read_text().strip()
    if not text:
        return np.asarray([], dtype=np.float64)
    values = []
    for row in text.splitlines():
        values.extend(float(value) for value in row.strip().split(",") if value)
    return np.asarray(values, dtype=np.float64)


def pad_to_slots(values: np.ndarray, slots: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size > slots:
        raise ValueError(f"cannot pad vector of length {values.size} to shorter slots={slots}")
    if values.size == slots:
        return values
    return np.pad(values, (0, slots - values.size), mode="constant")


def mask_mod(n: int, custom_value: float, slots: int) -> tuple[str, np.ndarray]:
    values = np.zeros(slots, dtype=np.float64)
    values[::n] = custom_value
    return f"mask_mod_{n}_{custom_value}_{slots}", values


def mask_first_n(n: int, slots: int) -> tuple[str, np.ndarray]:
    values = np.zeros(slots, dtype=np.float64)
    values[:n] = 1.0
    return f"mask_first_n_{n}_{slots}", values


def mask_scecond_n(n: int, slots: int) -> tuple[str, np.ndarray]:
    values = np.zeros(slots, dtype=np.float64)
    values[n:] = 1.0
    return f"mask_scecond_n_{n}_{slots}", values


def mask_from_to(from_: int, to: int, slots: int) -> tuple[str, np.ndarray]:
    values = np.zeros(slots, dtype=np.float64)
    values[from_:to] = 1.0
    return f"mask_from_to_{from_}_{to}_{slots}", values


def gen_mask(n: int, slots: int) -> tuple[str, np.ndarray]:
    values = np.zeros(slots, dtype=np.float64)
    copy_interval = n
    for idx in range(slots):
        if copy_interval > 0:
            values[idx] = 1.0
        copy_interval -= 1
        if copy_interval <= -n:
            copy_interval = n
    return f"gen_mask_{n}_{slots}", values


def mask_first_n_mod(n: int, padding: int, pos: int, num_channel: int, slots: int, prefix: str) -> tuple[str, np.ndarray]:
    values = []
    for _ in range(num_channel):
        values.extend([0.0] * (pos * n))
        values.extend([1.0] * n)
        values.extend([0.0] * (padding - n - (pos * n)))
    return f"{prefix}_{n}_{padding}_{pos}_{slots}", pad_to_slots(np.asarray(values, dtype=np.float64), slots)


def mask_channel(n: int, num_channel: int, spatial_size: int, num_cipher: int) -> tuple[str, np.ndarray]:
    channel_per_cipher = num_channel // num_cipher
    values = []
    for _ in range(n):
        values.extend([0.0] * spatial_size)
    values.extend([1.0] * (spatial_size // 4))
    values.extend([0.0] * (spatial_size - spatial_size // 4))
    for _ in range(2 * channel_per_cipher - 1 - n):
        values.extend([0.0] * spatial_size)
    return (
        f"mask_channel_{n}_{channel_per_cipher}_{spatial_size}",
        np.asarray(values, dtype=np.float64),
    )


def slot_conversion_mask(from_slots: int, to_slots: int) -> tuple[str, np.ndarray]:
    values = np.asarray([1.0] * to_slots + [0.0] * (from_slots - to_slots), dtype=np.float64)
    return f"slot_conversion_mask_{from_slots}to{to_slots}", values


def packed_fc(source: np.ndarray, num_channel: int, spatial_size: int, slots: int) -> tuple[str, np.ndarray]:
    values = np.asarray(source, dtype=np.float64).reshape(-1)
    packed = []
    for channel in range(num_channel):
        for cls in range(10):
            packed.append(values[(10 * channel) + cls])
        packed.extend([0.0] * (spatial_size - 10))
    return f"fc_{slots}", pad_to_slots(np.asarray(packed, dtype=np.float64), slots)


def packed_bias(source: np.ndarray, num_channel: int, spatial_size: int, slots: int) -> tuple[str, np.ndarray]:
    values = np.asarray(source, dtype=np.float64).reshape(-1)
    packed = []
    for _ in range(num_channel):
        packed.extend(values[:10])
        packed.extend([0.0] * (spatial_size - 10))
    return f"bias_{slots}", pad_to_slots(np.asarray(packed, dtype=np.float64), slots)


def add_generated_resnet20_aespa_plaintexts(raw: dict[str, np.ndarray]) -> None:
    generated = []

    generated.append(packed_fc(raw["fc"], num_channel=64, spatial_size=64, slots=4096))
    generated.append(packed_bias(raw["bias"], num_channel=64, spatial_size=16, slots=4096))

    generated.append(mask_from_to(0, 1024, 16384))

    generated.append(mask_first_n(16384, 32768))
    generated.append(mask_scecond_n(16384, 32768))
    for n in (2, 4, 8):
        generated.append(gen_mask(n, 32768))
    for pos in range(16):
        generated.append(mask_first_n_mod(16, 1024, pos, 32, 32768, "mask_first_n_mod"))
    for idx in range(32):
        generated.append(mask_channel(idx, num_channel=16, spatial_size=1024, num_cipher=1))
    generated.append(slot_conversion_mask(32768, 8192))

    generated.append(mask_first_n(8192, 16384))
    generated.append(mask_scecond_n(8192, 16384))
    for n in (2, 4):
        generated.append(gen_mask(n, 16384))
    for pos in range(32):
        generated.append(mask_first_n_mod(8, 256, pos, 64, 16384, "mask_first_n_mod2"))
    for idx in range(64):
        generated.append(mask_channel(idx, num_channel=32, spatial_size=256, num_cipher=1))
    generated.append(slot_conversion_mask(16384, 4096))

    generated.append(mask_mod(64, 1.0 / 64.0, 4096))

    for name, values in generated:
        if name in raw:
            raise ValueError(f"generated plaintext name collides with existing key: {name}")
        raw[name] = values


def build_weight_arrays(weight_dir: Path) -> dict[str, np.ndarray]:
    raw = {path.stem: read_bin_vector(path) for path in sorted(weight_dir.glob("*.bin"))}
    add_generated_resnet20_aespa_plaintexts(raw)
    return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-dir", type=Path, default=DEFAULT_WEIGHT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = build_weight_arrays(args.weight_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **raw)

    total_values = sum(values.size for values in raw.values())
    generated_count = len(raw) - len(list(args.weight_dir.glob("*.bin")))
    print(f"wrote {args.output}")
    print(f"arrays={len(raw)} generated={generated_count} total_values={total_values}")
    print(f"float64_size={total_values * 8 / 1024 / 1024:.2f} MiB before compression")


if __name__ == "__main__":
    main()
