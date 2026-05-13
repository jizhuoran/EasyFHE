from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import easyfhe as torch

from .sample_arithmetic import (
    CipherArrays,
    CkksParams,
    EncryptSamples,
    KeyMaterial,
    KeygenSamples,
    SampleBundle,
    decode_ckks_phase,
)


def _native_uint64_array(value):
    arr = np.asarray(value)
    if arr.dtype != np.uint64:
        arr = arr.astype(np.uint64, copy=False)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def _native_op(name):
    op = getattr(torch, name, None)
    if op is not None:
        return op
    aten_ops = getattr(getattr(torch, "ops", None), "aten", None)
    op = getattr(aten_ops, name, None) if aten_ops is not None else None
    if op is None:
        raise RuntimeError(
            f"easyfhe does not expose {name}; rebuild EasyFHE after registering the native sampler ATen op"
        )
    return op


def _tensor_to_numpy(value):
    if torch.is_tensor(value):
        value = value.cpu().numpy()
    return np.ascontiguousarray(np.asarray(value))


def _tensor_scalar_int(value):
    if torch.is_tensor(value):
        return int(value.cpu().item())
    return int(value)


def _flatten_rotation_groups(rotation_index_groups, rotation_indices=None):
    groups = [list(group) for group in (rotation_index_groups or [])]
    if not groups and rotation_indices:
        groups = [list(rotation_indices)]
    flat = []
    offsets = [0]
    for group in groups:
        flat.extend(int(value) for value in group)
        offsets.append(len(flat))
    if not flat:
        offsets = []
    return flat, offsets


def _rotation_trim_arrays(trim_map):
    trim_map = dict(trim_map or {})
    keys = [int(key) for key in trim_map]
    values = [int(trim_map[key]) for key in trim_map]
    return keys, values


def _decode_rotation_tensors(tensors, index=0):
    if index >= len(tensors):
        return [], index
    manifest = _tensor_to_numpy(tensors[index]).astype(np.int64, copy=False).reshape(-1, 3)
    index += 1
    rotation_keys = []
    for auto_idx, limb, trimmed in manifest:
        if index + 1 >= len(tensors):
            raise ValueError("native sampler returned an incomplete rotation-key tensor list")
        key_b = _native_uint64_array(_tensor_to_numpy(tensors[index]))
        key_a = _native_uint64_array(_tensor_to_numpy(tensors[index + 1]))
        index += 2
        item = (int(auto_idx), key_b, key_a)
        if int(trimmed):
            item = item + (int(limb), True)
        rotation_keys.append(item)
    return rotation_keys, index


def _decode_ckks_tensors(tensors, *, include_encrypt_trace, include_eval_mult_key):
    names = [
        "moduli_q",
        "roots_q",
        "moduli_p",
        "roots_p",
        "p_mod_q",
        "depth",
        "sk",
        "sk_coeff",
        "pk_b",
        "pk_a",
        "pk_e",
    ]
    index = len(names)
    if len(tensors) < index:
        raise ValueError("native sampler returned too few tensors for CKKS material")
    raw = {name: _tensor_to_numpy(tensor) for name, tensor in zip(names, tensors[:index])}
    raw["depth"] = _tensor_scalar_int(tensors[5])

    if include_encrypt_trace:
        encrypt_names = [
            "v",
            "e0",
            "e1",
            "ct0_zero",
            "ct1_zero",
            "ptx",
            "cipher",
            "phase",
            "input_real",
            "actual_slots",
        ]
        for name in encrypt_names:
            raw[name] = _tensor_to_numpy(tensors[index])
            index += 1
        raw["actual_slots"] = _tensor_scalar_int(raw["actual_slots"])

    if include_eval_mult_key:
        eval_names = [
            "evalmult_sk_squared",
            "evalmult_sk_ext",
            "evalmult_p_mod_q",
            "evalmult_key_b",
            "evalmult_key_a",
            "evalmult_key_e",
        ]
        for name in eval_names:
            raw[name] = _tensor_to_numpy(tensors[index])
            index += 1

    if index != len(tensors):
        raise ValueError("native sampler returned trailing tensors after CKKS material")
    raw["rotation_keys"] = []
    return raw


@dataclass(frozen=True)
class CkksSamplerConfig:
    log_n: int = 12
    depth: int = 2
    dcrt_bits: int = 50
    first_mod: int = 60
    dnum: int = 3
    secret_key_dist: str = "SPARSE_TERNARY"
    scaling_technique: str = "FIXEDMANUAL"
    include_eval_mult_key: bool = False
    rotation_indices: Sequence[int] | None = None
    rotation_index_groups: Sequence[Sequence[int]] | None = None
    rotation_trim_limbs_by_auto_index: dict[int, int] | None = None
    include_encrypt_trace: bool = True
    include_decoded_real: bool = True
    random_mode: str = "sequential"
    rotation_random_mode: str = "fresh"


class NativeCppSampleProvider:
    """Adapter for the EasyFHE native ATen sampler ops."""

    def __init__(self, config: CkksSamplerConfig, module_name: str = "easyfhe_native_sampler"):
        self.config = config
        self.module_name = module_name

    def generate(
        self,
        values: Iterable[float] | None = None,
        *,
        scale_deg: int = 1,
        level: int = 0,
        slots: int = 0,
    ) -> SampleBundle:
        sample_ckks = _native_op("fhe_native_sample_ckks")
        value_tensor = torch.as_tensor(
            [] if values is None else list(values),
            dtype=torch.float64,
            device="cpu",
        )
        tensors = sample_ckks(
            value_tensor,
            int(self.config.log_n),
            int(self.config.depth),
            int(self.config.dcrt_bits),
            int(self.config.first_mod),
            int(self.config.dnum),
            str(self.config.secret_key_dist),
            str(self.config.scaling_technique),
            bool(self.config.include_eval_mult_key),
            bool(self.config.include_encrypt_trace),
            int(scale_deg),
            int(level),
            int(slots),
            str(self.config.random_mode),
            3.19,
            1,
        )
        raw = _decode_ckks_tensors(
            list(tensors),
            include_encrypt_trace=self.config.include_encrypt_trace,
            include_eval_mult_key=self.config.include_eval_mult_key,
        )
        required = {"moduli_q", "sk", "pk_a", "pk_e"}
        if self.config.include_encrypt_trace:
            required.update({"v", "e0", "e1"})
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(f"{self.module_name}.sample_ckks did not return required arrays: {missing}")

        expected_key = None
        if "pk_b" in raw:
            expected_key = KeyMaterial(sk=raw["sk"], pk_b=raw["pk_b"], pk_a=raw["pk_a"])

        expected_zero = None
        if "ct0_zero" in raw and "ct1_zero" in raw:
            expected_zero = CipherArrays(ct0=raw["ct0_zero"], ct1=raw["ct1_zero"])

        expected_cipher = None
        if "cipher" in raw:
            cipher = np.asarray(raw["cipher"], dtype=np.uint64)
            expected_cipher = CipherArrays(ct0=cipher[0], ct1=cipher[1])
        elif "ct0" in raw and "ct1" in raw:
            expected_cipher = CipherArrays(ct0=raw["ct0"], ct1=raw["ct1"])

        return SampleBundle(
            params=CkksParams(
                moduli_q=raw["moduli_q"],
                roots_q=raw.get("roots_q"),
                moduli_p=raw.get("moduli_p"),
                roots_p=raw.get("roots_p"),
                scaling_factors=raw.get("scaling_factors"),
                depth=raw.get("depth", self.config.depth),
            ),
            keygen=KeygenSamples(sk=raw["sk"], pk_a=raw["pk_a"], pk_e=raw["pk_e"]),
            encrypt=(
                EncryptSamples(v=raw["v"], e0=raw["e0"], e1=raw["e1"])
                if {"v", "e0", "e1"}.issubset(raw)
                else None
            ),
            ptx=raw.get("ptx"),
            expected_key=expected_key,
            expected_zero=expected_zero,
            expected_cipher=expected_cipher,
            expected_phase=raw.get("phase"),
            secret_key_coeff=raw.get("sk_coeff"),
            eval_mult_key_b=raw.get("evalmult_key_b"),
            eval_mult_key_a=raw.get("evalmult_key_a"),
            rotation_keys=raw.get("rotation_keys"),
            decoded_real=self._decoded_real(raw),
        )

    def _decoded_real(self, raw: dict):
        if "decoded_real" in raw:
            return raw["decoded_real"]
        if not self.config.include_decoded_real:
            return None
        if "phase" not in raw or raw.get("roots_q") is None:
            return None
        return decode_ckks_phase(
            raw["phase"],
            CkksParams(
                moduli_q=raw["moduli_q"],
                roots_q=raw.get("roots_q"),
                moduli_p=raw.get("moduli_p"),
                roots_p=raw.get("roots_p"),
                scaling_factors=raw.get("scaling_factors"),
                depth=raw.get("depth", self.config.depth),
            ),
            plaintext_modulus_bits=self.config.dcrt_bits,
            noise_scale_deg=1,
            slots=int(raw.get("actual_slots", 0)),
        )

    def generate_rotation_keys(
        self,
        secret_key: object,
        secret_key_coeff: object,
        rotation_index_groups: Sequence[Sequence[int]],
    ):
        sample_rotation_keys = _native_op("fhe_native_sample_rotation_keys")
        rotation_indices, rotation_offsets = _flatten_rotation_groups(rotation_index_groups)
        trim_auto_indices, trim_limbs = _rotation_trim_arrays(self.config.rotation_trim_limbs_by_auto_index)
        tensors = sample_rotation_keys(
            torch.as_tensor(np.asarray(secret_key, dtype=np.uint64), dtype=torch.uint64, device="cpu"),
            torch.as_tensor(np.asarray(secret_key_coeff, dtype=np.int64), dtype=torch.int64, device="cpu"),
            int(self.config.log_n),
            int(self.config.depth),
            int(self.config.dcrt_bits),
            int(self.config.first_mod),
            int(self.config.dnum),
            str(self.config.secret_key_dist),
            str(self.config.scaling_technique),
            rotation_indices,
            rotation_offsets,
            trim_auto_indices,
            trim_limbs,
            str(self.config.rotation_random_mode),
            str(self.config.random_mode),
            3.19,
            1,
        )
        rotation_keys, index = _decode_rotation_tensors(list(tensors), 0)
        if index != len(tensors):
            raise ValueError("native sampler returned trailing tensors after rotation-key material")
        normalized = []
        for item in rotation_keys:
            auto_idx, key_b, key_a = item[:3]
            normalized_item = (
                int(auto_idx),
                _native_uint64_array(key_b),
                _native_uint64_array(key_a),
            )
            if len(item) > 3:
                normalized_item = normalized_item + tuple(item[3:])
            normalized.append(normalized_item)
        return normalized
