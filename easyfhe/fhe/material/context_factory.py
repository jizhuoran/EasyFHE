from __future__ import annotations

import copy

import numpy as np

from ..runtime.options import RuntimeOptions
from .context_material import runtime_material_from_builder
from .context_material_builder import ContextMaterialBuilder


def _ensure_native_sampler_path():
    import easyfhe as torch

    aten_ops = getattr(getattr(torch, "ops", None), "aten", None)
    if getattr(torch, "fhe_native_sample_ckks", None) is None and getattr(aten_ops, "fhe_native_sample_ckks", None) is None:
        raise RuntimeError(
            "EasyFHE native context generation requires the ATen native sampler ops. "
            "Rebuild EasyFHE after registering fhe_native_sample_ckks in native_functions.yaml."
        )


def _plan_initial_rotations(rot_index_list, log_n, secret_key_dist):
    from .rotation import plan_rotation_groups

    return plan_rotation_groups(
        rot_index_list,
        [],
        [],
        log_n,
        secret_key_dist,
        True,
    )


def _make_sampler_config(
    log_n,
    depth,
    dcrt_bits,
    first_mod,
    dnum,
    secret_key_dist,
    rescale_tech,
    rotation_groups,
    rotation_trim_limbs_by_auto_index=None,
    rotation_random_mode="fresh",
):
    from .native_sampler import CkksSamplerConfig

    return CkksSamplerConfig(
        log_n=log_n,
        depth=depth,
        dcrt_bits=dcrt_bits,
        first_mod=first_mod,
        dnum=dnum,
        secret_key_dist=secret_key_dist,
        scaling_technique=rescale_tech,
        include_eval_mult_key=True,
        include_encrypt_trace=False,
        include_decoded_real=False,
        random_mode="parallel_deterministic",
        rotation_random_mode=str(rotation_random_mode),
        rotation_index_groups=rotation_groups,
        rotation_trim_limbs_by_auto_index=rotation_trim_limbs_by_auto_index,
    )


def _rotation_trim_limbs_by_auto_index(options, auto_idx_to_rot_idx, ring_dim):
    from .rotation import normalize_rotation_index

    max_limbs_by_rot = getattr(options, "rotation_key_limb_limits", None) or {}
    if not max_limbs_by_rot:
        return {}
    trim_map = {}
    for auto_idx, rot_idx in auto_idx_to_rot_idx.items():
        normalized_rot = normalize_rotation_index(rot_idx, ring_dim)
        if normalized_rot in max_limbs_by_rot:
            trim_map[int(auto_idx)] = int(max_limbs_by_rot[normalized_rot])
    return trim_map


def _sample_native_context_material(sampler_config, log_n):
    from .native_sampler import NativeCppSampleProvider

    provider = NativeCppSampleProvider(sampler_config)
    bundle = provider.generate([0.0], slots=max(1, 1 << (int(log_n) - 1)))
    if bundle.eval_mult_key_b is None or bundle.eval_mult_key_a is None:
        raise RuntimeError("native sampler did not return eval-mult key material")
    if bundle.expected_key is None:
        raise RuntimeError("native sampler did not return public-key material")
    if bundle.secret_key_coeff is None:
        raise RuntimeError("native sampler did not return coefficient-form secret key")
    return bundle


def _sample_native_rotation_keys(sampler_config, bundle, rotation_groups):
    if not rotation_groups:
        return []
    from .native_sampler import NativeCppSampleProvider

    provider = NativeCppSampleProvider(sampler_config)
    return provider.generate_rotation_keys(
        bundle.keygen.sk,
        bundle.secret_key_coeff,
        rotation_groups,
    )


def _make_ckks_params(bundle, crypto_context):
    from .sample_arithmetic import CkksParams

    return CkksParams(
        moduli_q=bundle.params.moduli_q,
        roots_q=bundle.params.roots_q,
        moduli_p=bundle.params.moduli_p,
        roots_p=bundle.params.roots_p,
        scaling_factors=crypto_context.scalingFactorsReal,
        depth=crypto_context.L - 1,
    )


def _make_key_material(bundle, crypto_context):
    from .key_material import ContextKeyMaterial

    return ContextKeyMaterial(
        secret_key=bundle.keygen.sk,
        public_key_b=bundle.expected_key.pk_b,
        public_key_a=bundle.expected_key.pk_a,
        params=_make_ckks_params(bundle, crypto_context),
        secret_key_coeff=bundle.secret_key_coeff,
    )


def _generation_metadata(spec):
    return {
        "depth": int(spec.depth),
        "logN": int(spec.log_n),
        "dnum": int(spec.dnum),
        "dcrtBits": int(spec.dcrt_bits),
        "firstMod": int(spec.first_mod),
        "secretKeyDist": spec.secret_key_dist,
        "rescaleTech": spec.rescale_tech,
    }


def build_context(context_cls, spec, device="cpu", options=None):
    if options is None:
        options = RuntimeOptions()
    _ensure_native_sampler_path()

    depth = int(spec.depth)
    rotation_groups, auto_idx_to_rot_idx = _plan_initial_rotations(
        spec.rotations,
        spec.log_n,
        spec.secret_key_dist,
    )
    sampler_config = _make_sampler_config(
        spec.log_n,
        depth,
        spec.dcrt_bits,
        spec.first_mod,
        spec.dnum,
        spec.secret_key_dist,
        spec.rescale_tech,
        rotation_groups,
        _rotation_trim_limbs_by_auto_index(options, auto_idx_to_rot_idx, 1 << int(spec.log_n)),
        getattr(options, "rotation_random_mode", "fresh"),
    )
    bundle = _sample_native_context_material(sampler_config, spec.log_n)
    rotation_keys = _sample_native_rotation_keys(sampler_config, bundle, rotation_groups)
    builder = ContextMaterialBuilder(
        spec.log_n,
        [],
        spec.dcrt_bits,
        60,
        spec.dnum,
        [],
        depth,
        np.asarray(bundle.params.moduli_q, dtype=np.uint64),
        np.asarray(bundle.params.moduli_p, dtype=np.uint64),
        np.asarray(bundle.params.roots_q, dtype=np.uint64),
        np.asarray(bundle.params.roots_p, dtype=np.uint64),
        np.asarray([bundle.eval_mult_key_b, bundle.eval_mult_key_a], dtype=np.uint64),
        {"native": list(rotation_keys)},
        auto_idx_to_rot_idx,
        spec.secret_key_dist,
        spec.rescale_tech,
        [0, 0],
        copy.copy(options),
    )

    context = context_cls(runtime_material_from_builder(builder), device, options)
    context.rootsQ = np.asarray(bundle.params.roots_q, dtype=np.uint64)
    context.rootsP = np.asarray(bundle.params.roots_p, dtype=np.uint64)
    context._attach_key_material(_make_key_material(bundle, context), sampler_config)
    context.context_generation_config = _generation_metadata(spec)
    print(
        "Generated EasyFHE native context "
        f"(depth={depth}, rotation keys={len(rotation_keys)})."
    )
    return context
