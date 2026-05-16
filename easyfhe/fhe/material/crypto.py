from __future__ import annotations

from typing import Optional

import numpy as np
import easyfhe as torch

from ..ciphertext import Cipher
from ..ops import kernels as F
from ..ops.encoding import encode
from .key_material import ContextKeyMaterial
from .sample_arithmetic import (
    CipherArrays,
    CkksParams,
    KeyMaterial as SampleKeyMaterial,
    as_moduli_q,
    as_uint64_matrix,
    decode_ckks_phase,
    decrypt_phase_from_arrays,
)


UIntArray = np.ndarray


def ckks_params_from_context(context) -> CkksParams:
    return CkksParams(
        moduli_q=np.asarray(context.moduliQ_scalar, dtype=np.uint64),
        roots_q=(
            np.asarray(context.rootsQ, dtype=np.uint64)
            if hasattr(context, "rootsQ")
            else None
        ),
        moduli_p=np.asarray(context.moduliP_scalar, dtype=np.uint64),
        scaling_factors=np.asarray(context.scalingFactorsReal, dtype=np.float64),
        depth=getattr(context, "L", None),
    )


def _encrypt(pk0, pk1, ptx, device, context):
    logn = context.logN
    cur_limbs = ptx.cur_limbs
    l = cur_limbs
    nh = context.N // 2

    def _cpu(value):
        return value.cpu() if torch.is_tensor(value) else value

    target_device = device
    moduli_p = torch.from_numpy(context.moduliP_scalar)
    moduli_q = torch.from_numpy(context.moduliQ_scalar)
    cv = torch.encrypt(
        ptx=ptx.cv[0].cpu(),
        pk0=pk0.cpu(),
        pk1=pk1.cpu(),
        l=l,
        logn=logn,
        nh=nh,
        moduliP_scalar=moduli_p,
        moduliQ_scalar=moduli_q,
        primes=_cpu(context.QplusP_map[cur_limbs]),
        max_int_diffs=_cpu(context.QmaxdiffplusPmaxdiff_map[cur_limbs]),
        barret_ratio=_cpu(context.QbarretRatioplusPbarretRatio_map[cur_limbs]),
        barret_k=_cpu(context.QbarretKplusPbarretK_map[cur_limbs]),
        power_of_roots_shoup=_cpu(context.power_of_roots_shoup),
        power_of_roots=_cpu(context.power_of_roots),
    )
    bx, ax = cv
    n = 1 << logn
    if bx.numel() != l * n:
        raise RuntimeError(f"Unexpected encrypted tensor size: got {bx.numel()}, expected {l * n}")
    return Cipher(
        [bx.view(l, n).to(target_device), ax.view(l, n).to(target_device)],
        cur_limbs,
        ptx.scaling_factor,
        ptx.noise_deg,
        ptx.slots,
        is_ext=False,
    )


def _raise_plaintext_scale_degree(ptx, scale_deg, context):
    if scale_deg == ptx.noise_deg:
        return ptx
    if scale_deg < 1 or ptx.noise_deg != 1:
        raise ValueError(f"unsupported plaintext scale degree transition: {ptx.noise_deg} -> {scale_deg}")

    cur_limbs = ptx.cur_limbs
    base_scale = context.scale_at(cur_limbs)
    base_scale_int = round(base_scale)
    scale_multiplier = [
        pow(base_scale_int, scale_deg - 1, int(context.moduliQ_scalar[i]))
        for i in range(cur_limbs)
    ]
    scale_multiplier = F.gen_scalar_tensor(
        scale_multiplier,
        context.moduliQ_scalar,
        cur_limbs,
    ).to(ptx.cv[0].device)
    cv = [
        F.cv_mul_scalar(
            ptx.cv[0],
            scale_multiplier,
            context.moduliQ,
            context.q_mu,
            cur_limbs,
        )
    ]
    return ptx.cipher_like(
        cv,
        scaling_factor=base_scale ** scale_deg,
        noise_deg=scale_deg,
    )


def encrypt_with_key_arrays(x, device, scale_deg, level, slots, public_key_b, public_key_a, context):
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    _, ptx = encode(x, context, level=level, slots=slots, is_ext=False)
    ptx = _raise_plaintext_scale_degree(ptx, scale_deg, context)
    cur_limbs = ptx.cur_limbs
    pk0 = torch.as_tensor(public_key_b[:cur_limbs], device=device, dtype=torch.uint64)
    pk1 = torch.as_tensor(public_key_a[:cur_limbs], device=device, dtype=torch.uint64)
    return _encrypt(pk0, pk1, ptx, device, context)


def encrypt_with_key_material(
    x,
    context,
    key_material: ContextKeyMaterial,
    *,
    device=None,
    scale_deg=1,
    level=0,
    slots=0,
):
    return encrypt_with_key_arrays(
        x,
        device or context.device,
        scale_deg,
        level,
        slots,
        key_material.public_key_b,
        key_material.public_key_a,
        context,
    )


def decrypt_phase(cipher, secret_key: object, moduli_q: object) -> UIntArray:
    """Return phase = ct0 + ct1 * s mod qi in evaluation format."""

    if len(cipher.cv) != 2:
        raise ValueError(f"Expected a degree-1 ciphertext with two components, got {len(cipher.cv)}")
    ct0 = as_uint64_matrix("ct0", cipher.cv[0].detach().cpu().numpy())
    ct1 = as_uint64_matrix("ct1", cipher.cv[1].detach().cpu().numpy())
    key = SampleKeyMaterial(
        sk=as_uint64_matrix("secret_key", secret_key)[: cipher.cur_limbs],
        pk_b=np.zeros_like(ct0),
        pk_a=np.zeros_like(ct0),
    )
    return decrypt_phase_from_arrays(
        CipherArrays(ct0=ct0, ct1=ct1),
        key,
        CkksParams(moduli_q=as_moduli_q(moduli_q)),
    )


def decrypt_phase_with_key_material(cipher, context, key_material: ContextKeyMaterial):
    params = key_material.params or ckks_params_from_context(context)
    phase = decrypt_phase(cipher, key_material.secret_key, params.moduli_q)
    return torch.tensor(phase, device=cipher.cv[0].device, dtype=torch.uint64)


def decrypt_with_key_material(cipher, context, key_material: ContextKeyMaterial):
    params = key_material.params or ckks_params_from_context(context)
    phase = decrypt_phase(cipher, key_material.secret_key, params.moduli_q)
    decoded = decode_ckks_phase(
        phase,
        params,
        plaintext_modulus_bits=getattr(context, "dcrtBits"),
        noise_scale_deg=getattr(cipher, "noise_deg", 1),
        scaling_factor=getattr(cipher, "scaling_factor", None),
        slots=getattr(cipher, "slots", 0),
    )
    return torch.tensor(decoded, device=cipher.cv[0].device, dtype=torch.float64)
