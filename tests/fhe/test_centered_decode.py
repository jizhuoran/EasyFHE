import numpy as np
import pytest

import easyfhe as torch
import easyfhe.fhe as fhe


_VALUES_REAL = np.asarray(
    [-0.349895, 0.78553, -1.176304, 1.149476], dtype=np.float64
)
_VALUES_COMPLEX = _VALUES_REAL + np.asarray(
    [0.125j, -0.25j, 0.375j, -0.5j], dtype=np.complex128
)
_LARGE_NEGATIVE_SCALAR = -9475.7401


def _multiply_by_large_negative_scalar(client, context, values, *, slots=8):
    scaling_factor = (
        context.scale_at(context.max_limbs)
        if context.scale_mode == "flexible"
        else None
    )
    cipher = client.encrypt(
        values,
        slots=slots,
        scaling_factor=scaling_factor,
    )
    scalar = fhe.encode_scalar(
        _LARGE_NEGATIVE_SCALAR,
        cur_limbs=cipher.state.cur_limbs,
        scale_degree=1,
        scaling_factor=cipher.state.scaling_factor,
        context=context,
    )
    return fhe.homo_mul_scalar_rescale(cipher, scalar, context)


@pytest.mark.parametrize(
    ("scale_mode", "chain_kwargs"),
    [
        ("fixed", {"depth": 8, "dcrt_bits": 30, "first_mod": 35}),
        (
            "flexible",
            {"limb_specs": (35, 29, 31, 30, 32, 29, 30, 31, 30)},
        ),
    ],
)
def test_native_cpu_centered_decode_handles_large_negative_complex_values(
    scale_mode, chain_kwargs
):
    client, context = fhe.generate_client_context(
        fhe.CKKSContextSpec(
            log_n=6,
            dnum=1,
            scale_mode=scale_mode,
            rescale_policy="manual",
            **chain_kwargs,
        ),
        device="cpu",
    )
    result = _multiply_by_large_negative_scalar(
        client, context, _VALUES_COMPLEX
    )

    decoded = client.decrypt(result, complex_output=True).cpu().numpy()

    np.testing.assert_allclose(
        decoded[: _VALUES_COMPLEX.size],
        _VALUES_COMPLEX * _LARGE_NEGATIVE_SCALAR,
        rtol=1e-5,
        atol=1e-4,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FHE kernels required")
def test_native_cuda_centered_decode_handles_large_negative_fixed_values():
    client, context = fhe.generate_client_context(
        fhe.CKKSContextSpec(
            depth=8,
            log_n=14,
            dnum=1,
            dcrt_bits=59,
            first_mod=60,
            scale_mode="fixed",
            rescale_policy="manual",
        ),
        device="cuda",
    )
    result = _multiply_by_large_negative_scalar(client, context, _VALUES_REAL)

    assert result.state.scaling_factor == 2.0**client.dcrt_bits
    decoded = client.decrypt(result).cpu().numpy()

    np.testing.assert_allclose(
        decoded[: _VALUES_REAL.size],
        _VALUES_REAL * _LARGE_NEGATIVE_SCALAR,
        rtol=1e-5,
        atol=1e-4,
    )
