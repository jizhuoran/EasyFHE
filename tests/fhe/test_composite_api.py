from types import SimpleNamespace

import numpy as np
import pytest

import easyfhe as torch
import easyfhe.fhe as fhe
from easyfhe.fhe.ciphertext import Cipher, CipherState, EncodedScalar
from easyfhe.fhe.ops import arithmetic, kernels, layout, rotation


def _cipher(name, *, batch_size=1):
    return Cipher(
        [f"{name}.c0", f"{name}.c1"],
        CipherState(3, 1, 2.0),
        slots=8,
        is_ext=False,
        batch_size=batch_size,
    )


def test_multiply_rescale_compositions_consume_one_rescale(monkeypatch):
    calls = []
    product = _cipher("product")
    result = _cipher("result")

    monkeypatch.setattr(
        arithmetic,
        "homo_mul_pt",
        lambda cipher, plaintext, context: calls.append(
            ("mul_pt", cipher, plaintext, context)
        )
        or product,
    )
    monkeypatch.setattr(
        arithmetic.alignment,
        "rescale",
        lambda cipher, context: calls.append(("rescale", cipher, context)) or result,
    )
    context = object()
    cipher = _cipher("cipher")
    plaintext = _cipher("plaintext")

    assert fhe.homo_mul_pt_rescale(cipher, plaintext, context) is result
    assert calls == [
        ("mul_pt", cipher, plaintext, context),
        ("rescale", product, context),
    ]


def test_scalar_rescale_requires_scaled_encoded_scalar(monkeypatch):
    scalar = EncodedScalar([1, 1, 1], 3, 1, 2.0)
    integer = EncodedScalar([1, 1, 1], 3, 0, 1.0)
    product = _cipher("product")
    result = _cipher("result")
    context = object()
    cipher = _cipher("cipher")

    monkeypatch.setattr(arithmetic, "homo_mul_scalar", lambda *_args: product)
    monkeypatch.setattr(arithmetic.alignment, "rescale", lambda *_args: result)

    assert fhe.homo_mul_scalar_rescale(cipher, scalar, context) is result
    try:
        fhe.homo_mul_scalar_rescale(cipher, integer, context)
    except ValueError as error:
        assert "scale_degree=1" in str(error)
    else:
        raise AssertionError("integer scalar unexpectedly accepted by rescale composition")


def test_grouped_and_hoisted_rescale_compositions(monkeypatch):
    product = _cipher("product")
    result = _cipher("result")
    context = object()
    calls = []

    monkeypatch.setattr(
        arithmetic,
        "grouped_pairwise_mac",
        lambda *args: calls.append(("grouped", args)) or product,
    )
    monkeypatch.setattr(
        arithmetic.alignment,
        "rescale",
        lambda cipher, ctx: calls.append(("rescale", cipher, ctx)) or result,
    )
    assert (
        fhe.grouped_pairwise_mac_rescale("ciphers", "plaintexts", 4, context)
        is result
    )

    calls.clear()
    monkeypatch.setattr(
        rotation,
        "hoisted_mac_sum",
        lambda *args, **kwargs: calls.append(("hoisted", args, kwargs)) or product,
    )
    # The function imports rescale locally from alignment, so patch that module.
    from easyfhe.fhe.ops import alignment

    monkeypatch.setattr(
        alignment,
        "rescale",
        lambda cipher, ctx: calls.append(("rescale", cipher, ctx)) or result,
    )
    assert (
        fhe.hoisted_mac_sum_rescale(
            "cipher",
            (0, 1),
            "plaintexts",
            8,
            2,
            context,
            strategy="normal",
        )
        is result
    )
    assert calls[-1] == ("rescale", product, context)


def test_sum_cipher_batch_uses_cipher_addition(monkeypatch):
    batched = _cipher("batch", batch_size=3)
    items = (_cipher("a"), _cipher("b"), _cipher("c"))
    context = SimpleNamespace()
    calls = []

    monkeypatch.setattr(layout, "unpack_cipher_batch", lambda value: items)
    monkeypatch.setattr(
        arithmetic,
        "_cipher_add",
        lambda left, right, ctx: calls.append((left, right, ctx))
        or _cipher(f"sum{len(calls)}"),
    )

    result = fhe.sum_cipher_batch(batched, context)
    assert result.cv[0] == "sum2.c0"
    assert len(calls) == 2


def test_removed_legacy_names_are_not_public():
    removed = (
        "UnpackedRaw",
        "rescale_one_level",
        "reduce_noise_to_one",
        "reduce_level_to_one",
        "homo_add_scalar_double",
        "homo_add_scalar_int",
        "homo_mul_scalar_double",
        "homo_mul_scalar_int",
        "HOIST_NORMAL",
        "HOIST_EXT_NORMAL",
        "HOIST_EXT_DOUBLE_HOIST",
        "encode_stage1_packed",
    )
    assert [name for name in removed if hasattr(fhe, name)] == []


def test_cpu_rescale_wrapper_adapts_component_to_native_four_dimensions(monkeypatch):
    component = torch.zeros((2, 3, 8), dtype=torch.uint64)
    seen = {}

    def fake_rescale(native_input, **kwargs):
        seen["shape"] = tuple(native_input.shape)
        seen["cur_limbs"] = kwargs["curr_limbs"]
        return torch.zeros((1, 2, 2, 8), dtype=torch.uint64)

    monkeypatch.setattr(kernels.torch, "rescale_one_level", fake_rescale)
    context = SimpleNamespace(
        L=3,
        N=8,
        primes_list=[11, 13, 17],
        primes="primes",
        switch_modulus_map="switch",
        power_of_roots_shoup="roots_shoup",
        power_of_roots="roots",
        inverse_power_of_roots_div_two="inverse_roots",
        inverse_scaled_power_of_roots_div_two="scaled_inverse_roots",
        qlql_inv_mod_ql_div_ql_mod_q="qlql",
        qlql_inv_mod_ql_div_ql_mod_q_shoup="qlql_shoup",
        q_inv_mod_q="q_inv",
        q_inv_mod_q_shoup="q_inv_shoup",
    )

    result = kernels.cv_rescale_one_level(component, 3, 0, context)
    assert seen == {"shape": (1, 2, 3, 8), "cur_limbs": 3}
    assert tuple(result.shape) == (2, 2, 8)


@pytest.mark.parametrize(
    ("scale_mode", "rescale_policy"),
    (("fixed", "manual"), ("flexible", "auto")),
)
def test_typed_constant_multiply_rescale_round_trip_cpu(
    scale_mode, rescale_policy
):
    client, context = fhe.generate_client_context(
        fhe.CKKSContextSpec(
            depth=4,
            log_n=6,
            dnum=1,
            dcrt_bits=30,
            first_mod=35,
            scale_mode=scale_mode,
            rescale_policy=rescale_policy,
        ),
        device="cpu",
    )
    values = np.linspace(-0.5, 0.5, 32, dtype=np.float64)
    weights = np.full(32, 0.25, dtype=np.float64)
    scaling_factor = (
        context.scale_at(context.max_limbs) if scale_mode == "flexible" else None
    )
    cipher = client.encrypt(
        values, slots=32, scaling_factor=scaling_factor
    )
    constants = fhe.ConstantBundle(
        vectors={"weights": fhe.PackedRaw(torch.from_numpy(weights))},
        scalars={"gain": 0.5, "shift": 0.125},
    )

    plaintext = constants.plaintext(
        "weights", state=cipher.state, slots=32, context=context
    )
    product = fhe.homo_mul_pt_rescale(cipher, plaintext, context)
    gain = constants.encoded_scalars(
        "gain",
        cur_limbs=product.state.cur_limbs,
        scale_degree=1,
        scaling_factor=product.state.scaling_factor,
        context=context,
    )[0]
    scaled = fhe.homo_mul_scalar_rescale(product, gain, context)
    shift = constants.encoded_scalars(
        "shift",
        cur_limbs=scaled.state.cur_limbs,
        scale_degree=scaled.state.scale_degree,
        scaling_factor=scaled.state.scaling_factor,
        context=context,
    )[0]
    output = fhe.homo_add_scalar(scaled, shift, context)

    expected = values * 0.25 * 0.5 + 0.125
    np.testing.assert_allclose(
        client.decrypt(output).numpy()[:32], expected, rtol=1e-5, atol=1e-5
    )
