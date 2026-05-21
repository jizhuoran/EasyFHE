from types import SimpleNamespace

import easyfhe as torch
import pytest

import easyfhe.fhe as fhe
from easyfhe.fhe.ciphertext import Cipher
from easyfhe.fhe.ops import arithmetic, kernels, plaintext, primitives, rotation


def _cipher(name, *, cur_limbs=3, scaling_factor=2.0, noise_deg=1, slots=8, is_ext=False):
    cipher = Cipher(
        [f"{name}.c0", f"{name}.c1"],
        cur_limbs=cur_limbs,
        scaling_factor=scaling_factor,
        noise_deg=noise_deg,
        slots=slots,
        is_ext=is_ext,
    )
    cipher.name = name
    return cipher


def _manual_fixed_context():
    return SimpleNamespace(rescale_policy="manual", scale_mode="fixed")


def _mul_context():
    return SimpleNamespace(
        rescale_policy="manual",
        scale_mode="fixed",
        L=4,
        N=4,
        mult_swk_bx="bx",
        mult_swk_ax="ax",
        rescale_divisor_at=lambda _level: 10.0,
        scale_at=lambda _level: 20.0,
    )


def test_homo_add_out_and_inplace(monkeypatch):
    def fake_add(left, right, context, *, out=None):
        assert context is ctx
        left_name = left.cv[0].split(".", 1)[0]
        right_name = right.cv[0].split(".", 1)[0]
        value = left.cipher_like(
            [f"add({left_name},{right_name}).c0", f"add({left_name},{right_name}).c1"],
            scaling_factor=7.0,
        )
        return value if out is None else out.replace_with(value)

    monkeypatch.setattr(arithmetic, "_cipher_add", fake_add)
    ctx = _manual_fixed_context()
    a = _cipher("a")
    b = _cipher("b")
    out = _cipher("out", cur_limbs=1, scaling_factor=1.0, noise_deg=2, slots=4)

    result = fhe.homo_add(a, b, ctx, out=out)

    assert result is out
    assert out.cv == ["add(a,b).c0", "add(a,b).c1"]
    assert out.cur_limbs == 3
    assert out.scaling_factor == 7.0
    assert out.noise_deg == 1
    assert out.slots == 8
    assert out.is_ext is False

    inplace = fhe.homo_add_inplace(a, b, ctx)
    assert inplace is a
    assert a.cv == ["add(a,b).c0", "add(a,b).c1"]


def test_homo_sub_out_and_inplace(monkeypatch):
    monkeypatch.setattr(
        arithmetic,
        "_cipher_sub",
        lambda left, right, context, *, out=None: (out or left).replace_with(
            left.cipher_like(
                [f"sub({left.cv[0].split('.', 1)[0]},{right.cv[0].split('.', 1)[0]}).c0", "sub.c1"]
            )
        ),
    )
    ctx = _manual_fixed_context()
    a = _cipher("a")
    b = _cipher("b")
    out = _cipher("out")

    assert fhe.homo_sub(a, b, ctx, out=out) is out
    assert out.cv == ["sub(a,b).c0", "sub.c1"]
    assert fhe.homo_sub_inplace(a, b, ctx) is a
    assert a.cv == ["sub(a,b).c0", "sub.c1"]


def test_plaintext_and_scalar_out_and_inplace(monkeypatch):
    monkeypatch.setattr(
        plaintext,
        "_cipher_add_plain",
        lambda cipher, plain, context, *, out=None: (out or cipher).replace_with(
            cipher.cipher_like(["add_pt.c0", cipher.cv[1]])
        ),
    )
    monkeypatch.setattr(
        plaintext,
        "_cipher_add_scalar",
        lambda cipher, scalar, context, *, out=None: (out or cipher).replace_with(
            cipher.cipher_like([f"add_scalar({scalar}).c0", cipher.cv[1]])
        ),
    )
    monkeypatch.setattr(
        plaintext,
        "_cipher_sub_scalar",
        lambda cipher, scalar, context, *, out=None: (out or cipher).replace_with(
            cipher.cipher_like([f"sub_scalar({scalar}).c0", cipher.cv[1]])
        ),
    )

    ctx = SimpleNamespace(moduliQ_scalar=[257, 263, 269], scale_at=lambda _level: 1.0)
    cipher = _cipher("cipher")
    plain = _cipher("plain")
    out = _cipher("out")

    assert fhe.homo_add_pt(cipher, plain, ctx, out=out) is out
    assert out.cv == ["add_pt.c0", "cipher.c1"]

    assert fhe.homo_add_pt_inplace(cipher, plain, ctx) is cipher
    assert cipher.cv == ["add_pt.c0", "cipher.c1"]

    out2 = _cipher("out2")
    assert fhe.homo_add_scalar_int(cipher, 5, ctx, out=out2) is out2
    assert out2.cv[0] == "add_scalar(5).c0"

    assert fhe.homo_sub_scalar_int(cipher, 6, ctx, out=out2) is out2
    assert out2.cv[0] == "sub_scalar(6).c0"

    assert fhe.homo_add_scalar_int_inplace(cipher, 7, ctx) is cipher
    assert cipher.cv[0] == "add_scalar(7).c0"


def test_plaintext_and_scalar_multiply_out_and_inplace(monkeypatch):
    monkeypatch.setattr(
        plaintext,
        "_cipher_mul_plain",
        lambda cipher, plain, context, *, out=None: (out or cipher).replace_with(
            cipher.cipher_like(
                ["mul_pt.c0", "mul_pt.c1"],
                scaling_factor=11.0,
                noise_deg=2,
            )
        ),
    )
    monkeypatch.setattr(
        plaintext,
        "_cipher_mul_scalar_int",
        lambda cipher, scalar, context, *, out=None: (out or cipher).replace_with(
            cipher.cipher_like([f"mul_int({scalar}).c0", cipher.cv[1]])
        ),
    )
    monkeypatch.setattr(
        plaintext,
        "_cipher_mul_scalar_double",
        lambda cipher, scalar, context, *, out=None: (out or cipher).replace_with(
            cipher.cipher_like(
                [f"mul_double({tuple(scalar)}).c0", cipher.cv[1]],
                scaling_factor=13.0,
                noise_deg=2,
            )
        ),
    )

    ctx = SimpleNamespace(moduliQ_scalar=[257, 263, 269], scale_at=lambda _level: 1.0)
    cipher = _cipher("cipher")
    plain = _cipher("plain")
    out = _cipher("out")

    assert fhe.homo_mul_pt(cipher, plain, ctx, out=out) is out
    assert out.cv == ["mul_pt.c0", "mul_pt.c1"]
    assert out.scaling_factor == 11.0
    assert out.noise_deg == 2

    assert fhe.homo_mul_pt_inplace(cipher, plain, ctx) is cipher
    assert cipher.cv == ["mul_pt.c0", "mul_pt.c1"]

    out2 = _cipher("out2")
    assert fhe.homo_mul_scalar_int(cipher, 3, ctx, out=out2) is out2
    assert out2.cv[0] == "mul_int(3).c0"

    cipher2 = _cipher("cipher2")
    assert fhe.homo_mul_scalar_double(cipher2, 2.0, ctx, out=out2) is out2
    assert out2.cv[0] == "mul_double((2, 2, 2)).c0"

    assert fhe.homo_mul_scalar_double_inplace(cipher2, 4.0, ctx) is cipher2
    assert cipher2.cv[0] == "mul_double((4, 4, 4)).c0"


def test_rotate_does_not_expose_out(monkeypatch):
    ctx = SimpleNamespace(
        L=3,
        N=8,
        rotation_key_limb_limits={},
        get_rotation_key=lambda _index: ("swk_bx", "swk_ax"),
        get_inverse_precompute_auto=lambda _index: "inverse_precompute",
    )
    monkeypatch.setattr(rotation.F, "cv_hrot", lambda *args, **kwargs: ("rot.c0", "rot.c1"))

    cipher = _cipher("cipher")
    out = _cipher("out")

    with pytest.raises(TypeError):
        fhe.homo_rotate(cipher, 1, ctx, out=out)


def test_cv_hrot_uses_allocating_native_op(monkeypatch):
    seen = {}

    def fake_hrot(*args, **kwargs):
        seen["op"] = "hrot"
        seen["curr_limbs"] = kwargs["curr_limbs"]
        return ("rot.c0", "rot.c1")

    monkeypatch.setattr(kernels.torch, "hrot", fake_hrot, raising=False)
    ctx = SimpleNamespace(
        alpha=2,
        L=3,
        N=4,
        hat_inverse_vec_modup="hat_modup",
        hat_inverse_vec_shoup_modup="hat_shoup_modup",
        prod_q_i_mod_q_j_modup=[None, None, "prod_modup"],
        hat_inverse_vec_moddown="hat_moddown",
        hat_inverse_vec_shoup_moddown="hat_shoup_moddown",
        prod_q_i_mod_q_j_moddown="prod_moddown",
        prod_inv_moddown="prod_inv",
        prod_inv_shoup_moddown="prod_inv_shoup",
        primes="primes",
        barret_ratio="barret_ratio",
        barret_k="barret_k",
        power_of_roots_shoup="roots_shoup",
        power_of_roots="roots",
        inverse_power_of_roots_div_two="inv_roots",
        inverse_scaled_power_of_roots_div_two="inv_scaled_roots",
        inner_workspace="workspace",
    )
    result = kernels.cv_hrot(
        torch.zeros((3, 4), dtype=torch.uint64),
        torch.zeros((3, 4), dtype=torch.uint64),
        3,
        3,
        "swk_bx",
        "swk_ax",
        "precompute",
        ctx,
    )

    assert result == ("rot.c0", "rot.c1")
    assert seen == {"op": "hrot", "curr_limbs": 3}


def test_cv_add_uses_out_view(monkeypatch):
    seen = {}

    def fake_add_mod(x, y, modulus, *, cur_limbs, out=None):
        seen["x_shape"] = tuple(x.shape)
        seen["out_shape"] = tuple(out.shape)
        seen["cur_limbs"] = cur_limbs
        return out

    monkeypatch.setattr(kernels.torch, "add_mod", fake_add_mod)
    x = torch.zeros((3, 4), dtype=torch.uint64)
    y = torch.zeros((3, 4), dtype=torch.uint64)
    out = torch.empty((3, 4), dtype=torch.uint64)

    assert kernels.cv_add(x, y, torch.ones((3,), dtype=torch.uint64), 3, out=out) is out
    assert seen == {"x_shape": (1, 1, 3, 4), "out_shape": (1, 1, 3, 4), "cur_limbs": 3}


def test_cipher_add_inplace_uses_component_out(monkeypatch):
    seen = []

    def fake_cv_add(left, right, modulus, cur_limbs, inplace=False, out=None):
        seen.append((left, right, out, inplace))
        return out

    monkeypatch.setattr(primitives.F, "cv_add", fake_cv_add)
    ctx = SimpleNamespace(rescale_policy="manual", scale_mode="fixed", moduliQ="q")
    a = Cipher(
        [torch.zeros((3, 4), dtype=torch.uint64), torch.ones((3, 4), dtype=torch.uint64)],
        3,
        2.0,
        1,
        8,
        False,
    )
    b = Cipher(
        [torch.full((3, 4), 2, dtype=torch.uint64), torch.full((3, 4), 3, dtype=torch.uint64)],
        3,
        2.0,
        1,
        8,
        False,
    )

    assert fhe.homo_add_inplace(a, b, ctx) is a
    assert len(seen) == 2
    assert seen[0][2] is a.cv[0]
    assert seen[1][2] is a.cv[1]
    assert seen[0][3] is False


def test_cipher_add_uses_fused_out_when_available(monkeypatch):
    calls = {}

    def fake_available(name, *tensors):
        return name == "fused_add_mod_write"

    def fake_fused_add_mod_write(out0, out1, *args):
        calls["out0"] = out0
        calls["out1"] = out1
        calls["cur_limbs"] = args[-1]
        return (out0, out1)

    monkeypatch.setattr(primitives, "_fused_cuda_available", fake_available)
    monkeypatch.setattr(primitives.F, "cv_fused_add_pair_write", fake_fused_add_mod_write)
    ctx = SimpleNamespace(rescale_policy="manual", scale_mode="fixed", moduliQ="q")
    a = Cipher([torch.zeros((3, 4), dtype=torch.uint64), torch.zeros((3, 4), dtype=torch.uint64)], 3, 2.0, 1, 8, False)
    b = Cipher([torch.zeros((3, 4), dtype=torch.uint64), torch.zeros((3, 4), dtype=torch.uint64)], 3, 2.0, 1, 8, False)
    out = Cipher([torch.empty((3, 4), dtype=torch.uint64), torch.empty((3, 4), dtype=torch.uint64)], 3, 1.0, 2, 4, False)

    assert fhe.homo_add(a, b, ctx, out=out) is out
    assert calls == {"out0": out.cv[0], "out1": out.cv[1], "cur_limbs": 3}
    assert out.scaling_factor == a.scaling_factor
    assert out.slots == a.slots


def test_mul_rescale_and_square_do_not_expose_out(monkeypatch):
    ctx = _mul_context()
    a = _cipher("a", cur_limbs=4)
    b = _cipher("b", cur_limbs=4)

    monkeypatch.setattr(
        arithmetic.F,
        "cv_hmul_double_rescale",
        lambda *args, **kwargs: torch.zeros((2, 1, 3, 4), dtype=torch.uint64),
    )

    with pytest.raises(TypeError):
        fhe.homo_mul_rescale(a, b, ctx, out=_cipher("out"))

    result = fhe.homo_mul_rescale(a, b, ctx)
    assert result is not a
    assert result.cur_limbs == 3
    assert result.noise_deg == 1
    assert result.scaling_factor == 4.0
    assert len(result.cv) == 2

    monkeypatch.setattr(
        arithmetic,
        "_cipher_square",
        lambda cipher, context: cipher.cipher_like(["square.c0", "square.c1", "square.c2"], noise_deg=2),
    )
    monkeypatch.setattr(
        arithmetic,
        "_relinearize",
        lambda cipher, context: cipher.cipher_like(["relin.c0", "relin.c1"], noise_deg=2),
    )

    with pytest.raises(TypeError):
        arithmetic.homo_square(a, ctx, out=_cipher("square_out"))

    square = arithmetic.homo_square(a, ctx)
    assert square.cv == ["relin.c0", "relin.c1"]
    assert square.noise_deg == 2


@pytest.mark.parametrize(
    "name",
    [
        "homo_add_inplace",
        "homo_sub_inplace",
        "homo_add_pt_inplace",
        "homo_mul_pt_inplace",
        "homo_add_scalar_double_inplace",
        "homo_add_scalar_int_inplace",
        "homo_mul_scalar_double_inplace",
        "homo_mul_scalar_int_inplace",
        "homo_sub_scalar_int",
        "homo_sub_scalar_int_inplace",
        "reduce_noise_to_one",
    ],
)
def test_new_out_inplace_symbols_are_public(name):
    assert hasattr(fhe, name)


@pytest.mark.parametrize(
    "name",
    [
        "homo_mul_double_rescale",
        "homo_square",
        "fused_broadcast_mac",
    ],
)
def test_internal_symbols_are_not_root_public(name):
    assert not hasattr(fhe, name)
