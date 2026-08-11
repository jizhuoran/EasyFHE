from types import SimpleNamespace

import easyfhe as torch
import pytest

import easyfhe.fhe as fhe
from easyfhe.fhe.ciphertext import Cipher, CipherState, EncodedScalar
from easyfhe.fhe.ops import arithmetic, kernels, primitives, rotation


def _cipher(name, *, cur_limbs=3, scaling_factor=2.0, scale_degree=1, slots=8, is_ext=False):
    cipher = Cipher(
        [f"{name}.c0", f"{name}.c1"],
        CipherState(cur_limbs, scale_degree, scaling_factor),
        slots=slots,
        is_ext=is_ext,
    )
    cipher.name = name
    return cipher


def _scalar(values, *, cur_limbs=None, scaling_factor=2.0, scale_degree=1):
    residues = torch.tensor(values, dtype=torch.uint64)
    cur_limbs = residues.shape[-1] if cur_limbs is None else int(cur_limbs)
    return EncodedScalar(residues, cur_limbs, scale_degree, scaling_factor)


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


def test_homo_add_does_not_expose_out_and_inplace_mutates_first_arg(monkeypatch):
    def fake_add(left, right, context):
        assert context is ctx
        left_name = left.cv[0].split(".", 1)[0]
        right_name = right.cv[0].split(".", 1)[0]
        return left.cipher_like(
            [f"add({left_name},{right_name}).c0", f"add({left_name},{right_name}).c1"],
            state=left.state.replace(scaling_factor=7.0),
        )

    def fake_add_inplace(left, right, context):
        return left.replace_with(fake_add(left, right, context))

    monkeypatch.setattr(arithmetic, "_cipher_add", fake_add)
    monkeypatch.setattr(arithmetic, "_cipher_add_inplace", fake_add_inplace)
    ctx = _manual_fixed_context()
    a = _cipher("a")
    b = _cipher("b")
    with pytest.raises(TypeError):
        fhe.homo_add(a, b, ctx, out=_cipher("out"))

    result = fhe.homo_add(a, b, ctx)
    assert result is not a
    assert result.cv == ["add(a,b).c0", "add(a,b).c1"]

    inplace = fhe.homo_add_inplace(a, b, ctx)
    assert inplace is a
    assert a.cv == ["add(a,b).c0", "add(a,b).c1"]


def test_homo_sub_does_not_expose_out_and_inplace_mutates_first_arg(monkeypatch):
    def fake_sub(left, right, context):
        return left.cipher_like(
            [f"sub({left.cv[0].split('.', 1)[0]},{right.cv[0].split('.', 1)[0]}).c0", "sub.c1"]
        )

    monkeypatch.setattr(
        arithmetic,
        "_cipher_sub",
        fake_sub,
    )
    monkeypatch.setattr(arithmetic, "_cipher_sub_inplace", lambda left, right, context: left.replace_with(fake_sub(left, right, context)))
    ctx = _manual_fixed_context()
    a = _cipher("a")
    b = _cipher("b")

    with pytest.raises(TypeError):
        fhe.homo_sub(a, b, ctx, out=_cipher("out"))

    result = fhe.homo_sub(a, b, ctx)
    assert result is not a
    assert result.cv == ["sub(a,b).c0", "sub.c1"]
    assert fhe.homo_sub_inplace(a, b, ctx) is a
    assert a.cv == ["sub(a,b).c0", "sub.c1"]


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("cur_limbs", {"cur_limbs": 2}),
        ("scale_degree", {"scale_degree": 2}),
        ("scaling_factor", {"scaling_factor": 3.0}),
    ],
)
@pytest.mark.parametrize("op", [fhe.homo_add_inplace, fhe.homo_sub_inplace])
def test_cipher_cipher_inplace_requires_already_aligned_operands(op, field, kwargs):
    ctx = _manual_fixed_context()
    a = _cipher("a")
    b = _cipher("b", **kwargs)

    with pytest.raises(ValueError, match=f"{field} mismatch"):
        op(a, b, ctx)


def test_plaintext_and_scalar_add_sub_do_not_expose_out_and_inplace_mutates(monkeypatch):
    def fake_add_plain(cipher, plain, context):
        return cipher.cipher_like(["add_pt.c0", cipher.cv[1]])

    def fake_add_scalar(cipher, scalar, context):
        return cipher.cipher_like([f"add_scalar({scalar[0]}).c0", cipher.cv[1]])

    def fake_sub_scalar(cipher, scalar, context):
        return cipher.cipher_like([f"sub_scalar({scalar[0]}).c0", cipher.cv[1]])

    monkeypatch.setattr(
        arithmetic,
        "_cipher_add_plain",
        fake_add_plain,
    )
    monkeypatch.setattr(arithmetic, "_cipher_add_plain_inplace", lambda cipher, plain, context: cipher.replace_with(fake_add_plain(cipher, plain, context)))
    monkeypatch.setattr(
        arithmetic,
        "_cipher_add_scalar",
        fake_add_scalar,
    )
    monkeypatch.setattr(arithmetic, "_cipher_add_scalar_inplace", lambda cipher, scalar, context: cipher.replace_with(fake_add_scalar(cipher, scalar, context)))
    monkeypatch.setattr(
        arithmetic,
        "_cipher_sub_scalar",
        fake_sub_scalar,
    )
    monkeypatch.setattr(arithmetic, "_cipher_sub_scalar_inplace", lambda cipher, scalar, context: cipher.replace_with(fake_sub_scalar(cipher, scalar, context)))

    ctx = SimpleNamespace(moduliQ_scalar=[257, 263, 269], scale_at=lambda _level: 1.0)
    cipher = _cipher("cipher")
    plain = _cipher("plain")

    with pytest.raises(TypeError):
        fhe.homo_add_pt(cipher, plain, ctx, out=_cipher("out"))
    with pytest.raises(TypeError):
        fhe.homo_add_scalar(cipher, 5, ctx, out=_cipher("out"))
    with pytest.raises(TypeError):
        fhe.homo_sub_scalar(cipher, 6, ctx, out=_cipher("out"))

    add_pt = fhe.homo_add_pt(cipher, plain, ctx)
    assert add_pt is not cipher
    assert add_pt.cv == ["add_pt.c0", "cipher.c1"]

    assert fhe.homo_add_pt_inplace(cipher, plain, ctx) is cipher
    assert cipher.cv == ["add_pt.c0", "cipher.c1"]

    add_scalar = fhe.homo_add_scalar(cipher, _scalar([5, 5, 5]), ctx)
    assert add_scalar.cv[0] == "add_scalar(5).c0"

    sub_scalar = fhe.homo_sub_scalar(cipher, _scalar([6, 6, 6]), ctx)
    assert sub_scalar.cv[0] == "sub_scalar(6).c0"

    assert fhe.homo_add_scalar_inplace(cipher, _scalar([7, 7, 7]), ctx) is cipher
    assert cipher.cv[0] == "add_scalar(7).c0"


def test_plaintext_and_scalar_multiply_do_not_expose_out_and_inplace_mutates(monkeypatch):
    def fake_mul_plain(cipher, plain, context):
        return cipher.cipher_like(
            ["mul_pt.c0", "mul_pt.c1"],
            state=CipherState(cipher.state.cur_limbs, 2, 11.0),
        )

    def fake_mul_int(cipher, scalar, context):
        return cipher.cipher_like([f"mul_int({scalar[0]}).c0", cipher.cv[1]])

    def fake_mul_double(cipher, scalar, context, **_kwargs):
        return cipher.cipher_like(
            [f"mul_double({tuple(int(value) for value in scalar)}).c0", cipher.cv[1]],
            state=CipherState(cipher.state.cur_limbs, 2, 13.0),
        )

    monkeypatch.setattr(
        arithmetic,
        "_cipher_mul_plain",
        fake_mul_plain,
    )
    monkeypatch.setattr(arithmetic, "_cipher_mul_plain_inplace", lambda cipher, plain, context: cipher.replace_with(fake_mul_plain(cipher, plain, context)))
    monkeypatch.setattr(
        arithmetic,
        "_cipher_mul_scalar_int",
        fake_mul_int,
    )
    monkeypatch.setattr(arithmetic, "_cipher_mul_scalar_int_inplace", lambda cipher, scalar, context: cipher.replace_with(fake_mul_int(cipher, scalar, context)))
    monkeypatch.setattr(
        arithmetic,
        "_cipher_mul_scalar_double",
        fake_mul_double,
    )
    monkeypatch.setattr(
        arithmetic,
        "_cipher_mul_scalar_double_inplace",
        lambda cipher, scalar, context, **kwargs: cipher.replace_with(
            fake_mul_double(cipher, scalar, context, **kwargs)
        ),
    )

    ctx = SimpleNamespace(moduliQ_scalar=[257, 263, 269], scale_at=lambda _level: 1.0)
    cipher = _cipher("cipher")
    plain = _cipher("plain")

    with pytest.raises(TypeError):
        fhe.homo_mul_pt(cipher, plain, ctx, out=_cipher("out"))
    with pytest.raises(TypeError):
        fhe.homo_mul_scalar(cipher, 3, ctx, out=_cipher("out"))
    with pytest.raises(TypeError):
        fhe.homo_mul_scalar(cipher, 2.0, ctx, out=_cipher("out"))

    mul_pt = fhe.homo_mul_pt(cipher, plain, ctx)
    assert mul_pt is not cipher
    assert mul_pt.cv == ["mul_pt.c0", "mul_pt.c1"]
    assert mul_pt.state.scaling_factor == 11.0
    assert mul_pt.state.scale_degree == 2

    assert fhe.homo_mul_pt_inplace(cipher, plain, ctx) is cipher
    assert cipher.cv == ["mul_pt.c0", "mul_pt.c1"]

    mul_int = fhe.homo_mul_scalar(
        cipher,
        _scalar([3, 3, 3], scaling_factor=1.0, scale_degree=0),
        ctx,
    )
    assert mul_int.cv[0] == "mul_int(3).c0"

    cipher2 = _cipher("cipher2")
    mul_double = fhe.homo_mul_scalar(cipher2, _scalar([2, 2, 2]), ctx)
    assert mul_double.cv[0] == "mul_double((2, 2, 2)).c0"

    assert fhe.homo_mul_scalar_inplace(cipher2, _scalar([4, 4, 4]), ctx) is cipher2
    assert cipher2.cv[0] == "mul_double((4, 4, 4)).c0"


def test_homo_mul_no_relin_exposes_raw_triplet(monkeypatch):
    def fake_mul(left, right, context):
        return left.cipher_like(
            ["raw.c0", "raw.c1", "raw.c2"],
            state=CipherState(left.state.cur_limbs, left.state.scale_degree + right.state.scale_degree, 5.0),
        )

    monkeypatch.setattr(arithmetic, "_cipher_mul", fake_mul)
    ctx = SimpleNamespace(rescale_policy="manual", scale_mode="fixed")
    left = _cipher("left", scale_degree=1)
    right = _cipher("right", scale_degree=2)

    result = fhe.homo_mul_no_relin(left, right, ctx)

    assert result.cv == ["raw.c0", "raw.c1", "raw.c2"]
    assert result.state.scale_degree == 3


def test_homo_mul_no_relin_rejects_triplet_inputs():
    ctx = SimpleNamespace(rescale_policy="manual", scale_mode="fixed")
    left = Cipher(["left.c0", "left.c1", "left.c2"], CipherState(3, 2, 2.0), 8, False)
    right = _cipher("right")

    with pytest.raises(ValueError, match="expected 2 components"):
        fhe.homo_mul_no_relin(left, right, ctx)


def test_homo_mul_no_relin_requires_matching_batch_size():
    ctx = SimpleNamespace(rescale_policy="manual", scale_mode="fixed")
    left = Cipher(["left.c0", "left.c1"], CipherState(3, 1, 2.0), 8, False, batch_size=2)
    right = Cipher(["right.c0", "right.c1"], CipherState(3, 1, 2.0), 8, False, batch_size=3)

    with pytest.raises(ValueError, match="batch_size mismatch"):
        fhe.homo_mul_no_relin(left, right, ctx)


def test_homo_mul_pt_supports_triplets(monkeypatch):
    calls = []

    def fake_cv_mul(component, plain, modulus, mu, cur_limbs, inplace=False):
        calls.append((component, plain, cur_limbs, inplace))
        return f"mul({component})"

    monkeypatch.setattr(primitives.F, "cv_mul", fake_cv_mul)
    ctx = SimpleNamespace(K=0, moduliQ="q", q_mu="mu")
    triplet = Cipher(["c0", "c1", "c2"], CipherState(3, 2, 2.0), 8, False)
    plain = Cipher(["p"], CipherState(3, 1, 2.0), 8, False)

    result = fhe.homo_mul_pt(triplet, plain, ctx)

    assert result.cv == ["mul(c0)", "mul(c1)", "mul(c2)"]
    assert result.state.scale_degree == 3
    assert result.state.scaling_factor == 4.0
    assert calls == [
        ("c0", "p", 3, False),
        ("c1", "p", 3, False),
        ("c2", "p", 3, False),
    ]


def test_homo_mul_pt_inplace_supports_triplets(monkeypatch):
    calls = []

    def fake_cv_mul(component, plain, modulus, mu, cur_limbs, inplace=False):
        calls.append((component, plain, cur_limbs, inplace))
        return component

    monkeypatch.setattr(primitives.F, "cv_mul", fake_cv_mul)
    ctx = SimpleNamespace(K=0, moduliQ="q", q_mu="mu")
    triplet = Cipher(["c0", "c1", "c2"], CipherState(3, 2, 2.0), 8, False)
    plain = Cipher(["p"], CipherState(3, 1, 2.0), 8, False)

    result = fhe.homo_mul_pt_inplace(triplet, plain, ctx)

    assert result is triplet
    assert triplet.cv == ["c0", "c1", "c2"]
    assert triplet.state.scale_degree == 3
    assert calls == [
        ("c0", "p", 3, True),
        ("c1", "p", 3, True),
        ("c2", "p", 3, True),
    ]


def test_scalar_ops_reject_raw_python_scalars():
    ctx = SimpleNamespace()
    cipher = _cipher("cipher")

    for op, scalar in (
        (fhe.homo_add_scalar, 1.5),
        (fhe.homo_add_scalar, 5),
        (fhe.homo_sub_scalar, 5),
        (fhe.homo_mul_scalar, 1.5),
        (fhe.homo_mul_scalar, 5),
    ):
        with pytest.raises(TypeError, match="expected EncodedScalar"):
            op(cipher, scalar, ctx)


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
        torch.zeros((1, 3, 4), dtype=torch.uint64),
        torch.zeros((1, 3, 4), dtype=torch.uint64),
        3,
        3,
        "swk_bx",
        "swk_ax",
        "precompute",
        ctx,
    )

    assert result == ("rot.c0", "rot.c1")
    assert seen == {"op": "hrot", "curr_limbs": 3}


def test_cv_add_allocates_result_view(monkeypatch):
    seen = {}

    def fake_add_mod(x, y, modulus, *, cur_limbs, out=None):
        seen["x_shape"] = tuple(x.shape)
        seen["out"] = out
        seen["cur_limbs"] = cur_limbs
        return torch.empty_like(x)

    monkeypatch.setattr(kernels.torch, "add_mod", fake_add_mod)
    x = torch.zeros((1, 3, 4), dtype=torch.uint64)
    y = torch.zeros((1, 3, 4), dtype=torch.uint64)

    result = kernels.cv_add(x, y, torch.ones((3,), dtype=torch.uint64), 3)
    assert tuple(result.shape) == (1, 3, 4)
    assert seen == {"x_shape": (1, 3, 4), "out": None, "cur_limbs": 3}


def test_cipher_add_inplace_uses_component_inplace(monkeypatch):
    seen = []

    def fake_cv_add(left, right, modulus, cur_limbs, inplace=False):
        seen.append((left, right, inplace))
        return left

    monkeypatch.setattr(primitives.F, "cv_add", fake_cv_add)
    ctx = SimpleNamespace(rescale_policy="manual", scale_mode="fixed", moduliQ="q")
    a = Cipher(
        [torch.zeros((1, 3, 4), dtype=torch.uint64), torch.ones((1, 3, 4), dtype=torch.uint64)],
        CipherState(3, 1, 2.0),
        8,
        False,
    )
    b = Cipher(
        [torch.full((1, 3, 4), 2, dtype=torch.uint64), torch.full((1, 3, 4), 3, dtype=torch.uint64)],
        CipherState(3, 1, 2.0),
        8,
        False,
    )

    assert fhe.homo_add_inplace(a, b, ctx) is a
    assert len(seen) == 2
    assert seen[0][0] is a.cv[0]
    assert seen[1][0] is a.cv[1]
    assert seen[0][2] is True


def test_cipher_add_inplace_uses_cv_add_pair_when_available(monkeypatch):
    calls = {}

    def fake_cv_add_pair_(in0_c0, in0_c1, in1_c0, in1_c1, modulus, cur_limbs):
        calls["c0"] = in0_c0
        calls["c1"] = in0_c1
        calls["cur_limbs"] = cur_limbs
        return (in0_c0, in0_c1)

    monkeypatch.setattr(primitives, "_all_cuda", lambda *tensors: True)
    monkeypatch.setattr(primitives.F, "cv_add_pair_", fake_cv_add_pair_)
    ctx = SimpleNamespace(rescale_policy="manual", scale_mode="fixed", moduliQ="q")
    a = Cipher([torch.zeros((1, 3, 4), dtype=torch.uint64), torch.zeros((1, 3, 4), dtype=torch.uint64)], CipherState(3, 1, 2.0), 8, False)
    b = Cipher([torch.zeros((1, 3, 4), dtype=torch.uint64), torch.zeros((1, 3, 4), dtype=torch.uint64)], CipherState(3, 1, 2.0), 8, False)

    assert fhe.homo_add_inplace(a, b, ctx) is a
    assert calls == {"c0": a.cv[0], "c1": a.cv[1], "cur_limbs": 3}
    assert a.state.scaling_factor == 2.0
    assert a.slots == 8


def test_ext_cipher_add_inplace_uses_cv_add_pair(monkeypatch):
    calls = {}

    def fake_cv_add_pair_(in0_c0, in0_c1, in1_c0, in1_c1, modulus, cur_limbs):
        calls["modulus"] = modulus
        calls["cur_limbs"] = cur_limbs
        calls["c0"] = in0_c0
        calls["c1"] = in0_c1
        return (in0_c0, in0_c1)

    monkeypatch.setattr(primitives, "_all_cuda", lambda *tensors: True)
    monkeypatch.setattr(primitives.F, "cv_add_pair_", fake_cv_add_pair_)
    qp = torch.ones((5,), dtype=torch.uint64)
    ctx = SimpleNamespace(
        rescale_policy="manual",
        scale_mode="fixed",
        K=2,
        QplusP_map={3: qp},
        moduliQ=torch.ones((3,), dtype=torch.uint64),
    )
    a = Cipher([torch.zeros((1, 5, 4), dtype=torch.uint64), torch.zeros((1, 5, 4), dtype=torch.uint64)], CipherState(3, 1, 2.0), 8, True)
    b = Cipher([torch.zeros((1, 5, 4), dtype=torch.uint64), torch.zeros((1, 5, 4), dtype=torch.uint64)], CipherState(3, 1, 2.0), 8, True)

    assert fhe.homo_add_inplace(a, b, ctx) is a
    assert calls["modulus"] is qp
    assert calls["cur_limbs"] == 5
    assert calls["c0"] is a.cv[0]
    assert calls["c1"] is a.cv[1]


def test_ext_cipher_sub_uses_cv_sub_pair(monkeypatch):
    calls = {}

    def fake_cv_sub_pair(in0_c0, in0_c1, in1_c0, in1_c1, modulus, cur_limbs):
        calls["modulus"] = modulus
        calls["cur_limbs"] = cur_limbs
        calls["lhs"] = (in0_c0, in0_c1)
        calls["rhs"] = (in1_c0, in1_c1)
        return ("sub.c0", "sub.c1")

    monkeypatch.setattr(primitives, "_all_cuda", lambda *tensors: True)
    monkeypatch.setattr(primitives.F, "cv_sub_pair", fake_cv_sub_pair)
    qp = torch.ones((5,), dtype=torch.uint64)
    ctx = SimpleNamespace(
        rescale_policy="manual",
        scale_mode="fixed",
        K=2,
        QplusP_map={3: qp},
        moduliQ=torch.ones((3,), dtype=torch.uint64),
    )
    a = Cipher([torch.zeros((1, 5, 4), dtype=torch.uint64), torch.zeros((1, 5, 4), dtype=torch.uint64)], CipherState(3, 1, 2.0), 8, True)
    b = Cipher([torch.zeros((1, 5, 4), dtype=torch.uint64), torch.zeros((1, 5, 4), dtype=torch.uint64)], CipherState(3, 1, 2.0), 8, True)

    result = fhe.homo_sub(a, b, ctx)

    assert result.cv == ["sub.c0", "sub.c1"]
    assert calls["modulus"] is qp
    assert calls["cur_limbs"] == 5
    assert calls["lhs"][0] is a.cv[0]
    assert calls["lhs"][1] is a.cv[1]
    assert calls["rhs"][0] is b.cv[0]
    assert calls["rhs"][1] is b.cv[1]


def test_ext_cipher_mul_plain_uses_cv_mul_pt_pair(monkeypatch):
    calls = {}

    def fake_cv_mul_pt_pair(c0, c1, plain, modulus, barret_mu, cur_limbs):
        calls["modulus"] = modulus
        calls["barret_mu"] = barret_mu
        calls["cur_limbs"] = cur_limbs
        return ("out.c0", "out.c1")

    monkeypatch.setattr(primitives, "_all_cuda", lambda *tensors: True)
    monkeypatch.setattr(primitives.F, "cv_mul_pt_pair", fake_cv_mul_pt_pair)
    qp = torch.ones((5,), dtype=torch.uint64)
    qp_mu = torch.ones((5, 2), dtype=torch.uint64)
    ctx = SimpleNamespace(
        K=2,
        QplusP_map={3: qp},
        QmuplusPmu_map={3: qp_mu},
        moduliQ=torch.ones((3,), dtype=torch.uint64),
        q_mu=torch.ones((3, 2), dtype=torch.uint64),
    )
    cipher = Cipher([torch.zeros((1, 5, 4), dtype=torch.uint64), torch.zeros((1, 5, 4), dtype=torch.uint64)], CipherState(3, 1, 2.0), 8, True)
    plain = Cipher([torch.zeros((1, 5, 4), dtype=torch.uint64)], CipherState(3, 1, 2.0), 8, True)

    result = fhe.homo_mul_pt(cipher, plain, ctx)

    assert result.cv == ["out.c0", "out.c1"]
    assert result.state.scaling_factor == 4.0
    assert calls["modulus"] is qp
    assert calls["barret_mu"] is qp_mu
    assert calls["cur_limbs"] == 5


def test_mul_relin_rescale_postop_does_not_expose_out(monkeypatch):
    ctx = _mul_context()
    a = _cipher("a", cur_limbs=4)
    b = _cipher("b", cur_limbs=4)

    monkeypatch.setattr(
        arithmetic.F,
        "cv_hmul_relin_rescale",
        lambda *args, **kwargs: (
            torch.zeros((1, 3, 4), dtype=torch.uint64),
            torch.zeros((1, 3, 4), dtype=torch.uint64),
        ),
    )

    with pytest.raises(TypeError):
        fhe.homo_mul_relin_rescale_postop(a, b, ctx, out=_cipher("out"))

    result = fhe.homo_mul_relin_rescale_postop(a, b, ctx)
    assert result is not a
    assert result.state.cur_limbs == 3
    assert result.state.scale_degree == 1
    assert result.state.scaling_factor == pytest.approx(0.4)
    assert len(result.cv) == 2


@pytest.mark.parametrize(
    "name",
    [
        "homo_add_inplace",
        "homo_sub_inplace",
        "homo_add_pt_inplace",
        "homo_mul_pt_inplace",
        "homo_add_scalar",
        "homo_add_scalar_inplace",
        "homo_mul_scalar",
        "homo_mul_scalar_inplace",
        "homo_sub_scalar",
        "homo_sub_scalar_inplace",
        "normalize_scale",
    ],
)
def test_inplace_symbols_are_public(name):
    assert hasattr(fhe, name)
