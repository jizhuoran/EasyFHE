from types import SimpleNamespace

import easyfhe as torch
import pytest

import easyfhe.fhe as fhe
from easyfhe.fhe.ciphertext import Cipher
from easyfhe.fhe.ops import alignment, arithmetic, kernels, rotation


def _uint64_tensor(shape, fill=0):
    return torch.full(shape, fill, dtype=torch.int64).to(torch.uint64)


def _capacity_tensor(active, *, capacity, inactive_fill):
    out = _uint64_tensor((capacity, active.shape[-1]), fill=inactive_fill)
    out[: active.shape[0]] = active
    return out


def _cipher(name, *, cur_limbs=3, capacity=5, noise_deg=1):
    c0 = _uint64_tensor((capacity, 64), fill=1)
    c1 = _uint64_tensor((capacity, 64), fill=2)
    cipher = Cipher(
        [c0, c1],
        cur_limbs=cur_limbs,
        scaling_factor=2.0,
        noise_deg=noise_deg,
        slots=8,
        is_ext=False,
    )
    cipher.name = name
    return cipher


def _cipher_from_active(name, c0, c1, *, capacity, inactive_fill, cur_limbs=None):
    cur_limbs = c0.shape[0] if cur_limbs is None else int(cur_limbs)
    cipher = Cipher(
        [
            _capacity_tensor(c0, capacity=capacity, inactive_fill=inactive_fill),
            _capacity_tensor(c1, capacity=capacity, inactive_fill=inactive_fill + 1),
        ],
        cur_limbs=cur_limbs,
        scaling_factor=2.0,
        noise_deg=1,
        slots=8,
        is_ext=False,
    )
    cipher.name = name
    return cipher


def _plaintext_from_active(name, values, *, capacity, inactive_fill, cur_limbs=None):
    cur_limbs = values.shape[0] if cur_limbs is None else int(cur_limbs)
    plain = Cipher(
        [_capacity_tensor(values, capacity=capacity, inactive_fill=inactive_fill)],
        cur_limbs=cur_limbs,
        scaling_factor=2.0,
        noise_deg=1,
        slots=8,
        is_ext=False,
    )
    plain.name = name
    return plain


def _context():
    moduli = _uint64_tensor((6,), fill=97)
    mu = _uint64_tensor((6, 2), fill=1)
    return SimpleNamespace(
        rescale_policy="manual",
        scale_mode="fixed",
        K=0,
        moduliQ=moduli,
        q_mu=mu,
        QplusP_map={index: moduli for index in range(1, 7)},
        QmuplusPmu_map={index: mu for index in range(1, 7)},
        moduliQ_scalar=[97] * 6,
        scale_at=lambda _cur_limbs: 2.0,
        rescale_divisor_at=lambda _drop_limb: 2.0,
    )


@pytest.mark.parametrize(
    ("op_name", "args"),
    [
        ("cv_neg", ()),
        ("cv_add", ("other",)),
        ("cv_sub", ("other",)),
        ("cv_mul", ("other", "barret")),
        ("cv_add_scalar", ("scalar",)),
        ("cv_sub_scalar", ("scalar",)),
        ("cv_mul_scalar", ("scalar", "barret")),
    ],
)
def test_cv_ops_use_cur_limbs_not_tensor_capacity(op_name, args):
    active_x = _uint64_tensor((3, 64), fill=5)
    active_y = _uint64_tensor((3, 64), fill=7)
    x_compact = active_x.clone()
    y_compact = active_y.clone()
    x_capacity = _capacity_tensor(active_x, capacity=5, inactive_fill=91)
    y_capacity = _capacity_tensor(active_y, capacity=4, inactive_fill=89)
    modulus = _uint64_tensor((5,), fill=97)
    barret = _uint64_tensor((5, 2), fill=1)
    scalar = _uint64_tensor((3,), fill=11)

    compact_call_args = []
    capacity_call_args = []
    for arg in args:
        if arg == "other":
            compact_call_args.append(y_compact)
            capacity_call_args.append(y_capacity)
        elif arg == "barret":
            compact_call_args.append(barret)
            capacity_call_args.append(barret)
        elif arg == "scalar":
            compact_call_args.append(scalar)
            capacity_call_args.append(scalar)
        else:
            raise AssertionError(arg)

    op = getattr(kernels, op_name)
    if op_name in ("cv_mul", "cv_mul_scalar"):
        compact = op(x_compact, *compact_call_args[:-1], modulus, compact_call_args[-1], 3)
        capacity = op(x_capacity, *capacity_call_args[:-1], modulus, capacity_call_args[-1], 3)
    else:
        compact = op(x_compact, *compact_call_args, modulus, 3)
        capacity = op(x_capacity, *capacity_call_args, modulus, 3)

    assert tuple(capacity.shape) == tuple(x_capacity.shape)
    assert capacity[:3].cpu().numpy().tolist() == compact[:3].cpu().numpy().tolist()


def test_cv_add_inplace_uses_cur_limbs_and_preserves_capacity():
    x = _uint64_tensor((5, 64), fill=1)
    y = _uint64_tensor((4, 64), fill=2)
    modulus = _uint64_tensor((5,), fill=97)

    result = kernels.cv_add(x, y, modulus, cur_limbs=3, inplace=True)

    assert result is x
    assert tuple(x.shape) == (5, 64)
    assert (x[:3].cpu().numpy() == 3).all()


def test_align_drop_limb_updates_metadata_without_compressing_non_ext_tensor():
    ctx = _context()
    cipher = _cipher("a", cur_limbs=4, capacity=6)

    result = alignment.align_to(cipher, alignment.CipherState(2, 1), ctx)

    assert result.cur_limbs == 2
    assert tuple(result.cv[0].shape) == (6, 64)
    assert result.cv[0] is cipher.cv[0]


def test_ext_cipher_cannot_drop_limbs_or_rescale_via_alignment():
    ctx = _context()
    cipher = _cipher("ext", cur_limbs=4, capacity=6, noise_deg=2)
    cipher.is_ext = True

    with pytest.raises(ValueError, match="moddowned before dropping limbs"):
        alignment.align_to(cipher, alignment.CipherState(3, 2), ctx)

    with pytest.raises(ValueError, match="moddowned before rescale"):
        alignment.rescale_one_level(cipher, ctx)


def test_rescale_updates_metadata_and_preserves_component_capacity(monkeypatch):
    ctx = _context()
    cipher = _cipher("a", cur_limbs=4, capacity=6, noise_deg=2)

    def fake_drop_last(component, cur_limbs, level, context):
        assert cur_limbs == 4
        return _uint64_tensor((3, 64), fill=11)

    monkeypatch.setattr(alignment.F, "cv_drop_last_element_and_scale", fake_drop_last)

    result = alignment.rescale_one_level(cipher, ctx)

    assert result.cur_limbs == 3
    assert result.noise_deg == 1
    assert tuple(result.cv[0].shape) == (6, 64)
    assert (result.cv[0][:3].cpu().numpy() == 11).all()


def test_homo_add_preserves_left_operand_capacity():
    ctx = _context()
    a = _cipher("a", cur_limbs=3, capacity=5)
    b = _cipher("b", cur_limbs=3, capacity=4)

    result = fhe.homo_add(a, b, ctx)

    assert result.cur_limbs == 3
    assert tuple(result.cv[0].shape) == (5, 64)
    assert result is not a


def test_homo_add_ignores_inactive_capacity_values():
    ctx = _context()
    active_a0 = _uint64_tensor((3, 64), fill=5)
    active_a1 = _uint64_tensor((3, 64), fill=6)
    active_b0 = _uint64_tensor((3, 64), fill=7)
    active_b1 = _uint64_tensor((3, 64), fill=8)
    compact_a = _cipher_from_active("compact_a", active_a0, active_a1, capacity=3, inactive_fill=0)
    compact_b = _cipher_from_active("compact_b", active_b0, active_b1, capacity=3, inactive_fill=0)
    capacity_a = _cipher_from_active("capacity_a", active_a0, active_a1, capacity=6, inactive_fill=91)
    capacity_b = _cipher_from_active("capacity_b", active_b0, active_b1, capacity=5, inactive_fill=89)

    compact = fhe.homo_add(compact_a, compact_b, ctx)
    capacity = fhe.homo_add(capacity_a, capacity_b, ctx)

    assert tuple(capacity.cv[0].shape) == (6, 64)
    assert capacity.cv[0][:3].cpu().numpy().tolist() == compact.cv[0][:3].cpu().numpy().tolist()
    assert capacity.cv[1][:3].cpu().numpy().tolist() == compact.cv[1][:3].cpu().numpy().tolist()


@pytest.mark.parametrize(
    "op",
    [
        "homo_add_scalar_int",
        "homo_sub_scalar_int",
        "homo_mul_scalar_int",
        "homo_mul_scalar_double",
    ],
)
def test_public_cipher_scalar_ops_ignore_inactive_capacity_values(op):
    ctx = _context()
    active_a0 = _uint64_tensor((3, 64), fill=5)
    active_a1 = _uint64_tensor((3, 64), fill=6)
    compact = _cipher_from_active("compact", active_a0, active_a1, capacity=3, inactive_fill=0)
    capacity = _cipher_from_active("capacity", active_a0, active_a1, capacity=6, inactive_fill=91)
    scalar = 3 if op != "homo_mul_scalar_double" else 1.5

    compact_result = getattr(fhe, op)(compact, scalar, ctx)
    capacity_result = getattr(fhe, op)(capacity, scalar, ctx)

    assert tuple(capacity_result.cv[0].shape) == (6, 64)
    assert capacity_result.cv[0][:3].cpu().numpy().tolist() == compact_result.cv[0][:3].cpu().numpy().tolist()
    assert capacity_result.cv[1][:3].cpu().numpy().tolist() == compact_result.cv[1][:3].cpu().numpy().tolist()


def test_homo_sub_ignores_inactive_capacity_values():
    ctx = _context()
    active_a0 = _uint64_tensor((3, 64), fill=9)
    active_a1 = _uint64_tensor((3, 64), fill=10)
    active_b0 = _uint64_tensor((3, 64), fill=2)
    active_b1 = _uint64_tensor((3, 64), fill=3)
    compact_a = _cipher_from_active("compact_a", active_a0, active_a1, capacity=3, inactive_fill=0)
    compact_b = _cipher_from_active("compact_b", active_b0, active_b1, capacity=3, inactive_fill=0)
    capacity_a = _cipher_from_active("capacity_a", active_a0, active_a1, capacity=6, inactive_fill=91)
    capacity_b = _cipher_from_active("capacity_b", active_b0, active_b1, capacity=5, inactive_fill=89)

    compact = fhe.homo_sub(compact_a, compact_b, ctx)
    capacity = fhe.homo_sub(capacity_a, capacity_b, ctx)

    assert tuple(capacity.cv[0].shape) == (6, 64)
    assert capacity.cv[0][:3].cpu().numpy().tolist() == compact.cv[0][:3].cpu().numpy().tolist()
    assert capacity.cv[1][:3].cpu().numpy().tolist() == compact.cv[1][:3].cpu().numpy().tolist()


@pytest.mark.parametrize("op", ["homo_add_pt", "homo_mul_pt"])
def test_public_cipher_plaintext_ops_ignore_inactive_capacity_values(op):
    ctx = _context()
    active_c0 = _uint64_tensor((3, 64), fill=5)
    active_c1 = _uint64_tensor((3, 64), fill=6)
    active_p = _uint64_tensor((3, 64), fill=7)
    compact_cipher = _cipher_from_active("compact_cipher", active_c0, active_c1, capacity=3, inactive_fill=0)
    compact_plain = _plaintext_from_active("compact_plain", active_p, capacity=3, inactive_fill=0)
    capacity_cipher = _cipher_from_active("capacity_cipher", active_c0, active_c1, capacity=6, inactive_fill=91)
    capacity_plain = _plaintext_from_active("capacity_plain", active_p, capacity=5, inactive_fill=89)

    compact = getattr(fhe, op)(compact_cipher, compact_plain, ctx)
    capacity = getattr(fhe, op)(capacity_cipher, capacity_plain, ctx)

    assert tuple(capacity.cv[0].shape) == (6, 64)
    assert capacity.cv[0][:3].cpu().numpy().tolist() == compact.cv[0][:3].cpu().numpy().tolist()
    assert capacity.cv[1][:3].cpu().numpy().tolist() == compact.cv[1][:3].cpu().numpy().tolist()


def test_homo_add_inplace_keeps_existing_capacity():
    ctx = _context()
    a = _cipher("a", cur_limbs=3, capacity=5)
    b = _cipher("b", cur_limbs=3, capacity=4)

    result = fhe.homo_add_inplace(a, b, ctx)

    assert result is a
    assert a.cur_limbs == 3
    assert tuple(a.cv[0].shape) == (5, 64)


def test_homo_mul_and_square_preserve_capacity_before_relinearize(monkeypatch):
    ctx = _context()
    active_a0 = _uint64_tensor((3, 64), fill=5)
    active_a1 = _uint64_tensor((3, 64), fill=6)
    active_b0 = _uint64_tensor((3, 64), fill=7)
    active_b1 = _uint64_tensor((3, 64), fill=8)
    compact_a = _cipher_from_active("compact_a", active_a0, active_a1, capacity=3, inactive_fill=0)
    compact_b = _cipher_from_active("compact_b", active_b0, active_b1, capacity=3, inactive_fill=0)
    capacity_a = _cipher_from_active("capacity_a", active_a0, active_a1, capacity=6, inactive_fill=91)
    capacity_b = _cipher_from_active("capacity_b", active_b0, active_b1, capacity=5, inactive_fill=89)
    monkeypatch.setattr(arithmetic, "_relinearize", lambda cipher, context: cipher)

    compact_mul = fhe.homo_mul(compact_a, compact_b, ctx)
    capacity_mul = fhe.homo_mul(capacity_a, capacity_b, ctx)
    compact_square = arithmetic.homo_square(compact_a, ctx)
    capacity_square = arithmetic.homo_square(capacity_a, ctx)

    assert tuple(capacity_mul.cv[0].shape) == (6, 64)
    assert capacity_mul.cv[0][:3].cpu().numpy().tolist() == compact_mul.cv[0][:3].cpu().numpy().tolist()
    assert tuple(capacity_square.cv[0].shape) == (6, 64)
    assert capacity_square.cv[0][:3].cpu().numpy().tolist() == compact_square.cv[0][:3].cpu().numpy().tolist()


def test_homo_mul_rescale_preserves_input_capacity(monkeypatch):
    ctx = _context()
    ctx.L = 4
    ctx.mult_swk_bx = "bx"
    ctx.mult_swk_ax = "ax"
    active_a0 = _uint64_tensor((4, 64), fill=5)
    active_a1 = _uint64_tensor((4, 64), fill=6)
    active_b0 = _uint64_tensor((4, 64), fill=7)
    active_b1 = _uint64_tensor((4, 64), fill=8)
    a = _cipher_from_active("a", active_a0, active_a1, capacity=6, inactive_fill=91, cur_limbs=4)
    b = _cipher_from_active("b", active_b0, active_b1, capacity=5, inactive_fill=89, cur_limbs=4)

    def fake_hmul(*args, **kwargs):
        return torch.stack([
            torch.stack([_uint64_tensor((3, 64), fill=11)]),
            torch.stack([_uint64_tensor((3, 64), fill=12)]),
        ])

    monkeypatch.setattr(arithmetic.F, "cv_hmul_double_rescale", fake_hmul)

    result = fhe.homo_mul_rescale(a, b, ctx)

    assert result.cur_limbs == 3
    assert tuple(result.cv[0].shape) == (6, 64)
    assert (result.cv[0][:3].cpu().numpy() == 11).all()


def test_homo_rotate_preserves_capacity_and_uses_cur_limbs(monkeypatch):
    ctx = SimpleNamespace(
        L=4,
        N=64,
        rotation_key_limb_limits={},
        get_rotation_key=lambda _index: ("bx", "ax"),
        get_inverse_precompute_auto=lambda _index: "inverse",
    )
    active_a0 = _uint64_tensor((3, 64), fill=5)
    active_a1 = _uint64_tensor((3, 64), fill=6)
    cipher = _cipher_from_active("capacity", active_a0, active_a1, capacity=6, inactive_fill=91)

    def fake_hrot(c0, c1, cur_limbs, *args, **kwargs):
        assert cur_limbs == 3
        return _uint64_tensor((cur_limbs, 64), fill=6), _uint64_tensor((cur_limbs, 64), fill=7)

    monkeypatch.setattr(rotation.F, "cv_hrot", fake_hrot)

    result = fhe.homo_rotate(cipher, 1, ctx)

    assert tuple(result.cv[0].shape) == (6, 64)
    assert (result.cv[0][:3].cpu().numpy() == 6).all()


def test_scalar_weighted_acc_preserves_capacity_and_ignores_inactive_limbs():
    _, ctx = fhe.generate_client_context(
        fhe.CKKSContextSpec(depth=3, log_n=6, dnum=1, dcrt_bits=30, first_mod=35),
        device="cpu",
    )
    batch_size = 2
    cur_limbs = 3
    capacity = 5
    active = torch.ones((batch_size, cur_limbs, ctx.N), dtype=torch.uint64)
    cv = [
        torch.empty((batch_size, capacity, ctx.N), dtype=torch.uint64),
        torch.empty((batch_size, capacity, ctx.N), dtype=torch.uint64),
    ]
    for component in cv:
        component[:, :cur_limbs] = active
        component[:, cur_limbs:] = _uint64_tensor((batch_size, capacity - cur_limbs, ctx.N), fill=91)
    scalars = torch.ones((batch_size, cur_limbs), dtype=torch.uint64)
    cipher = Cipher(cv, cur_limbs, 1.0, 1, ctx.N // 2, False, batch_size=batch_size)

    result = kernels.cipher_scalar_weighted_acc(cipher, scalars, ctx)

    assert tuple(result[0].shape) == (capacity, ctx.N)
