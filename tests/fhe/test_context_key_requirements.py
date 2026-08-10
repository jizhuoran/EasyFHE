import easyfhe.fhe as fhe
import numpy as np
import pytest
from types import SimpleNamespace
from easyfhe.fhe._keygen.context_material_builder import ContextMaterialBuilder
from easyfhe.fhe._keygen.native_sampler import _find_auto_index_2n_complex


def test_context_has_no_runtime_keygen_or_client_crypto_methods():
    assert not hasattr(fhe.Context, "add_keys")
    assert not hasattr(fhe.Context, "addkeys")
    assert not hasattr(fhe.Context, "ensure_rotation_keys")
    assert not hasattr(fhe.Context, "encrypt")
    assert not hasattr(fhe.Context, "decrypt")

def test_rotation_offsets_map_to_auto_indices():
    cycl_order = 1 << 11
    assert {
        _find_auto_index_2n_complex(1, cycl_order): 1,
        _find_auto_index_2n_complex(-2, cycl_order): -2,
    }


def test_ckks_context_spec_uses_explicit_scale_options():
    spec = fhe.CKKSContextSpec(
        depth=3,
        log_n=5,
        dnum=1,
        dcrt_bits=30,
        first_mod=35,
        scale_mode="fixed",
        rescale_policy="auto",
    )
    assert spec.scale_mode == "fixed"
    assert spec.rescale_policy == "auto"


def test_ckks_context_spec_accepts_flexible_scale_mode():
    spec = fhe.CKKSContextSpec(
        depth=3,
        log_n=5,
        dnum=1,
        dcrt_bits=30,
        first_mod=35,
        scale_mode="flexible",
        rescale_policy="auto",
    )
    assert spec.scale_mode == "flexible"


def test_ckks_context_spec_accepts_explicit_u64_limb_specs():
    spec = fhe.CKKSContextSpec(
        log_n=5,
        dnum=1,
        limb_specs=(35, 30, 30, 30),
        scale_mode="fixed",
    )

    assert spec.depth == 3
    assert spec.first_mod == 35
    assert spec.dcrt_bits == 30
    assert spec.q_prime_bits == (35, 30, 30, 30)


def test_u64_prime_chain_plan_has_single_limb_rescale_semantics():
    plan = fhe.plan_prime_chain(limb_specs=(60, 50, 49))

    assert plan.dcrt_bits == (60, 50, 49)
    assert plan.depth == 2
    assert plan.physical_limb_count == 3
    assert plan.first_mod_limb_count == 1
    assert plan.rescale_limb_count == 1


def test_u64_limb_specs_reject_composite_and_fixed_heterogeneous_chains():
    with pytest.raises(ValueError, match="scalar prime bit-sizes"):
        fhe.plan_prime_chain(limb_specs=((60, 30, 30), 30))

    with pytest.raises(ValueError, match="fixed scale_mode"):
        fhe.CKKSContextSpec(log_n=5, dnum=1, limb_specs=(35, 29, 30))

    flexible = fhe.CKKSContextSpec(
        log_n=5,
        dnum=1,
        limb_specs=(35, 29, 30),
        scale_mode="flexible",
    )
    assert flexible.q_prime_bits == (35, 29, 30)


def test_ckks_context_spec_rejects_flex_alias():
    with pytest.raises(ValueError, match="fixed.*flexible"):
        fhe.CKKSContextSpec(
            depth=3,
            log_n=5,
            dnum=1,
            dcrt_bits=30,
            first_mod=35,
            scale_mode="flex",
            rescale_policy="auto",
        )


def test_ckks_context_spec_rejects_legacy_combined_scale_option():
    legacy_kwargs = {"rescale" + "_tech": "FIXED" + "MANUAL"}
    with pytest.raises(TypeError):
        fhe.CKKSContextSpec(
            depth=3,
            log_n=5,
            dnum=1,
            dcrt_bits=30,
            first_mod=35,
            **legacy_kwargs,
        )


def test_flexible_context_scale_tables_are_level_aware():
    builder = ContextMaterialBuilder.from_public_params(
        log_n=5,
        depth=3,
        dcrt_bits=8,
        special_mod=9,
        dnum=1,
        secret_key_dist="SPARSE_TERNARY",
        scale_mode="flexible",
        rescale_policy="auto",
        moduli_q=np.array([257, 263, 269], dtype=np.uint64),
        roots_q=np.array([1, 1, 1], dtype=np.uint64),
        moduli_p=np.array([271], dtype=np.uint64),
        roots_p=np.array([1], dtype=np.uint64),
        eval_mult_key=np.zeros((2, 1, 4, 32), dtype=np.uint64),
    )
    context = fhe.Context(builder.to_runtime_material(), "cpu")

    assert context.scale_mode == "flexible"
    assert context.scale_at(3) == pytest.approx(269.0)
    assert context.scale_at(1) == pytest.approx(269.0 * 269.0 / 263.0)
    assert context.big_scale_at(3) == pytest.approx(269.0 * 269.0)
    assert context.rescale_divisor_at(2) == pytest.approx(269.0)


def test_rotation_key_cuda_cache_uses_exact_limb_shape():
    loads = []
    context = SimpleNamespace(
        device="cuda",
        auto_load_keys_resolved=True,
        alpha=2,
        _rotation_key_cuda_cache={},
        _rotation_key_special_mod_start=lambda _rot_index: 5,
        _load_rotation_key_to_cuda=lambda rot_index, cur_limbs, beta, available: (
            loads.append((rot_index, cur_limbs, beta, available)) or f"bx-{cur_limbs}",
            f"ax-{cur_limbs}",
        ),
    )

    key4, special4 = fhe.Context.get_rotation_key_for_limbs(context, 3, 4)
    key3, special3 = fhe.Context.get_rotation_key_for_limbs(context, 3, 3)
    cached4, cached_special4 = fhe.Context.get_rotation_key_for_limbs(context, 3, 4)

    assert loads == [(3, 4, 2, 5), (3, 3, 2, 5)]
    assert key4 == cached4 == ["bx-4", "ax-4"]
    assert key3 == ["bx-3", "ax-3"]
    assert (special4, special3, cached_special4) == (4, 3, 4)
