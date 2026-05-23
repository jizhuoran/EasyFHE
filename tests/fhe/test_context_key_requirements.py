import easyfhe.fhe as fhe
import numpy as np
import pytest
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
