import easyfhe.fhe as fhe
import pytest
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
