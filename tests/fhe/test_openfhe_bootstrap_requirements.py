from types import SimpleNamespace

import pytest

import easyfhe.bs.openfhe as bs
from easyfhe.fhe.ops import rotation


def test_bootstrap_spec_derives_context_depth_and_rotations():
    spec = bs.BootstrapSpec(
        log_slots=14,
        level_budget=(4, 4),
        output_levels=12,
    )

    requirements = bs.requirements(spec, log_n=16)

    assert requirements.bootstrap_depth == 18
    assert requirements.context_depth == 30
    assert requirements.rotations
    assert len(requirements.rotations) == len(set(requirements.rotations))
    assert requirements.rotations[-1] == (1 << 17) - 1


def test_bootstrap_requirements_merge_multiple_specs():
    specs = (
        bs.BootstrapSpec(log_slots=14, level_budget=(4, 4), output_levels=4),
        bs.BootstrapSpec(
            log_slots=15,
            level_budget=(3, 3),
            output_levels=8,
            strategy="normal_bsgs",
        ),
    )

    requirements = bs.requirements(specs, log_n=16)

    expected_depths = (18 + 4, 16 + 8)
    assert requirements.bootstrap_depth == 18
    assert requirements.context_depth == max(expected_depths)
    assert len(requirements.rotations) == len(set(requirements.rotations))


def test_bootstrap_requirements_include_explicit_raise_target_capacity():
    spec = bs.BootstrapSpec(
        log_slots=14,
        level_budget=(4, 4),
        output_levels=2,
        raise_to_limbs=24,
    )

    requirements = bs.requirements(spec, log_n=16)

    assert requirements.bootstrap_depth == 18
    assert requirements.context_depth == 23


def test_bootstrap_requirements_reject_too_small_raise_target():
    spec = bs.BootstrapSpec(
        log_slots=14,
        level_budget=(4, 4),
        output_levels=2,
        raise_to_limbs=20,
    )

    with pytest.raises(ValueError, match="raise_to_limbs is too small"):
        bs.requirements(spec, log_n=16)


@pytest.mark.parametrize("mode", ["classic", "slots_first", "mod_raise_first"])
def test_bootstrap_spec_rejects_mode_aliases(mode):
    with pytest.raises(ValueError, match="bootstrap mode"):
        bs.BootstrapSpec(
            log_slots=14,
            level_budget=(4, 4),
            output_levels=2,
            mode=mode,
        )


@pytest.mark.parametrize("secret_key_dist", ["SPARSE_TERNARY", "UNIFORM_TERNARY"])
@pytest.mark.parametrize("level_budget", [(1, 1), (1, 3), (4, 1)])
def test_openfhe_bootstrap_rejects_linear_transform_budget(secret_key_dist, level_budget):
    del secret_key_dist
    with pytest.raises(NotImplementedError, match="linear-transform route"):
        bs.BootstrapSpec(
            log_slots=14,
            level_budget=level_budget,
            output_levels=2,
        )


@pytest.mark.parametrize(
    ("secret_key_dist", "expected_depth"),
    [
        ("SPARSE_TERNARY", 18),
        ("UNIFORM_TERNARY", 21),
    ],
)
def test_openfhe_bootstrap_depth_supports_sparse_and_uniform(secret_key_dist, expected_depth):
    spec = bs.BootstrapSpec(
        log_slots=14,
        level_budget=(4, 4),
        output_levels=0,
    )
    requirements = bs.requirements(spec, log_n=16, secret_key_dist=secret_key_dist)

    assert requirements.bootstrap_depth == expected_depth
    assert requirements.context_depth == expected_depth


@pytest.mark.parametrize("secret_key_dist", ["SPARSE_TERNARY", "UNIFORM_TERNARY"])
@pytest.mark.parametrize("strategy", ["double_hoist", "normal_giant", "normal_bsgs"])
def test_openfhe_bootstrap_rot_keys_supports_sparse_and_uniform(secret_key_dist, strategy):
    spec = bs.BootstrapSpec(
        log_slots=14,
        level_budget=(4, 4),
        output_levels=2,
        strategy=strategy,
    )
    rotations = bs.requirements(
        spec,
        log_n=16,
        secret_key_dist=secret_key_dist,
    ).rotations

    assert len(rotations) == len(set(rotations))
    assert rotations[-1] == (1 << 17) - 1
    assert rotations


def test_double_hoist_full_slot_rotations_match_runtime_giant_keys():
    spec = bs.BootstrapSpec(
        log_slots=15,
        level_budget=(2, 4),
        output_levels=2,
        strategy="double_hoist",
    )

    rotations = bs.requirements(spec, log_n=16).rotations
    runtime_offsets = tuple(
        rotation._double_hoist_giant_key_offset(index * 8192, SimpleNamespace(N=1 << 16))
        for index in range(1, 8)
    )

    assert runtime_offsets == (8192, 16384, 24576, 32768, 8192, 16384, 24576)
    assert set(runtime_offsets).issubset(rotations)
    assert all(offset not in rotations for offset in (40960, 49152, 57344))
