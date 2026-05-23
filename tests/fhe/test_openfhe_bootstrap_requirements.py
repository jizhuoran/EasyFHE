import pytest

import easyfhe.bs.openfhe as bs


@pytest.mark.parametrize("secret_key_dist", ["SPARSE_TERNARY", "UNIFORM_TERNARY"])
@pytest.mark.parametrize("level_budget", [(1, 1), (1, 3), (4, 1)])
def test_openfhe_bootstrap_rejects_linear_transform_budget(secret_key_dist, level_budget):
    with pytest.raises(NotImplementedError, match="linear-transform route"):
        bs.depth(
            log_bs_slots=14,
            level_budget=level_budget,
            secret_key_dist=secret_key_dist,
        )

    with pytest.raises(NotImplementedError, match="linear-transform route"):
        bs.plan_rot_keys(
            log_n=16,
            log_bs_slots=14,
            level_budget=level_budget,
        )


@pytest.mark.parametrize(
    ("secret_key_dist", "expected_depth"),
    [
        ("SPARSE_TERNARY", 18),
        ("UNIFORM_TERNARY", 21),
    ],
)
def test_openfhe_bootstrap_depth_supports_sparse_and_uniform(secret_key_dist, expected_depth):
    assert (
        bs.depth(
            log_bs_slots=14,
            level_budget=(4, 4),
            secret_key_dist=secret_key_dist,
        )
        == expected_depth
    )


@pytest.mark.parametrize("secret_key_dist", ["SPARSE_TERNARY", "UNIFORM_TERNARY"])
@pytest.mark.parametrize("strategy", ["double_hoist", "normal_giant", "normal_bsgs"])
def test_openfhe_bootstrap_rot_keys_supports_sparse_and_uniform(secret_key_dist, strategy):
    rotations = bs.plan_rot_keys(
        log_n=16,
        log_bs_slots=14,
        level_budget=(4, 4),
        strategy=strategy,
    )

    assert len(rotations) == len(set(rotations))
    assert rotations[-1] == (1 << 17) - 1
    assert rotations
