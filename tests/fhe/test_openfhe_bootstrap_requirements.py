import pytest

import easyfhe.bs.openfhe as bs


@pytest.mark.parametrize("level_budget", [(1, 1), (1, 3), (4, 1)])
def test_openfhe_bootstrap_rejects_linear_transform_budget(level_budget):
    with pytest.raises(NotImplementedError, match="linear-transform route"):
        bs.depth(
            log_bs_slots=14,
            level_budget=level_budget,
            secret_key_dist="SPARSE_TERNARY",
        )

    with pytest.raises(NotImplementedError, match="linear-transform route"):
        bs.plan_rot_keys(
            log_n=16,
            log_bs_slots=14,
            level_budget=level_budget,
            secret_key_dist="SPARSE_TERNARY",
        )
