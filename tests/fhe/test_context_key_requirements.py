import easyfhe.fhe as fhe


class _BootstrapPlan:
    required_rotations = (1, -2)


def test_context_key_requirements_accept_bootstrap_outputs():
    constants = fhe.ConstantBundle(vectors={})
    plan = _BootstrapPlan()

    assert fhe.Context._rotation_groups_from_key_requirements((1, -2)) == (1, -2)
    assert fhe.Context._rotation_groups_from_key_requirements(plan) == ((1, -2),)
    assert fhe.Context._rotation_groups_from_key_requirements(((1, -2), constants, plan)) == (1, -2)
    assert fhe.Context._rotation_groups_from_key_requirements((34, (1, -2), ("pt",))) == (1, -2)
