import easyfhe.fhe as fhe


def test_context_key_requirements_accept_bootstrap_outputs():
    constants = fhe.ConstantBundle(
        info={"required_rotations": (1, -2)},
        vectors={},
    )

    assert fhe.Context._rotation_groups_from_key_requirements((1, -2)) == (1, -2)
    assert fhe.Context._rotation_groups_from_key_requirements(constants) == ((1, -2),)
    assert fhe.Context._rotation_groups_from_key_requirements(((1, -2), constants)) == (1, -2)
    assert fhe.Context._rotation_groups_from_key_requirements((34, (1, -2), ("pt",))) == (1, -2)
