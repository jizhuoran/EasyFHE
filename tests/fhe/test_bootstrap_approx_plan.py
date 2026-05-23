from easyfhe.bs.openfhe.generation.plan import (
    KIND_C,
    KIND_Q,
    KIND_S,
    SPACE_NODE,
    SPACE_SMALL,
    compile_flat_ps_plan,
    describe_flat_ps_plan,
    get_bootstrap_approx_plan,
)


def _walk_nodes(node, path=("root",)):
    yield node, path
    if node.q_node is not None:
        yield from _walk_nodes(node.q_node, (*path, "q_node"))
    if node.s_node is not None:
        yield from _walk_nodes(node.s_node, (*path, "s_node"))


def _expected_small_paths(root):
    paths = set()
    for node, path in _walk_nodes(root):
        paths.add((KIND_C, (*path, "c")))
        if node.q_node is None:
            paths.add((KIND_Q, (*path, "q")))
        if node.s_node is None:
            paths.add((KIND_S, (*path, "s")))
    return paths


def _assert_ref_ready(ref, combine_idx):
    space, idx = ref
    if space == SPACE_SMALL:
        return
    if space == SPACE_NODE:
        assert idx < combine_idx
        return
    raise AssertionError(f"unexpected ref space: {space}")


def test_flat_ps_plan_preserves_recursive_paths_and_postorder():
    for secret_key_dist in ("SPARSE_TERNARY", "UNIFORM_TERNARY"):
        root = get_bootstrap_approx_plan(secret_key_dist).ps_root
        flat = compile_flat_ps_plan(root)
        nodes = tuple(_walk_nodes(root))

        assert flat.k == root.k
        assert flat.m == root.m
        assert flat.node_count == len(nodes)
        assert len(flat.combine_specs) == len(nodes)
        assert flat.root_ref == (SPACE_NODE, flat.node_count - 1)
        assert flat.tail_max_deg == max(spec.deg for spec in flat.tail_specs)

        suffix_by_kind = {KIND_C: "c", KIND_Q: "q", KIND_S: "s"}
        actual_small_paths = {
            (spec.kind, (*spec.path, suffix_by_kind[spec.kind]))
            for spec in flat.small_specs
        }
        assert actual_small_paths == _expected_small_paths(root)

        for tail_idx, spec in enumerate(flat.tail_specs):
            assert spec.out_idx == tail_idx
            assert spec.deg > 0
            assert spec.deg <= flat.tail_max_deg
            assert (*spec.path, suffix_by_kind[spec.kind]) == spec.scalar_path

        for spec in flat.small_specs:
            assert spec.tail_idx is None or spec.tail_idx < len(flat.tail_specs)
            assert not (spec.direct_t1 and spec.tail_idx is not None)

        for index, spec in enumerate(flat.combine_specs):
            assert spec.out_idx == index
            assert spec.base_idx == spec.node.m - 1
            _assert_ref_ready(spec.q_ref, index)
            _assert_ref_ready(spec.s_ref, index)
            assert spec.c_ref[0] == SPACE_SMALL


def test_flat_ps_plan_description_names_tables():
    flat = compile_flat_ps_plan(get_bootstrap_approx_plan("SPARSE_TERNARY").ps_root)

    text = describe_flat_ps_plan(flat)

    assert "Approx PS Plan" in text
    assert "Tails" in text
    assert "Small" in text
    assert "Combine" in text
    assert "tail[00]" in text
    assert "small[00]" in text
    assert f"node[{flat.node_count - 1:02d}]" in text
