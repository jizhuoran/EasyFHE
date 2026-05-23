import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FHE_DIR = ROOT / "easyfhe" / "fhe"
BS_DIR = ROOT / "easyfhe" / "bs"
OPENFHE_BS_DIR = BS_DIR / "openfhe"


def _literal_assigned_value(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return _literal_value(tree, node.value)
    raise AssertionError(f"{name} assignment not found")


def _literal_value(tree, node):
    if isinstance(node, ast.Tuple):
        values = []
        for element in node.elts:
            if isinstance(element, ast.Starred) and isinstance(element.value, ast.Name):
                values.extend(_literal_assigned_value(tree, element.value.id))
            else:
                values.append(ast.literal_eval(element))
        return tuple(values)
    return ast.literal_eval(node)


def _imported_names(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def test_fhe_public_api_allowlist_is_well_formed():
    tree = ast.parse((FHE_DIR / "_public_api.py").read_text())
    public_api = _literal_assigned_value(tree, "PUBLIC_API")
    groups = _public_api_groups(tree)

    assert isinstance(public_api, tuple)
    assert len(public_api) == len(set(public_api))
    assert all(isinstance(name, str) for name in public_api)
    assert all(not name.startswith("_") for name in public_api)
    assert tuple(name for group in groups for name in group) == public_api


def _public_api_groups(tree):
    groups = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target_names = [
            target.id for target in node.targets if isinstance(target, ast.Name)
        ]
        if not target_names:
            continue
        name = target_names[0]
        if name.endswith("_API") and name != "PUBLIC_API":
            groups.append(_literal_assigned_value(tree, name))
    return groups


def test_fhe_root_exports_only_public_api_allowlist():
    init_tree = ast.parse((FHE_DIR / "__init__.py").read_text())
    public_api_tree = ast.parse((FHE_DIR / "_public_api.py").read_text())
    public_api = set(_literal_assigned_value(public_api_tree, "PUBLIC_API"))

    assert _uses_public_api_for_all(init_tree)
    assert public_api <= _imported_names(init_tree)


def test_bs_root_is_namespace_only():
    init_tree = ast.parse((BS_DIR / "__init__.py").read_text())
    assert _literal_assigned_value(init_tree, "__all__") == []


def test_openfhe_bs_public_api_allowlist_is_well_formed():
    tree = ast.parse((OPENFHE_BS_DIR / "_public_api.py").read_text())
    public_api = _literal_assigned_value(tree, "PUBLIC_API")
    groups = _public_api_groups(tree)

    assert isinstance(public_api, tuple)
    assert len(public_api) == len(set(public_api))
    assert all(isinstance(name, str) for name in public_api)
    assert all(not name.startswith("_") for name in public_api)
    assert tuple(name for group in groups for name in group) == public_api


def test_openfhe_bs_root_exports_only_public_api_allowlist():
    init_tree = ast.parse((OPENFHE_BS_DIR / "__init__.py").read_text())
    public_api_tree = ast.parse((OPENFHE_BS_DIR / "_public_api.py").read_text())
    public_api = set(_literal_assigned_value(public_api_tree, "PUBLIC_API"))

    assert _uses_public_api_for_all(init_tree)
    assert public_api <= _imported_names(init_tree) | _lazy_exported_names(init_tree)


def _lazy_exported_names(tree):
    names = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == "name"
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Eq)
            and len(node.comparators) == 1
            and isinstance(node.comparators[0], ast.Constant)
            and isinstance(node.comparators[0].value, str)
        ):
            names.add(node.comparators[0].value)
    return names


def _uses_public_api_for_all(tree):
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
            continue
        value = node.value
        return (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "list"
            and len(value.args) == 1
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id == "_PUBLIC_API"
        )
    return False
