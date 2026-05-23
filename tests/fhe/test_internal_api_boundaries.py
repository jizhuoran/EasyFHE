import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EASYFHE_DIR = ROOT / "easyfhe"
RESNET20_DIR = ROOT / "examples" / "resnet20_aespa"
KERNELS_PATH = EASYFHE_DIR / "fhe" / "ops" / "kernels.py"
NATIVE_SAMPLER_PATH = EASYFHE_DIR / "fhe" / "_keygen" / "native_sampler.py"
NATIVE_SAMPLER_OPS = {
    "fhe_native_sample_ckks",
    "fhe_native_sample_rotation_keys",
}

NATIVE_FHE_OPS = {
    "fhe_native_sample_ckks",
    "fhe_native_sample_rotation_keys",
    "neg_mod",
    "neg_mod_",
    "add_mod",
    "add_mod_",
    "sub_mod",
    "sub_mod_",
    "mul_mod",
    "mul_mod_",
    "add_scalar_mod",
    "add_scalar_mod_",
    "sub_scalar_mod",
    "sub_scalar_mod_",
    "mul_scalar_mod",
    "mul_scalar_mod_",
    "cv_add_pair",
    "cv_add_pair_",
    "cv_sub_pair",
    "cv_sub_pair_",
    "cv_mul_pt_pair",
    "cv_mul_pt_pair_",
    "cv_mul_scalar_pair",
    "cv_mul_scalar_pair_",
    "modup",
    "moddown",
    "innerproduct_broadcast",
    "rescale_one_level",
    "finalize_fast_rotation_ext",
    "finalize_fast_rotation_q",
    "hrot",
    "hmul_relin_rescale",
    "mod_raise",
    "extend_ciphertext",
    "mul_by_monomial",
    "mul_by_monomial_",
    "encode",
    "encrypt",
    "pre_encode",
    "batched_pairwise_mac",
    "grouped_scalar_weighted_acc",
}


def _python_files(root):
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _tree(path):
    return ast.parse(path.read_text(), filename=str(path))


def test_resnet20_aespa_imports_only_public_easyfhe_roots():
    violations = []
    for path in _python_files(RESNET20_DIR):
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if _is_forbidden_resnet_import(name):
                        violations.append((path, node.lineno, f"import {name}"))
            elif isinstance(node, ast.ImportFrom) and node.module:
                if _is_forbidden_resnet_import(node.module):
                    violations.append((path, node.lineno, f"from {node.module} import ..."))

    assert violations == []


def _is_forbidden_resnet_import(module_name):
    return (
        module_name.startswith("easyfhe.fhe.")
        or module_name.startswith("easyfhe.bs.openfhe.")
    )


def test_bootstrap_internals_do_not_import_ops_homo_barrel():
    violations = []
    for path in _python_files(EASYFHE_DIR / "bs"):
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "easyfhe.fhe.ops.homo":
                        violations.append((path, node.lineno, f"import {alias.name}"))
            elif isinstance(node, ast.ImportFrom) and node.module == "easyfhe.fhe.ops":
                for alias in node.names:
                    if alias.name == "homo":
                        violations.append((path, node.lineno, "from easyfhe.fhe.ops import homo"))

    assert violations == []


def test_native_fhe_ops_are_called_only_from_kernel_wrappers():
    violations = []
    for path in _python_files(EASYFHE_DIR):
        if path == KERNELS_PATH:
            continue
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "torch"
                and func.attr in NATIVE_FHE_OPS
            ):
                if path == NATIVE_SAMPLER_PATH and func.attr in NATIVE_SAMPLER_OPS:
                    continue
                violations.append((path, node.lineno, f"torch.{func.attr}"))

    assert violations == []
