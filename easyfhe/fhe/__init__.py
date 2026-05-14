"""Public CKKS-oriented FHE frontend for EasyFHE."""

from . import config, utils
from .bootstrap import (
    BootstrapConstants,
    BootstrapPlan,
    generate_bootstrap_constants,
    homo_bootstrap,
)
from .ciphertext import PreparedPlaintext
from .context import Context
from .ops import (
    CipherState,
    align_to,
    encode,
    eval_fast_rotate,
    extract_cv,
    fused_pairwise_mac,
    homo_add,
    homo_add_pt,
    homo_add_scalar_double,
    homo_add_scalar_int,
    homo_conjugate,
    homo_mul,
    homo_mul_pt,
    homo_mul_scalar_double,
    homo_mul_scalar_int,
    homo_rotate,
    homo_square,
    homo_sub,
    make_plaintext,
    prepare_plaintext,
    reduce_noise_to_one,
    rescale_one_level,
    slot_resize,
)
from .ops import homo as homo_ops
from .ops.keyswitch import (
    key_switch_P_ext,
    moddown_from_ext,
    modup_to_ext,
    mult_rot_key_and_sum_ext,
)
from .runtime.cli import add_output_args, add_runtime_args, runtime_options_from_args
from .runtime.instrumentation import NullInstrumentation, OpInstrumentation, profile
from .runtime.options import RuntimeOptions
from .runtime.spec import (
    BootstrapSpec,
    CKKSContextSpec,
    bootstrap_depth,
    generate_context,
)


__all__ = [
    "BootstrapConstants",
    "BootstrapPlan",
    "BootstrapSpec",
    "CKKSContextSpec",
    "CipherState",
    "Context",
    "NullInstrumentation",
    "OpInstrumentation",
    "PreparedPlaintext",
    "RuntimeOptions",
    "add_output_args",
    "add_runtime_args",
    "align_to",
    "bootstrap_depth",
    "config",
    "encode",
    "eval_fast_rotate",
    "extract_cv",
    "fused_pairwise_mac",
    "generate_bootstrap_constants",
    "generate_context",
    "homo_add",
    "homo_add_pt",
    "homo_add_scalar_double",
    "homo_add_scalar_int",
    "homo_bootstrap",
    "homo_conjugate",
    "homo_mul",
    "homo_mul_pt",
    "homo_mul_scalar_double",
    "homo_mul_scalar_int",
    "homo_ops",
    "homo_rotate",
    "homo_square",
    "homo_sub",
    "key_switch_P_ext",
    "make_plaintext",
    "moddown_from_ext",
    "modup_to_ext",
    "mult_rot_key_and_sum_ext",
    "prepare_plaintext",
    "profile",
    "reduce_noise_to_one",
    "rescale_one_level",
    "runtime_options_from_args",
    "slot_resize",
    "utils",
]
