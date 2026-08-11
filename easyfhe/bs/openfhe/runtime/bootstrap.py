from easyfhe.fhe.ciphertext import CipherState
from easyfhe.fhe.ops import alignment


def _require_supported_transform_mode(level_budget):
    if int(level_budget[0]) == 1 or int(level_budget[1]) == 1:
        raise NotImplementedError(
            "OpenFHE bootstrap does not support the linear-transform route; "
            f"both level_budget entries must be greater than 1, got {tuple(level_budget)}"
        )


def eval_bootstrap(
    ciphertext,
    crypto_context,
    bootstrap_constants,
    bootstrap_plan,
    *,
    raise_to_limbs,
    mode,
):
    _require_supported_transform_mode(bootstrap_plan.level_budget)
    if mode == "stc_first":
        from .bootstrap_stc_first import eval_bootstrap_stc_first

        return eval_bootstrap_stc_first(
            ciphertext,
            crypto_context,
            bootstrap_constants,
            bootstrap_plan,
            raise_to_limbs,
        )
    if mode != "modraise_first":
        raise ValueError("bootstrap mode must be one of: modraise_first, stc_first")

    from .bootstrap_modraise_first import eval_bootstrap_modraise_first

    return eval_bootstrap_modraise_first(
        ciphertext,
        crypto_context,
        bootstrap_constants,
        bootstrap_plan,
        raise_to_limbs,
    )


def homo_bootstrap(
    cipher,
    crypto_context,
    bootstrap_constants,
    bootstrap_plan,
    *,
    raise_to_limbs,
    output_state: CipherState,
    mode,
):
    result = eval_bootstrap(
        cipher,
        crypto_context,
        bootstrap_constants,
        bootstrap_plan,
        raise_to_limbs=raise_to_limbs,
        mode=mode,
    )
    result = alignment.reduce_noise_to_one(result, crypto_context)
    if result.state != output_state:
        result = alignment.align_to(result, output_state, crypto_context)
    return result
