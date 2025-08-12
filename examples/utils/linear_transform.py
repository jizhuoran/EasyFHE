import torch.fhe as fhe


def eval_linear_transform(A, ct, cryptoContext, bStep, gStep):
    slots = bStep * gStep
    digits = fhe.hybrid_keyswitch.modup_to_ext(fhe.homo_ops.extract_cv(ct, 1, cryptoContext), cryptoContext)
    fastRotationExt = []
    for j in range(1, bStep):
        fastRotationExt.append(fhe.homo_ops.eval_fast_rotate(digits, ct, j, True, False, cryptoContext))
    inner_ext_copy = fhe.hybrid_keyswitch.key_switch_P_ext(ct, cryptoContext)
    result_ext = fhe.homo_ops.homo_mul_pt(inner_ext_copy, A[0], cryptoContext)
    for i in range(1, bStep):
        if i < slots:
            tmp_ext = fhe.homo_ops.homo_mul_pt(fastRotationExt[i - 1], A[i], cryptoContext)

            result_ext = fhe.homo_ops.homo_add(result_ext, tmp_ext, cryptoContext)

    inner_ext = None
    for j in range(1, gStep):
        inner_ext = fhe.homo_ops.homo_mul_pt(inner_ext_copy, A[bStep * j], cryptoContext)

        for i in range(1, bStep):
            if bStep * j + i < slots:
                tmp_ext = fhe.homo_ops.homo_mul_pt(fastRotationExt[i - 1], A[bStep * j + i], cryptoContext)
                inner_ext = fhe.homo_ops.homo_add(inner_ext, tmp_ext, cryptoContext)
        inner_ax = fhe.hybrid_keyswitch.moddown_from_ext(fhe.homo_ops.extract_cv(inner_ext, 1, cryptoContext),
                                                         cryptoContext)  # todo: rename to inner_Down
        inner_digits = fhe.hybrid_keyswitch.modup_to_ext(inner_ax, cryptoContext)
        tmp = fhe.homo_ops.eval_fast_rotate(inner_digits, inner_ext, bStep * j, True, False, cryptoContext)
        result_ext = fhe.homo_ops.homo_add(result_ext, tmp, cryptoContext)
    result = fhe.hybrid_keyswitch.moddown_from_ext(result_ext, cryptoContext)
    return result
