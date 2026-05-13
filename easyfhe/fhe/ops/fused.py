from . import kernels as F
from ..runtime.instrumentation import run_instrumented_op
from .arithmetic import homo_add
from .plaintext import homo_mul_pt


def fused_pairwise_mac(ctxs, ptxs, cryptoContext):
    return run_instrumented_op(cryptoContext, "fused_pairwise_mac", _fused_pairwise_mac, ctxs, ptxs, cryptoContext)


def _fused_pairwise_mac(ctxs, ptxs, cryptoContext):
    if len(ctxs) != 9 or len(ptxs) != 9:
        raise ValueError("The length of ctxs and ptxs must be 9, but got {} and {}".format(len(ctxs), len(ptxs)))

    if cryptoContext.device == "cpu":
        total = homo_mul_pt(ctxs[0], ptxs[0], cryptoContext)
        for i in range(1, 9):
            tmp = homo_mul_pt(ctxs[i], ptxs[i], cryptoContext)
            total = homo_add(tmp, total, cryptoContext)
        return total

    ctx_axs, ctx_bxs, ptx_bxs = [], [], []

    if ctxs[0].is_ext and ctxs[0].cv[0].shape[0] != ctxs[0].cur_limbs + cryptoContext.K:
        raise ValueError(
            "fused_pairwise_mac: ext cipher component has wrong active limb count: "
            f"shape[0]={ctxs[0].cv[0].shape[0]}, cur_limbs={ctxs[0].cur_limbs}, K={cryptoContext.K}"
        )

    for idx in range(len(ctxs)):
        if ctxs[idx].cur_limbs != ctxs[0].cur_limbs:
            raise ValueError(f"ctxs[{idx}].cur_limbs != ctxs[0].cur_limbs")
        if ctxs[idx].slots != ctxs[0].slots:
            raise ValueError(f"ctxs[{idx}].slots != ptxs[0].slots")
        if ctxs[idx].noise_deg != ctxs[0].noise_deg:
            raise ValueError(f"ctxs[{idx}].noise_deg != ctxs[0].noise_deg")
        if ctxs[idx].is_ext != ctxs[0].is_ext:
            raise ValueError(f"ctxs[{idx}].is_ext != ctxs[0].is_ext")

        if ptxs[idx].cur_limbs != ctxs[0].cur_limbs:
            raise ValueError(f"ptxs[{idx}].cur_limbs != ctxs[0].cur_limbs")
        if ptxs[idx].slots != ctxs[0].slots:
            raise ValueError(f"ptxs[{idx}].slots != ctxs[0].slots")
        if ptxs[idx].noise_deg != ctxs[0].noise_deg:
            raise ValueError(f"ptxs[{idx}].noise_deg != ctxs[0].noise_deg")
        if ptxs[idx].is_ext != ctxs[0].is_ext:
            raise ValueError(f"ptxs[{idx}].is_ext={ptxs[idx].is_ext} != ctxs[0].is_ext={ctxs[0].is_ext}")

        ctx_bxs.append(ctxs[idx].cv[0])
        ctx_axs.append(ctxs[idx].cv[1])
        ptx_bxs.append(ptxs[idx].cv[0])

    res = F.cipher_fused_pairwise_mac(
        ctx_bxs,
        ctx_axs,
        ptx_bxs,
        cryptoContext.QplusP_map[ctxs[0].cur_limbs],
        cryptoContext.QmuplusPmu_map[ctxs[0].cur_limbs],
        len(ctx_bxs),
        ctxs[0].cur_limbs + (cryptoContext.K if ctxs[0].is_ext else 0),
        cryptoContext.N,
    )
    return ctxs[0].cipher_like(
        [res[0], res[1]],
        scaling_factor=ctxs[0].scaling_factor * ptxs[0].scaling_factor,
        noise_deg=ctxs[0].noise_deg + ptxs[0].noise_deg,
    )


def fused_broadcast_mac(ctx, ptxs, cryptoContext):
    return run_instrumented_op(cryptoContext, "fused_broadcast_mac", _fused_broadcast_mac, ctx, ptxs, cryptoContext)


def _fused_broadcast_mac(ctx, ptxs, cryptoContext):
    if not (len(ptxs) == 16 or len(ptxs) == 32 or len(ptxs) == 64):
        raise ValueError("The length of ptxs must be 16, 32 or 64, but got {}".format(len(ptxs)))

    ptx_bxs = []

    for idx in range(len(ptxs)):
        if ptxs[idx].cur_limbs != ctx.cur_limbs:
            raise ValueError(f"ptxs[{idx}].cur_limbs != ctx.cur_limbs")
        if ptxs[idx].slots != ctx.slots:
            raise ValueError(f"ptxs[{idx}].slots != ptxs[0].slots")
        if ptxs[idx].noise_deg != ctx.noise_deg:
            raise ValueError(f"ptxs[{idx}].noise_deg != ctx.noise_deg")
        if ptxs[idx].scaling_factor != ctx.scaling_factor:
            raise ValueError(f"ptxs[{idx}].scaling_factor != ctx.scaling_factor")
        if ptxs[idx].is_ext != ctx.is_ext:
            raise ValueError(f"ptxs[{idx}].is_ext != ctx.is_ext")
        ptx_bxs.append(ptxs[idx].cv[0])

    res = F.cipher_fused_broadcast_mac(ctx.cv[0], ctx.cv[1], ptx_bxs, cryptoContext.moduliQ, cryptoContext.q_mu, len(ptx_bxs), ctx.cur_limbs, cryptoContext.N)
    return ctx.cipher_like([res[0], res[1]], scaling_factor=ctx.scaling_factor * ptxs[0].scaling_factor, noise_deg=ctx.noise_deg + ptxs[0].noise_deg)
