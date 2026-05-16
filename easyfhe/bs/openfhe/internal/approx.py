import math

import numpy as np

from easyfhe.fhe.ops import alignment
from easyfhe.fhe.ops import homo
from easyfhe.fhe.ops.primitives import _cipher_add_scalar, _cipher_sub_scalar
from .approx_plan import ChebyshevPSNode, degree, get_bootstrap_approx_plan
from easyfhe.fhe.context import Context

BASE_NUM_LEVELS_TO_DROP = 1
MAX_BITS_IN_WORD = 61
MAX_LOG_STEP = 60


def _match_state(ciphertext, cur_limbs, noise_deg, scaling_factor, cryptoContext):
    return alignment.align_to(
        ciphertext,
        alignment.CipherState(cur_limbs, noise_deg, scaling_factor),
        cryptoContext,
    )


def _rescale(ciphertext, levels, cryptoContext):
    return _match_state(
        ciphertext,
        ciphertext.cur_limbs - levels,
        ciphertext.noise_deg - levels,
        None,
        cryptoContext,
    )


def _crt_mult(xs, ys, mods):
    return [(int(x) * int(y)) % int(mod) for x, y, mod in zip(xs, ys, mods)]


def _encode_scalar_for_add(constant, cur_limbs, noise_deg, cryptoContext):
    sc_factor = cryptoContext.scale_at(cur_limbs)

    log_approx = 0
    magnitude = math.fabs(constant * sc_factor)
    if magnitude > 0:
        log_sf = int(math.ceil(math.log2(magnitude)))
        log_valid = min(log_sf, MAX_BITS_IN_WORD)
        log_approx = log_sf - log_valid

    approx_factor = float(2 ** log_approx)
    sc_constant = int(constant * sc_factor / approx_factor + 0.5)
    crt_constant = cur_limbs * [sc_constant]

    if log_approx > 0:
        log_step = min(log_approx, MAX_LOG_STEP)
        int_step = 2 ** log_step
        crt_approx = cur_limbs * [int_step]
        log_approx -= log_step

        while log_approx > 0:
            log_step = min(log_approx, MAX_LOG_STEP)
            int_step = 2 ** log_step
            crt_sf = cur_limbs * [int_step]
            crt_approx = _crt_mult(crt_approx, crt_sf, cryptoContext.moduliQ_scalar)
            log_approx -= log_step

        crt_constant = _crt_mult(crt_constant, crt_approx, cryptoContext.moduliQ_scalar)

    int_sc_factor = int(sc_factor + 0.5)
    crt_sc_factor = cur_limbs * [int_sc_factor]
    for _ in range(1, noise_deg):
        crt_constant = _crt_mult(crt_constant, crt_sc_factor, cryptoContext.moduliQ_scalar)

    return crt_constant


def _add_scalar_double_preserve_noise(ciphertext, constant, cryptoContext):
    encoded = _encode_scalar_for_add(
        math.fabs(constant),
        ciphertext.cur_limbs,
        ciphertext.noise_deg,
        cryptoContext,
    )
    if constant < 0:
        return _cipher_sub_scalar(ciphertext, encoded, cryptoContext)
    return _cipher_add_scalar(ciphertext, encoded, cryptoContext)


def eval_linear_wsum_mutable(ciphertexts, constants, cryptoContext: Context):
    if cryptoContext.rescaleTech != "FIXEDMANUAL":
        target_idx = min(range(len(ciphertexts)), key=lambda i: ciphertexts[i].cur_limbs - ciphertexts[i].noise_deg)
        if ciphertexts[target_idx].noise_deg == 2:
            ciphertexts[target_idx] = _rescale(ciphertexts[target_idx], 1, cryptoContext)
        for i in range(len(ciphertexts)):
            ciphertexts[i] = _match_state(
                ciphertexts[i], ciphertexts[target_idx].cur_limbs, ciphertexts[target_idx].noise_deg, ciphertexts[target_idx].scaling_factor, cryptoContext
            )

    wsum = homo.homo_mul_scalar_double(ciphertexts[0], constants[0], cryptoContext)
    for i in range(1, len(constants)):
        tmp = homo.homo_mul_scalar_double(ciphertexts[i], constants[i], cryptoContext)
        wsum = homo.homo_add(wsum, tmp, cryptoContext)
    wsum = _rescale(wsum, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
    return wsum

def inner_eval_chebyshev_ps(node: ChebyshevPSNode, T, T2, cryptoContext: Context):
    k = node.k
    m = node.m
    divqr_q = node.divqr_q
    divcs_q = node.divcs_q
    s2 = node.s2

    # Evaluate c at u
    dc = degree(divcs_q)
    flag_c = False
    if dc >= 1:
        if dc == 1:
            if divcs_q[1] != 1:
                cu = homo.homo_mul_scalar_double(T[0], divcs_q[1], cryptoContext)
                cu = _rescale(cu, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
            else:
                cu = T[0]
        else:
            ctxs = [T[i] for i in range(dc)]
            weights = divcs_q[1:dc + 1]
            cu = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)

        # adds the free term (at x^0)
        cu = homo.homo_add_scalar_double(cu, divcs_q[0] / 2, cryptoContext)
        # Need to reduce levels up to the level of T2[m-1].
        if cryptoContext.rescaleTech == "FIXEDMANUAL":
            cu = _match_state(cu, T2[m - 1].cur_limbs, T2[m - 1].noise_deg, T2[m - 1].scaling_factor, cryptoContext)
        flag_c = True

    # Evaluate q and s2 at u
    if node.q_node is not None:
        qu = inner_eval_chebyshev_ps(node.q_node, T, T2, cryptoContext)
    else:
        qcopy = np.copy(divqr_q)
        qcopy.resize(k, refcheck=False)
        deg_qcopy = degree(qcopy)
        if deg_qcopy > 0:
            ctxs = [T[i] for i in range(deg_qcopy)]
            weights = divqr_q[1:deg_qcopy + 1]
            qu = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)
            sum = T[k - 1]
            sum = homo.homo_mul_scalar_int(T[k - 1], 2 ** math.floor(math.log2(divqr_q[-1] + 1.1)), cryptoContext)
            # for i in range(int(math.log2(divqr_q[-1]))):
            # sum = homo.homo_add(sum, sum, cryptoContext)
            qu = homo.homo_add(qu, sum, cryptoContext)
        else:
            sum = T[k - 1]
            sum = homo.homo_mul_scalar_int(T[k - 1], 2 ** math.floor(math.log2(divqr_q[-1])), cryptoContext)
            # for i in range(int(math.log2(divqr_q[-1]))):
            # sum = homo.homo_add(sum, sum, cryptoContext)
            qu = sum

        qu = homo.homo_add_scalar_double(qu, divqr_q[0] / 2, cryptoContext)

    # Evaluate s2 at u
    if node.s_node is not None:
        su = inner_eval_chebyshev_ps(node.s_node, T, T2, cryptoContext)
    else:
        scopy = np.copy(s2)
        scopy.resize(k, refcheck=False)
        deg_scopy = degree(scopy)
        if deg_scopy > 0:
            ctxs = [T[i] for i in range(deg_scopy)]
            weights = s2[1:deg_scopy + 1]
            su = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)
            su = homo.homo_add(su, T[k - 1], cryptoContext)
        else:
            su = T[k - 1]

        su = homo.homo_add_scalar_double(su, s2[0] / 2, cryptoContext)
        if cryptoContext.rescaleTech == "FIXEDMANUAL":
            su = _match_state(su, su.cur_limbs - 1, 1, None, cryptoContext)

    if flag_c:
        result = homo.homo_add(T2[m - 1], cu, cryptoContext)
    else:
        result = homo.homo_add_scalar_double(T2[m - 1], divcs_q[0] / 2, cryptoContext)

    result = homo.homo_mul(result, qu, cryptoContext)
    result = _rescale(result, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
    result = homo.homo_add(result, su, cryptoContext)

    return result

# note: EvalChebyshevSeriesPS in ckksrns-advancedshe.cpp
# @profile_pytorch_function
def eval_bootstrapping_chebyshev(x, a, b, cryptoContext):
    plan = get_bootstrap_approx_plan(cryptoContext.secretKeyDist)
    root = plan.ps_root
    k = root.k
    m = root.m
    divqr_q = root.divqr_q
    divcs_q = root.divcs_q
    s2 = root.s2

    T = [x]
    alpha = 2 / (b - a)
    if not math.isclose(alpha, 1.0):
        T[0] = homo.homo_mul_scalar_double(x, alpha, cryptoContext)
        T[0] = _rescale(T[0], 1, cryptoContext)
    beta = 2 * a / (b - a)
    if not math.isclose(beta, -1.0):
        T[0] = homo.homo_add_scalar_double(T[0], -1.0 - beta, cryptoContext)

    for i in range(2, k + 1):
        prod = homo.homo_mul(T[i // 2 - 1], T[(i + 1) // 2 - 1], cryptoContext)
        tmp = homo.homo_add(prod, prod, cryptoContext)
        tmp = _rescale(tmp, 1, cryptoContext)
        if i & 1 == 1:  # i is odd
            tmp = homo.homo_sub(tmp, T[0], cryptoContext)
        else:
            tmp = homo.homo_add_scalar_double(tmp, -1.0, cryptoContext)
        T.append(tmp)

    for i in range(k):
        T[i] = _match_state(T[i], T[-1].cur_limbs, T[-1].noise_deg, T[-1].scaling_factor, cryptoContext)

    # Compute the Chebyshev polynomials T_k(y), T_{2k}(y), T_{4k}(y), ... , T_{2^{m-1}k}(y)
    # T2[0] is used as a placeholder
    T2 = [T[-1]]
    for i in range(1, m):
        tmp = homo.homo_square(T2[i - 1], cryptoContext)
        tmp = homo.homo_add(tmp, tmp, cryptoContext)
        tmp = _rescale(tmp, 1, cryptoContext)
        tmp = homo.homo_add_scalar_double(tmp, -1.0, cryptoContext)
        T2.append(tmp)



    # computes T_{k(2*m - 1)}(y)
    T2km1 = T2[0]
    for i in range(1, m):
        # compute T_{k(2*m - 1)} = 2*T_{k(2^{m-1}-1)}(y)*T_{k*2^{m-1}}(y) - T_k(y)
        prod = homo.homo_mul(T2km1, T2[i], cryptoContext)
        T2km1 = homo.homo_add(prod, prod, cryptoContext)
        T2km1 = _rescale(T2km1, 1, cryptoContext)
        T2km1 = homo.homo_sub(T2km1, T2[0], cryptoContext)



    dc = degree(divcs_q)
    flag_c = False
    if dc >= 1:
        if dc == 1:
            if divcs_q[1] != 1:
                cu = homo.homo_mul_scalar_double(T[0], divcs_q[1], cryptoContext)
                cu = _rescale(cu, 1, cryptoContext)
            else:
                cu = T[0]
        else:
            ctxs = [T[i] for i in range(dc)]
            weights = divcs_q[1:dc + 1]
            cu = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)

        # adds the free term (at x^0)
        cu = homo.homo_add_scalar_double(cu, divcs_q[0] / 2, cryptoContext)
        flag_c = True



    # Evaluate q and s2 at u. If their degrees are larger than k, then recursively apply the Paterson-Stockmeyer algorithm.
    if root.q_node is not None:
        qu = inner_eval_chebyshev_ps(root.q_node, T, T2, cryptoContext)
    else:
        # dq = k from construction
        # perform scalar multiplication for all other terms and sum them up if there are non-zero coefficients
        q_copy = np.copy(divqr_q[:k])
        deg_qcopy = degree(q_copy)
        if deg_qcopy > 0:
            ctxs = [T[i] for i in range(deg_qcopy)]
            weights = divqr_q[1:deg_qcopy + 1]
            qu = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)
            # the highest order coefficient will always be 2 after one division because of the Chebyshev division rule
            sum = homo.homo_add(T[k - 1], T[k - 1], cryptoContext)
            qu = homo.homo_add(qu, sum, cryptoContext)
        else:
            qu = T[k - 1]
            for _ in range(1, divqr_q[- 1]):
                qu = homo.homo_add(qu, T[k - 1], cryptoContext)

        # adds the free term (at x^0)
        qu = homo.homo_add_scalar_double(qu, divqr_q[0] / 2, cryptoContext)
        # The number of levels of qu is the same as the number of levels of T[k-1] + 1.
        # Will only get here when m = 2, so the number of levels of qu and T2[m-1] will be the same.

    # Evaluate s2 at u
    if root.s_node is not None:
        su = inner_eval_chebyshev_ps(root.s_node, T, T2, cryptoContext)
    else:
        # ds = k from construction
        # perform scalar multiplication for all other terms and sum them up if there are non-zero coefficients
        scopy = np.copy(s2[:k])
        deg_scopy = degree(scopy)
        if deg_scopy > 0:
            ctxs = [T[i] for i in range(deg_scopy)]
            weights = s2[1:deg_scopy + 1]
            su = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)
            # the highest order coefficient will always be 1 because s2 is monic.
            su = homo.homo_add(su, T[k - 1], cryptoContext)
        else:
            su = T[k - 1]
        # adds the free term (at x^0)
        su = homo.homo_add_scalar_double(su, s2[0] / 2, cryptoContext)
        # The number of levels of su is the same as the number of levels of T[k-1] + 1.
        # Will only get here when m = 2, so need to reduce the number of levels by 1.

    if flag_c:
        result = homo.homo_add(T2[m - 1], cu, cryptoContext)
    else:
        result = homo.homo_add_scalar_double(T2[m - 1], divcs_q[0] / 2, cryptoContext)



    result = homo.homo_mul(result, qu, cryptoContext)
    result = _rescale(result, 1, cryptoContext)
    result = homo.homo_add(result, su, cryptoContext)


    result = homo.homo_sub(result, T2km1, cryptoContext)



    return result

def apply_double_angle_iterations(ciphertext, cryptoContext):
    plan = get_bootstrap_approx_plan(cryptoContext.secretKeyDist)

    for j in range(1, plan.double_angle_iterations + 1):
        ciphertext = homo.homo_square(ciphertext, cryptoContext)
        ciphertext = homo.homo_add(ciphertext, ciphertext, cryptoContext)
        scalar = -1.0 / math.pow(
            2.0 * math.pi,
            math.pow(2.0, j - plan.double_angle_iterations),
        )
        ciphertext = _add_scalar_double_preserve_noise(ciphertext, scalar, cryptoContext)
        ciphertext = _rescale(ciphertext, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
    return ciphertext


def eval_bootstrap_approx_mod(ciphertext, cryptoContext):
    ciphertext = eval_bootstrapping_chebyshev(ciphertext, -1, 1, cryptoContext)
    if cryptoContext.rescaleTech != "FIXEDMANUAL":
        ciphertext = _rescale(ciphertext, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
    return apply_double_angle_iterations(ciphertext, cryptoContext)
