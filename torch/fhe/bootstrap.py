from .homo_ops import *


def InnerL1(
    T,
    T2,
    cryptoContext,
    divcs_q,
    s_divcs_q,
    s_divqr_q,
    q_divcs_q,
    q_divqr_q,
    q_s2,
    s_s2,
    s_lg2_divqr_q_last,
    q_lg2_divqr_q_last,
):
    # weightedSum Core computation
    # level = T[4] -1
    tmp = cipher_level_reduce(T[4], 1)
    q_su = homo_mul_scalar_double(tmp, s_divcs_q[5], cryptoContext)
    qs_qu = homo_mul_scalar_double(tmp, s_divqr_q[5], cryptoContext)

    # level = T[4]
    q_qu = homo_mul_scalar_double(T[4], q_divcs_q[5], cryptoContext)
    qq_qu = homo_mul_scalar_double(T[4], q_divqr_q[5], cryptoContext)

    for i in range(4):
        # level = T[i] -1
        tmp1 = cipher_level_reduce(T[i], 1)
        tmp = homo_mul_scalar_double(tmp1, s_divcs_q[i + 1], cryptoContext)
        q_su = homo_add(q_su, tmp, cryptoContext)
        tmp = homo_mul_scalar_double(tmp1, s_divqr_q[i + 1], cryptoContext)
        qs_qu = homo_add(qs_qu, tmp, cryptoContext)

        # level = T[i]
        tmp = homo_mul_scalar_double(T[i], q_divcs_q[i + 1], cryptoContext)
        q_qu = homo_add(q_qu, tmp, cryptoContext)

        tmp = homo_mul_scalar_double(T[i], q_divqr_q[i + 1], cryptoContext)
        # NOTE: set the last element of q_divqr_q zero
        qq_qu = homo_add(qq_qu, tmp, cryptoContext)

    qq_qu = cipher_rescale(qq_qu, cryptoContext)
    qs_qu = cipher_rescale(qs_qu, cryptoContext)

    tmp = cipher_level_reduce(T[5], 1)
    tmp = homo_mul_scalar_int(tmp, (1 << int(s_lg2_divqr_q_last)), cryptoContext)
    qs_qu = homo_add(qs_qu, tmp, cryptoContext)
    qs_qu = homo_add_scalar_double(qs_qu, s_divqr_q[0], cryptoContext)

    tmp = T[5]
    tmp = homo_mul_scalar_int(tmp, (1 << int(q_lg2_divqr_q_last)), cryptoContext)
    qq_qu = homo_add(qq_qu, tmp, cryptoContext)
    qq_qu = homo_add_scalar_double(qq_qu, q_divqr_q[0], cryptoContext)

    q_su = cipher_rescale(q_su, cryptoContext)
    q_su = homo_add_scalar_double(q_su, s_divcs_q[0], cryptoContext)
    tmp = cipher_level_reduce(T2[1], 1)
    q_su = homo_add(q_su, tmp, cryptoContext)

    q_qu = cipher_rescale(q_qu, cryptoContext)
    q_qu = homo_add_scalar_double(q_qu, q_divcs_q[0], cryptoContext)
    q_qu = homo_add(q_qu, T2[1], cryptoContext)

    q_qu = homo_mul(q_qu, qq_qu, cryptoContext)
    tmp = cipher_level_reduce(T[4], 1)
    qq_su = homo_mul_scalar_double(tmp, q_s2[5], cryptoContext)

    for i in range(4):
        tmp = cipher_level_reduce(T[i], 1)
        tmp = homo_mul_scalar_double(tmp, q_s2[i + 1], cryptoContext)
        qq_su = homo_add(qq_su, tmp, cryptoContext)

    q_qu = homo_add(q_qu, qq_su, cryptoContext)
    q_qu = cipher_rescale(q_qu, cryptoContext)
    tmp = cipher_level_reduce(T[5], 1)
    q_qu = homo_add(q_qu, tmp, cryptoContext)
    q_qu = homo_add_scalar_double(q_qu, q_s2[0], cryptoContext)

    # lazy KS: merge KS in computing `q_su` and `qu`
    tmp = cipher_level_reduce(T[4], 1)
    qu = homo_mul_scalar_double(tmp, divcs_q[5], cryptoContext)
    for i in range(4):
        # level = T[i] -1
        tmp = cipher_level_reduce(T[i], 1)
        tmp = homo_mul_scalar_double(tmp, divcs_q[i + 1], cryptoContext)
        qu = homo_add(qu, tmp, cryptoContext)
    qu = cipher_rescale(qu, cryptoContext)
    qu = homo_add_scalar_double(qu, divcs_q[0], cryptoContext)
    qu = homo_add(qu, T2[2], cryptoContext)

    tmp_qu = cipher_mul(qu, q_qu, cryptoContext)
    tmp_q_su = cipher_mul(q_su, qs_qu, cryptoContext)
    tmp = cipher_level_reduce(T[4], 2)
    qs_su = homo_mul_scalar_double(tmp, s_s2[5], cryptoContext)
    for i in range(4):
        # level = T[i] -1
        tmp = cipher_level_reduce(T[i], 2)
        tmp = homo_mul_scalar_double(tmp, s_s2[i + 1], cryptoContext)
        qs_su = homo_add(qs_su, tmp, cryptoContext)

    qu = cipher_add(tmp_qu, tmp_q_su, cryptoContext)
    axax = qu.drop_axax()

    qu = cipher_add(qu, qs_su, cryptoContext)

    summult = F.cv_keyswitch(axax, qu.cur_limbs, cryptoContext)

    summult = Cipher(summult, q_qu.cur_limbs)

    qu = cipher_add(qu, summult, cryptoContext)

    qu = cipher_rescale(qu, cryptoContext)
    tmp = cipher_level_reduce(T[5], 2)
    qu = homo_add(qu, tmp, cryptoContext)
    qu = homo_add_scalar_double(qu, s_s2[0], cryptoContext)

    return qu


def EvalChebyshevSeries(x, cryptoContext):

    T = [
        None,
        None,
        None,
        None,
        None,
        None,
    ]

    # no linear transformation is needed if a = -1, b = 1, T_1(y) = y
    T[0] = x
    # Computes Chebyshev polynomials up to degree k
    # for y: T_1(y) = y, T_2(y), ... , T_k(y)
    # uses binary tree multiplication
    T[1] = homo_square(T[0], cryptoContext)
    T[1] = homo_add(T[1], T[1], cryptoContext)
    T[1] = cipher_rescale(T[1], cryptoContext)
    T[1] = homo_add_scalar_double(T[1], -1.0, cryptoContext)

    T[0] = cipher_level_reduce(T[0], 1)
    T[2] = homo_mul(T[0], T[1], cryptoContext)
    T[2] = homo_add(T[2], T[2], cryptoContext)
    T[2] = cipher_rescale(T[2], cryptoContext)
    T[0] = cipher_level_reduce(T[0], 1)
    T[2] = homo_sub(T[2], T[0], cryptoContext)
    T[1] = cipher_level_reduce(T[1], 1)

    T[3] = homo_square(T[1], cryptoContext)
    T[3] = homo_add(T[3], T[3], cryptoContext)
    T[3] = cipher_rescale(T[3], cryptoContext)
    T[3] = homo_add_scalar_double(T[3], -1.0, cryptoContext)

    T[0] = cipher_level_reduce(T[0], 1)

    T[4] = homo_mul(T[1], T[2], cryptoContext)
    T[4] = homo_add(T[4], T[4], cryptoContext)
    T[4] = cipher_rescale(T[4], cryptoContext)
    T[4] = homo_sub(T[4], T[0], cryptoContext)

    T[5] = homo_square(T[2], cryptoContext)
    T[5] = homo_add(T[5], T[5], cryptoContext)
    T[5] = cipher_rescale(T[5], cryptoContext)
    T[5] = homo_add_scalar_double(T[5], -1.0, cryptoContext)

    T[1] = cipher_level_reduce(T[1], 1)
    T[2] = cipher_level_reduce(T[2], 1)

    T2 = [
        None,
        None,
        None,
        None,
    ]

    # Compute the Chebyshev polynomials T_{2k}(y), T_{4k}(y), ... , T_{2^{m-1}k}(y)
    T2[1] = homo_square(T[5], cryptoContext)
    T[5] = cipher_level_reduce(T[5], 1)
    T2[1] = homo_add(T2[1], T2[1], cryptoContext)
    T2[1] = cipher_rescale(T2[1], cryptoContext)
    T2[1] = homo_add_scalar_double(T2[1], -1.0, cryptoContext)

    # compute T_{k(2*m - 1)} = 2*T_{k(2^{m-1}-1)}(y)*T_{k*2^{m-1}}(y) - T_k(y)
    tmpct2 = T2[1]
    T2km1 = T[5]
    tmpct2 = homo_add(tmpct2, tmpct2, cryptoContext)
    tmpct2 = homo_add_scalar_double(tmpct2, -1.0, cryptoContext)
    T2km1 = homo_mul(T2km1, tmpct2, cryptoContext)
    T2km1 = cipher_rescale(T2km1, cryptoContext)

    for i in range(2, 4):
        square = homo_square(T2[i - 1], cryptoContext)
        T2[i] = homo_add(square, square, cryptoContext)
        T2[i] = cipher_rescale(T2[i], cryptoContext)
        T2[i] = homo_add_scalar_double(T2[i], -1.0, cryptoContext)

        # compute T_{k(2*m - 1)} = 2*T_{k(2^{m-1}-1)}(y)*T_{k*2^{m-1}}(y) - T_k(y)
        tmpct2 = T2[i]
        T2km1 = homo_mul(T2km1, tmpct2, cryptoContext)
        T2km1 = homo_add(T2km1, T2km1, cryptoContext)
        T2km1 = cipher_rescale(T2km1, cryptoContext)
        tmpct2 = T[5]
        tmpct2 = cipher_level_reduce(tmpct2, i)
        T2km1 = homo_sub(T2km1, tmpct2, cryptoContext)

    qu = InnerL1(
        T,
        T2,
        cryptoContext,
        q_divcs_q,
        qs_divcs_q,
        qs_divqr_q,
        qq_divcs_q,
        qq_divqr_q,
        qq_s2,
        qs_s2,
        qs_lg2_divqr_q_last,
        qq_lg2_divqr_q_last,
    )
    su = InnerL1(
        T,
        T2,
        cryptoContext,
        s_divcs_q,
        ss_divcs_q,
        ss_divqr_q,
        sq_divcs_q,
        sq_divqr_q,
        sq_s2,
        ss_s2,
        ss_lg2_divqr_q_last,
        sq_lg2_divqr_q_last,
    )

    result = homo_mul_scalar_double(T[0], first_divcs_q[1], cryptoContext)
    for i in range(1, 5):
        tmp = homo_mul_scalar_double(T[i], first_divcs_q[i + 1], cryptoContext)
        result = homo_add(result, tmp, cryptoContext)

    result = cipher_level_reduce(result, 2)
    result = cipher_rescale(result, cryptoContext)

    result = homo_add_scalar_double(result, first_divcs_q[0], cryptoContext)
    result = homo_add(result, T2[3], cryptoContext)

    result = homo_mul(result, qu, cryptoContext)
    result = cipher_rescale(result, cryptoContext)
    su = cipher_level_reduce(su, 1)
    result = homo_add(result, su, cryptoContext)
    result = homo_sub(result, T2km1, cryptoContext)

    return result


def DoubleAngleIteration(in0, cryptoContext):
    r = int(R)
    scalar = [
        -0.94418452709144784,
        -0.89148442119890103,
        -0.79474447324033948,
        -0.63161877774606467,
        -0.3989422804014327,
        -0.15915494309189535,
    ]

    for j in range(1, r + 1):
        in0 = homo_square(in0, cryptoContext)
        in0 = cipher_mod_reduce(in0, 1, cryptoContext)
        in0 = homo_add(in0, in0, cryptoContext)
        # scalar = np.float64(np.float64(-1.0) / np.float64(math.pow((2.0 * M_PI), np.float64(math.pow(2.0, j - r)))))
        in0 = homo_add_scalar_double(in0, scalar[j - 1], cryptoContext)

    return in0
