from .context import *
from .bs_context import *
from . import homo_ops
import numpy as np

BASE_NUM_LEVELS_TO_DROP = 1


def eval_linear_wsum_mutable(ciphertexts, constants, cryptoContext: Context):
    input_size = len(constants)

    minLimbs = ciphertexts[0].cur_limbs
    minIdx = 0
    for i in range(1, input_size):
        if (
            ciphertexts[i].cur_limbs < minLimbs
            or (ciphertexts[i].cur_limbs == minLimbs)
            and ciphertexts[i].noise_deg == 2
        ):
            minLimbs = ciphertexts[i].cur_limbs
            minIdx = i
    for i in range(minIdx):
        ciphertexts[i], ciphertexts[minIdx] = homo_ops.adjust_levels_and_depth(
            ciphertexts[i], ciphertexts[minIdx], cryptoContext
        )
    for i in range(minIdx + 1, input_size):
        ciphertexts[i], ciphertexts[minIdx] = homo_ops.adjust_levels_and_depth(
            ciphertexts[i], ciphertexts[minIdx], cryptoContext
        )

    if ciphertexts[minIdx].noise_deg == 2:
        for i in range(0, input_size):
            ciphertexts[i] = homo_ops.homo_rescale_internal(
                ciphertexts[i], BASE_NUM_LEVELS_TO_DROP, cryptoContext
            )

    wsum = homo_ops.homo_mul_scalar_double(ciphertexts[0], constants[0], cryptoContext)
    for i in range(1, input_size):
        tmp = homo_ops.homo_mul_scalar_double(
            ciphertexts[i], constants[i], cryptoContext
        )
        wsum = homo_ops.homo_add(wsum, tmp, cryptoContext)
    return wsum


def degree(lst):
    for i in range(len(lst) - 1, -1, -1):
        if lst[i] != 0:
            return i
    return 0


def long_division_chebyshev(f, g):
    assert (not math.isclose(f[-1], 0)) and (not math.isclose(g[-1], 0))
    n, k = len(f) - 1, len(g) - 1

    if n < k:
        return np.array([1.0]), np.array(f)

    q = np.zeros(n - k + 1)
    r = np.copy(f)
    d = np.zeros(len(g) + n)

    while n > k:
        q[n - k] = 2 * r[-1] / g[-1]
        d = np.zeros(n + 1)
        if k == (n - k):
            d[0] = 2 * g[n - k]
            for i in range(1, 2 * k + 1):
                d[i] = g[abs(n - k - i)]
        else:
            if k > (n - k):
                d[0] = 2 * g[n - k]
                for i in range(1, k - (n - k) + 1):
                    d[i] = g[abs(n - k - i)] + g[n - k + i]
                for i in range(k - (n - k) + 1, n + 1):
                    d[i] = g[abs(i - n + k)]
            else:
                d[n - k] = g[0]
                for i in range(n - 2 * k, n + 1):
                    if i != n - k:
                        d[i] = g[abs(i - n + k)]

        r = r - d * r[-1] / g[-1]
        if len(r) > 1:
            n = degree(r)
            r.resize(n + 1, refcheck=False)

    if n == k:
        q[0] = r[-1] / g[-1]
        r = r - g * q[0]
        if len(r) > 1:
            n = degree(r)
            r.resize(n + 1, refcheck=False)

    q[0] *= 2
    return q, r


def inner_eval_chebyshev_ps(coefficients, k, m, T, T2, cryptoContext: Context):

    k2m2k = k * (1 << (m - 1)) - k

    Tkm = np.zeros(int(k2m2k + k) + 1)
    Tkm[-1] = 1.0
    divqr_q, divqr_r = long_division_chebyshev(coefficients, Tkm)

    r2 = np.copy(divqr_r)
    if int(k2m2k - degree(divqr_r)) <= 0:
        r2[k2m2k] -= 1
        r2.resize(degree(r2) + 1, refcheck=False)
    else:
        r2.resize(k2m2k + 1, refcheck=False)
        r2[-1] = -1

    divcs_q, divcs_r = long_division_chebyshev(r2, divqr_q)

    s2 = np.copy(divcs_r)
    s2.resize(k2m2k + 1, refcheck=False)
    s2[-1] = 1.0

    dc = degree(divcs_q)
    flag_c = False
    if dc >= 1:
        if dc == 1:
            if divcs_q[1] != 1:
                cu = homo_ops.homo_mul_scalar_double(T[0], divcs_q[1], cryptoContext)
            else:
                cu = T[0]
        else:
            ctxs = [T[i] for i in range(dc)]
            weights = divcs_q[1 : dc + 1]
            cu = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)

        cu = homo_ops.homo_add_scalar_double(cu, divcs_q[0] / 2, cryptoContext)
        flag_c = True

    if degree(divqr_q) > k:
        qu = inner_eval_chebyshev_ps(divqr_q, k, m - 1, T, T2, cryptoContext)
    else:
        qcopy = np.copy(divqr_q)
        qcopy.resize(k, refcheck=False)
        deg_qcopy = degree(qcopy)
        if deg_qcopy > 0:
            ctxs = [T[i] for i in range(deg_qcopy)]
            weights = divqr_q[1 : deg_qcopy + 1]
            qu = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)
            sum = T[k - 1]
            divqr_q[-1] += 1.1
            sum = homo_ops.homo_mul_scalar_int(
                T[k - 1], 2 ** math.floor(math.log2(divqr_q[-1])), cryptoContext
            )
            qu = homo_ops.homo_add(qu, sum, cryptoContext)
        else:
            sum = T[k - 1]
            sum = homo_ops.homo_mul_scalar_int(
                T[k - 1], 2 ** math.floor(math.log2(divqr_q[-1])), cryptoContext
            )
            qu = sum

        qu = homo_ops.homo_add_scalar_double(qu, divqr_q[0] / 2, cryptoContext)

    if degree(s2) > k:
        su = inner_eval_chebyshev_ps(s2, k, m - 1, T, T2, cryptoContext)
    else:
        scopy = np.copy(s2)
        scopy.resize(k, refcheck=False)
        deg_scopy = degree(scopy)
        if deg_scopy > 0:
            ctxs = [T[i] for i in range(deg_scopy)]
            weights = s2[1 : deg_scopy + 1]
            su = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)
            su = homo_ops.homo_add(su, T[k - 1], cryptoContext)
        else:
            su = T[k - 1]

        su = homo_ops.homo_add_scalar_double(su, s2[0] / 2, cryptoContext)

    if flag_c:
        result = homo_ops.homo_add(T2[m - 1], cu, cryptoContext)
    else:
        result = homo_ops.homo_add_scalar_double(
            T2[m - 1], divcs_q[0] / 2, cryptoContext
        )

    result = homo_ops.homo_mul(result, qu, cryptoContext)
    result = homo_ops.homo_add(result, su, cryptoContext)

    return result


def PopulateParameterPS(upper_bound_degree):
    mlist = np.zeros(upper_bound_degree, dtype=np.int32)

    ranges = [
        (1, 2, 1),
        (3, 11, 2),
        (12, 13, 3),
        (14, 17, 2),
        (18, 55, 3),
        (56, 59, 4),
        (60, 76, 3),
        (77, 239, 4),
        (240, 247, 5),
        (248, 284, 4),
        (285, 991, 5),
        (992, 1007, 6),
        (1008, 1083, 5),
        (1084, 2015, 6),
        (2016, 2031, 7),
        (2032, 2204, 6),
    ]

    for start, end, m in ranges:
        if upper_bound_degree < start:
            break
        actual_end = min(end, upper_bound_degree)
        mlist[start - 1 : actual_end] = m

    return mlist


def ComputeDegreesPS(n):
    if n == 0:
        raise ValueError(
            "ComputeDegreesPS: The degree is zero. There is no need to evaluate the polynomial."
        )

    UPPER_BOUND_PS = 2204

    if n <= UPPER_BOUND_PS:
        mlist = PopulateParameterPS(UPPER_BOUND_PS)
        m = mlist[n - 1]
        k = math.floor(n / ((1 << m) - 1)) + 1
        return [k, m]
    else:
        klist = []
        mlist = []
        multlist = []

        sqrt_half_n = math.sqrt(n / 2)
        floor_log2_sqrt_half_n = (
            math.floor(math.log2(sqrt_half_n)) if sqrt_half_n > 0 else 0
        )

        for k in range(1, n + 1):
            max_m = math.ceil(math.log2(n / k) + 1) + 1
            for m in range(1, int(max_m) + 1):
                lhs = n
                rhs = k * ((1 << m) - 1)
                if lhs - rhs < 0:
                    floor_log2_k = math.floor(math.log2(k))
                    if abs(floor_log2_k - floor_log2_sqrt_half_n) <= 1:
                        klist.append(k)
                        mlist.append(m)
                        mult = k + 2 * m + (1 << (m - 1)) - 4
                        multlist.append(mult)

        if not multlist:
            raise ValueError("No valid (k, m) pairs found for the given n.")

        min_mult = min(multlist)
        min_index = multlist.index(min_mult)

        return [klist[min_index], mlist[min_index]]


def eval_chebyshev_series_ps(x, coefficients, a, b, cryptoContext):

    n = degree(coefficients)
    f2 = np.copy(coefficients)
    if coefficients[-1] == 0:
        f2.resize(n + 1, refcheck=False)

    degs = ComputeDegreesPS(n)
    k = degs[0]
    m = degs[1]

    k2m2k = k * (1 << (m - 1)) - k

    f2.resize(2 * k2m2k + k + 1, refcheck=False)
    f2[-1] = 1
    Tkm = np.zeros(k2m2k + k + 1)
    Tkm[-1] = 1

    divqr_q, divqr_r = long_division_chebyshev(f2, Tkm)

    r2 = np.copy(divqr_r)
    if k2m2k - degree(divqr_r) <= 0:
        r2[k2m2k] -= 1
        r2.resize(degree(r2) + 1, refcheck=False)
    else:
        r2.resize(k2m2k + 1, refcheck=False)
        r2[-1] = -1

    divcs_q, divcs_r = long_division_chebyshev(r2, divqr_q)

    s2 = np.copy(divcs_r)
    s2.resize(k2m2k + 1, refcheck=False)
    s2[-1] = 1

    cu = None

    T = [x]
    alpha = 2 / (b - a)
    if not math.isclose(alpha, 1.0):
        T[0] = homo_ops.homo_mul_scalar_double(x, alpha, cryptoContext)
    beta = 2 * a / (b - a)
    if not math.isclose(beta, -1.0):
        T[0] = homo_ops.homo_add_scalar_double(T[0], -1.0 - beta, cryptoContext)

    for i in range(2, k + 1):
        prod = homo_ops.homo_mul(T[i // 2 - 1], T[(i + 1) // 2 - 1], cryptoContext)
        tmp = homo_ops.homo_add(prod, prod, cryptoContext)
        if i & 1 == 1:
            tmp = homo_ops.homo_sub(tmp, T[0], cryptoContext)
        else:
            tmp = homo_ops.homo_add_scalar_double(tmp, -1.0, cryptoContext)
        T.append(tmp)

    for i in range(1, k):
        T[i - 1], T[k - 1] = homo_ops.adjust_levels_and_depth(
            T[i - 1], T[k - 1], cryptoContext
        )

    T2 = [T[-1]]
    for i in range(1, m):
        tmp = homo_ops.homo_square(T2[i - 1], cryptoContext)
        tmp = homo_ops.homo_add(tmp, tmp, cryptoContext)
        tmp = homo_ops.homo_add_scalar_double(tmp, -1.0, cryptoContext)
        T2.append(tmp)

    T2km1 = T2[0]
    for i in range(1, m):
        prod = homo_ops.homo_mul(T2km1, T2[i], cryptoContext)
        T2km1 = homo_ops.homo_add(prod, prod, cryptoContext)
        T2km1 = homo_ops.homo_sub(T2km1, T2[0], cryptoContext)

    dc = degree(divcs_q)
    flag_c = False
    if dc >= 1:
        if dc == 1:
            if divcs_q[1] != 1:
                cu = homo_ops.homo_mul_scalar_double(T[0], divcs_q[1], cryptoContext)
            else:
                cu = T[0]
        else:
            ctxs = [T[i] for i in range(dc)]
            weights = divcs_q[1 : dc + 1]
            cu = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)

        cu = homo_ops.homo_add_scalar_double(cu, divcs_q[0] / 2, cryptoContext)
        flag_c = True

    if degree(divqr_q) > k:
        qu = inner_eval_chebyshev_ps(divqr_q, k, m - 1, T, T2, cryptoContext)
    else:
        q_copy = np.copy(divqr_q[:k])
        deg_qcopy = degree(q_copy)
        if deg_qcopy > 0:
            ctxs = [T[i] for i in range(deg_qcopy)]
            weights = divqr_q[1 : deg_qcopy + 1]
            qu = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)
            sum = homo_ops.homo_add(T[k - 1], T[k - 1], cryptoContext)
            qu = homo_ops.homo_add(qu, sum, cryptoContext)
        else:
            qu = T[k - 1]
            for _ in range(1, divqr_q[-1]):
                qu = homo_ops.homo_add(qu, T[k - 1], cryptoContext)

        qu = homo_ops.homo_add_scalar_double(qu, divqr_q[0] / 2, cryptoContext)

    deg_s2 = degree(s2)
    if deg_s2 > k:
        su = inner_eval_chebyshev_ps(s2, k, m - 1, T, T2, cryptoContext)
    else:
        scopy = np.copy(s2[:k])
        deg_scopy = degree(scopy)
        if deg_scopy > 0:
            ctxs = [T[i] for i in range(deg_scopy)]
            weights = s2[1 : deg_scopy + 1]
            su = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)
            su = homo_ops.homo_add(su, T[k - 1], cryptoContext)
        else:
            su = T[k - 1]
        su = homo_ops.homo_add_scalar_double(su, s2[0] / 2, cryptoContext)

    if flag_c:
        result = homo_ops.homo_add(T2[m - 1], cu, cryptoContext)
    else:
        result = homo_ops.homo_add_scalar_double(
            T2[m - 1], divcs_q[0] / 2, cryptoContext
        )

    result = homo_ops.homo_mul(result, qu, cryptoContext)
    result = homo_ops.homo_add(result, su, cryptoContext)
    result = homo_ops.homo_sub(result, T2km1, cryptoContext)
    return result


def eval_chebyshev_coefficients(func, a, b, degree):
    if degree == 0:
        raise ValueError("The degree of approximation cannot be zero")

    coeff_total = degree + 1
    b_minus_a = 0.5 * (b - a)
    b_plus_a = 0.5 * (b + a)
    pi_by_deg = math.pi / coeff_total

    function_points = [
        func(math.cos(pi_by_deg * (i + 0.5)) * b_minus_a + b_plus_a)
        for i in range(coeff_total)
    ]

    mult_factor = 2.0 / coeff_total
    coefficients = [0.0] * coeff_total
    for i in range(coeff_total):
        for j in range(coeff_total):
            coefficients[i] += function_points[j] * math.cos(pi_by_deg * i * (j + 0.5))
        coefficients[i] *= mult_factor

    return coefficients
