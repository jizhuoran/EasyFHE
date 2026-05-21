import math
import easyfhe.fhe as fhe
import numpy as np


BASE_NUM_LEVELS_TO_DROP = 1 #todo: to be removed, or move to cryptoContext


def _uses_manual_rescale(cryptoContext):
    return cryptoContext.scale_mode == "fixed" and cryptoContext.rescale_policy == "manual"


def get_relu_depth(degree):
    ranges = [
        (1, 5, 3),
        (6, 13, 4),
        (14, 27, 5),
        (28, 59, 6),
        (60, 119, 7),
        (120, 247, 8),
        (248, 495, 9),
        (496, 1007, 10),
        (1008, 2031, 11)
    ]

    for lower, upper, depth in ranges:
        if lower <= degree <= upper:
            return depth

    raise ValueError("Set a valid degree for ReLU")


def eval_linear_wsum_mutable(ciphertexts, constants, cryptoContext):
    if not _uses_manual_rescale(cryptoContext):
        target_idx = min(range(len(ciphertexts)), key=lambda i: ciphertexts[i].cur_limbs - ciphertexts[i].noise_deg)
        if ciphertexts[target_idx].noise_deg == 2:
            ciphertexts[target_idx] = fhe.align_to(ciphertexts[target_idx], fhe.CipherState(ciphertexts[target_idx].cur_limbs - (1), ciphertexts[target_idx].noise_deg - (1)), cryptoContext)
        for i in range(len(ciphertexts)):
            ciphertexts[i] = fhe.align_to(ciphertexts[i], fhe.CipherState(ciphertexts[target_idx].cur_limbs, ciphertexts[target_idx].noise_deg, ciphertexts[target_idx].scaling_factor), cryptoContext)

    wsum = fhe.homo_mul_scalar_double(ciphertexts[0], constants[0], cryptoContext)
    for i in range(1, len(constants)):
        tmp = fhe.homo_mul_scalar_double(ciphertexts[i], constants[i], cryptoContext)
        wsum = fhe.homo_add(wsum, tmp, cryptoContext)
    wsum = fhe.align_to(wsum, fhe.CipherState(wsum.cur_limbs - (BASE_NUM_LEVELS_TO_DROP), wsum.noise_deg - (BASE_NUM_LEVELS_TO_DROP)), cryptoContext)
    return wsum


def degree(lst):
    for i in range(len(lst) - 1, -1, -1):
        if lst[i] != 0: return i
    return 0  # All elements are zero

# f and g are vectors of Chebyshev interpolation coefficients of the two polynomials.
# We assume their dominant coefficient is not zero. LongDivisionChebyshev returns the
# vector of Chebyshev interpolation coefficients for the quotient and remainder of the
# division f/g. longDiv is a struct that contains the vectors of coefficients for the
# quotient and rest. We assume that the zero-th coefficient is c0, not c0/2 and returns
# the same format.

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

def inner_eval_chebyshev_ps(coefficients,
                            k, m, T, T2, cryptoContext):
    # Compute k * 2^(m-1) - k
    k2m2k = k * (1 << (m - 1)) - k

    # Divide coefficients by T^{k*2^{m-1}}
    Tkm = np.zeros(int(k2m2k + k) + 1)
    Tkm[-1] = 1.0  # Tkm.back() = 1
    divqr_q, divqr_r = long_division_chebyshev(coefficients, Tkm)

    # Subtract x^(k(2^(m-1) - 1)) from r
    r2 = np.copy(divqr_r)
    if (int(k2m2k - degree(divqr_r)) <= 0):
        r2[k2m2k] -= 1
        r2.resize(degree(r2) + 1, refcheck=False)
    else:
        r2.resize(k2m2k + 1, refcheck=False)
        r2[-1] = -1

    # Divide r2 by q
    divcs_q, divcs_r = long_division_chebyshev(r2, divqr_q)

    # Add x^(k(2^(m-1) - 1)) to s
    s2 = np.copy(divcs_r)
    s2.resize(k2m2k + 1, refcheck=False)
    s2[-1] = 1.0

    # Evaluate c at u
    dc = degree(divcs_q)
    flag_c = False
    if dc >= 1:
        if dc == 1:
            if divcs_q[1] != 1:
                cu = fhe.homo_mul_scalar_double(T[0], divcs_q[1], cryptoContext)
                cu = fhe.align_to(cu, fhe.CipherState(cu.cur_limbs - (BASE_NUM_LEVELS_TO_DROP), cu.noise_deg - (BASE_NUM_LEVELS_TO_DROP)), cryptoContext)
            else:
                cu = T[0]
        else:
            ctxs = [T[i] for i in range(dc)]
            weights = divcs_q[1:dc + 1]
            cu = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)

        # adds the free term (at x^0)
        cu = fhe.homo_add_scalar_double(cu, divcs_q[0] / 2, cryptoContext)
        # Need to reduce levels up to the level of T2[m-1].
        if _uses_manual_rescale(cryptoContext):
            cu = fhe.align_to(cu, fhe.CipherState(T2[m - 1].cur_limbs, T2[m - 1].noise_deg, T2[m - 1].scaling_factor), cryptoContext)
        flag_c = True

    # Evaluate q and s2 at u
    if degree(divqr_q) > k:
        qu = inner_eval_chebyshev_ps(divqr_q, k, m - 1, T, T2, cryptoContext)
    else:
        qcopy = np.copy(divqr_q)
        qcopy.resize(k, refcheck=False)
        deg_qcopy = degree(qcopy)
        if deg_qcopy > 0:
            ctxs = [T[i] for i in range(deg_qcopy)]
            weights = divqr_q[1:deg_qcopy + 1]
            qu = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)
            sum = T[k - 1]
            divqr_q[-1] += 1.1
            sum = fhe.homo_mul_scalar_int(T[k - 1], 2 ** math.floor(math.log2(divqr_q[-1])), cryptoContext)
            # for i in range(int(math.log2(divqr_q[-1]))):
            # sum = fhe.homo_add(sum, sum, cryptoContext)
            qu = fhe.homo_add(qu, sum, cryptoContext)
        else:
            sum = T[k - 1]
            sum = fhe.homo_mul_scalar_int(T[k - 1], 2 ** math.floor(math.log2(divqr_q[-1])), cryptoContext)
            # for i in range(int(math.log2(divqr_q[-1]))):
            # sum = fhe.homo_add(sum, sum, cryptoContext)
            qu = sum

        qu = fhe.homo_add_scalar_double(qu, divqr_q[0] / 2, cryptoContext)

    # Evaluate s2 at u
    if degree(s2) > k:
        su = inner_eval_chebyshev_ps(s2, k, m - 1, T, T2, cryptoContext)
    else:
        scopy = np.copy(s2)
        scopy.resize(k, refcheck=False)
        deg_scopy = degree(scopy)
        if deg_scopy > 0:
            ctxs = [T[i] for i in range(deg_scopy)]
            weights = s2[1:deg_scopy + 1]
            su = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)
            su = fhe.homo_add(su, T[k - 1], cryptoContext)
        else:
            su = T[k - 1]

        su = fhe.homo_add_scalar_double(su, s2[0] / 2, cryptoContext)
        if _uses_manual_rescale(cryptoContext):
            su = fhe.align_to(su, fhe.CipherState(su.cur_limbs - 1, 1, None), cryptoContext)

    if flag_c:
        result = fhe.homo_add(T2[m - 1], cu, cryptoContext)
    else:
        result = fhe.homo_add_scalar_double(T2[m - 1], divcs_q[0] / 2, cryptoContext)

    result = fhe.homo_mul(result, qu, cryptoContext)
    result = fhe.align_to(result, fhe.CipherState(result.cur_limbs - (BASE_NUM_LEVELS_TO_DROP), result.noise_deg - (BASE_NUM_LEVELS_TO_DROP)), cryptoContext)
    result = fhe.homo_add(result, su, cryptoContext)

    return result


def PopulateParameterPS(upper_bound_degree):
    # Initialize the mlist array with zeros
    mlist = np.zeros(upper_bound_degree, dtype=np.int32)

    # Define the degree ranges and corresponding m values
    # Each tuple is (start_n, end_n, m)
    ranges = [
        (1, 2, 1),  # n in [1,2], m = 1
        (3, 11, 2),  # n in [3,11], m = 2
        (12, 13, 3),  # n in [12,13], m = 3
        (14, 17, 2),  # n in [14,17], m = 2
        (18, 55, 3),  # n in [18,55], m = 3
        (56, 59, 4),  # n in [56,59], m = 4
        (60, 76, 3),  # n in [60,76], m = 3
        (77, 239, 4),  # n in [77,239], m = 4
        (240, 247, 5),  # n in [240,247], m = 5
        (248, 284, 4),  # n in [248,284], m = 4
        (285, 991, 5),  # n in [285,991], m = 5
        (992, 1007, 6),  # n in [992,1007], m = 6
        (1008, 1083, 5),  # n in [1008,1083], m = 5
        (1084, 2015, 6),  # n in [1084,2015], m = 6
        (2016, 2031, 7),  # n in [2016,2031], m = 7
        (2032, 2204, 6)  # n in [2032,2204], m = 6
    ]

    for start, end, m in ranges:
        if upper_bound_degree < start:
            # If the upper bound is less than the start of the current range, no need to continue
            break
        # Determine the actual end for slicing to avoid exceeding upper_bound_degree
        actual_end = min(end, upper_bound_degree)
        # Set the value m for the slice [start-1, actual_end)
        # In Python, slicing is end-exclusive
        mlist[start - 1: actual_end] = m

    return mlist


# Compute positive integers k,m such that n < k(2^m-1), k is close to sqrt(n/2)
# and the depth = ceil(log2(k))+m is minimized. Moreover, for that depth the
# number of homomorphic multiplications = k+2m+2^(m-1)-4 is minimized.
# Since finding these parameters involve testing many possible values, we
# hardcode them for commonly used degrees, and provide a heuristic which
# minimizes the number of homomorphic multiplications for the rest of the
# degrees.
def ComputeDegreesPS(n):
    if n == 0:
        raise ValueError("ComputeDegreesPS: The degree is zero. There is no need to evaluate the polynomial.")

    UPPER_BOUND_PS = 2204

    # Index n-1 in the list corresponds to degree n
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
        floor_log2_sqrt_half_n = math.floor(math.log2(sqrt_half_n)) if sqrt_half_n > 0 else 0

        for k in range(1, n + 1):
            # Calculate the upper bound for m to avoid excessive iterations
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



# note: EvalChebyshevSeriesPS in ckksrns-advancedshe.cpp
# @profile_pytorch_function
def eval_chebyshev_series_ps(x, coefficients, a, b, cryptoContext):

    n = degree(coefficients)
    f2 = np.copy(coefficients)
    # Make sure the coefficients do not have the zero dominant terms
    if coefficients[- 1] == 0:
        f2.resize(n + 1, refcheck=False)

    degs = ComputeDegreesPS(n)
    k = degs[0]
    m = degs[1]

    # Compute k*2^{m-1}-k because we use it a lot
    k2m2k = k * (1 << (m - 1)) - k

    f2.resize(2 * k2m2k + k + 1, refcheck=False)
    f2[-1] = 1

    # Divide f2 by T^{k*2^{m-1}}
    Tkm = np.zeros(k2m2k + k + 1)
    Tkm[- 1] = 1

    divqr_q, divqr_r = long_division_chebyshev(f2, Tkm)

    r2 = np.copy(divqr_r)
    if k2m2k - degree(divqr_r) <= 0:
        r2[k2m2k] -= 1
        r2.resize(degree(r2) + 1, refcheck=False)
    else:
        r2.resize(k2m2k + 1, refcheck=False)
        r2[-1] = -1

    # Divide r2 by q
    divcs_q, divcs_r = long_division_chebyshev(r2, divqr_q)

    # Add x^{k(2^{m-1} - 1)} to s
    s2 = np.copy(divcs_r)
    s2.resize(k2m2k + 1, refcheck=False)
    s2[-1] = 1

    # Evaluate c at u
    cu = None

    # computes linear transformation y = -1 + 2 (x-a)/(b-a)
    # consumes one level when a <> -1 && b <> 1

    T = [x]
    alpha = 2 / (b - a)
    if not math.isclose(alpha, 1.0):
        T[0] = fhe.homo_mul_scalar_double(x, alpha, cryptoContext)
        T[0] = fhe.align_to(T[0], fhe.CipherState(T[0].cur_limbs - (1), T[0].noise_deg - (1)), cryptoContext)
    beta = 2 * a / (b - a)
    if not math.isclose(beta, -1.0):
        T[0] = fhe.homo_add_scalar_double(T[0], -1.0 - beta, cryptoContext)

    for i in range(2, k + 1):
        prod = fhe.homo_mul(T[i // 2 - 1], T[(i + 1) // 2 - 1], cryptoContext)
        tmp = fhe.homo_add(prod, prod, cryptoContext)
        tmp = fhe.align_to(tmp, fhe.CipherState(tmp.cur_limbs - (1), tmp.noise_deg - (1)), cryptoContext)
        if i & 1 == 1:  # i is odd
            tmp = fhe.homo_sub(tmp, T[0], cryptoContext)
        else:
            tmp = fhe.homo_add_scalar_double(tmp, -1.0, cryptoContext)
        T.append(tmp)

    assert T[-1].cur_limbs == min(
        t.cur_limbs for t in T), "T[-1].cur_limbs is not the minimum; it should be adjusted to the minimum value."
    for i in range(k):
        T[i] = fhe.align_to(T[i], fhe.CipherState(T[-1].cur_limbs, T[-1].noise_deg, T[-1].scaling_factor), cryptoContext)

    # Compute the Chebyshev polynomials T_k(y), T_{2k}(y), T_{4k}(y), ... , T_{2^{m-1}k}(y)
    # T2[0] is used as a placeholder
    T2 = [T[-1]]
    for i in range(1, m):
        tmp = fhe.homo_square(T2[i - 1], cryptoContext)
        tmp = fhe.homo_add(tmp, tmp, cryptoContext)
        tmp = fhe.align_to(tmp, fhe.CipherState(tmp.cur_limbs - (1), tmp.noise_deg - (1)), cryptoContext)
        tmp = fhe.homo_add_scalar_double(tmp, -1.0, cryptoContext)
        T2.append(tmp)



    # computes T_{k(2*m - 1)}(y)
    T2km1 = T2[0]
    for i in range(1, m):
        # compute T_{k(2*m - 1)} = 2*T_{k(2^{m-1}-1)}(y)*T_{k*2^{m-1}}(y) - T_k(y)
        prod = fhe.homo_mul(T2km1, T2[i], cryptoContext)
        T2km1 = fhe.homo_add(prod, prod, cryptoContext)
        T2km1 = fhe.align_to(T2km1, fhe.CipherState(T2km1.cur_limbs - (1), T2km1.noise_deg - (1)), cryptoContext)
        T2km1 = fhe.homo_sub(T2km1, T2[0], cryptoContext)



    dc = degree(divcs_q)
    flag_c = False
    if dc >= 1:
        if dc == 1:
            if divcs_q[1] != 1:
                cu = fhe.homo_mul_scalar_double(T[0], divcs_q[1], cryptoContext)
                cu = fhe.align_to(cu, fhe.CipherState(cu.cur_limbs - (1), cu.noise_deg - (1)), cryptoContext)
            else:
                cu = T[0]
        else:
            ctxs = [T[i] for i in range(dc)]
            weights = divcs_q[1:dc + 1]
            cu = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)

        # adds the free term (at x^0)
        cu = fhe.homo_add_scalar_double(cu, divcs_q[0] / 2, cryptoContext)
        flag_c = True



    # Evaluate q and s2 at u. If their degrees are larger than k, then recursively apply the Paterson-Stockmeyer algorithm.
    if degree(divqr_q) > k:
        qu = inner_eval_chebyshev_ps(divqr_q, k, m - 1, T, T2, cryptoContext)
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
            sum = fhe.homo_add(T[k - 1], T[k - 1], cryptoContext)
            qu = fhe.homo_add(qu, sum, cryptoContext)
        else:
            qu = T[k - 1]
            for _ in range(1, divqr_q[- 1]):
                qu = fhe.homo_add(qu, T[k - 1], cryptoContext)

        # adds the free term (at x^0)
        qu = fhe.homo_add_scalar_double(qu, divqr_q[0] / 2, cryptoContext)
        # The number of levels of qu is the same as the number of levels of T[k-1] + 1.
        # Will only get here when m = 2, so the number of levels of qu and T2[m-1] will be the same.

    # Evaluate s2 at u
    deg_s2 = degree(s2)
    if deg_s2 > k:
        su = inner_eval_chebyshev_ps(s2, k, m - 1, T, T2, cryptoContext)
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
            su = fhe.homo_add(su, T[k - 1], cryptoContext)
        else:
            su = T[k - 1]
        # adds the free term (at x^0)
        su = fhe.homo_add_scalar_double(su, s2[0] / 2, cryptoContext)
        # The number of levels of su is the same as the number of levels of T[k-1] + 1.
        # Will only get here when m = 2, so need to reduce the number of levels by 1.

    if flag_c:
        result = fhe.homo_add(T2[m - 1], cu, cryptoContext)
    else:
        result = fhe.homo_add_scalar_double(T2[m - 1], divcs_q[0] / 2, cryptoContext)



    result = fhe.homo_mul(result, qu, cryptoContext)
    result = fhe.align_to(result, fhe.CipherState(result.cur_limbs - (1), result.noise_deg - (1)), cryptoContext)
    result = fhe.homo_add(result, su, cryptoContext)


    result = fhe.homo_sub(result, T2km1, cryptoContext)



    return result

def eval_chebyshev_coefficients(func, a, b, degree):
    if degree == 0:
        raise ValueError("The degree of approximation cannot be zero")

    # The number of coefficients to be generated should be degree + 1 as zero is also included
    coeff_total = degree + 1
    b_minus_a = 0.5 * (b - a)
    b_plus_a = 0.5 * (b + a)
    pi_by_deg = math.pi / coeff_total

    # Calculate function points
    function_points = [
        func(math.cos(pi_by_deg * (i + 0.5)) * b_minus_a + b_plus_a)
        for i in range(coeff_total)
    ]

    # Calculate coefficients
    mult_factor = 2.0 / coeff_total
    coefficients = [0.0] * coeff_total
    for i in range(coeff_total):
        for j in range(coeff_total):
            coefficients[i] += function_points[j] * math.cos(pi_by_deg * i * (j + 0.5))
        coefficients[i] *= mult_factor

    return coefficients


def eval_chebyshev_function(function, ciphertext, lowerBound, upperBound, poly_degree, cryptoContext):
    coefficients = eval_chebyshev_coefficients(
        function, lowerBound, upperBound, poly_degree)
    result = eval_chebyshev_series_ps(
        ciphertext, coefficients, lowerBound, upperBound, cryptoContext
    )
    return result

def long_division_poly(f, g):
    assert not math.isclose(f[-1], 0.0)
    assert not math.isclose(g[-1], 0.0)

    n = degree(f)
    k = degree(g)
    if n != len(f) - 1 or k != len(g) - 1:
        raise ValueError("Leading coefficient is zero")

    if n < k:
        return np.array([1.0]), np.array(f)

    q = np.zeros(n - k + 1)
    r = np.copy(f)

    while n >= k:
        coeff = r[-1] / g[-1] if not math.isclose(g[-1], 1.0) else r[-1]
        q[n - k] = coeff

        d = np.zeros(n - k)
        d = np.concatenate([d, g])
        d *= coeff

        r = r - d
        if len(r) > 1:
            n = degree(r)
            r = r[:n + 1]

    return q, r



def eval_poly(x, coefficients, cryptoContext):
    n = degree(coefficients)

    assert n >= 5, "EvalPolyLinear(x, coefficients); for n<5 is to be implemented"
    # todo: EvalPolyLinear(x, coefficients); to be implemented

    return eval_poly_ps(x, coefficients, cryptoContext)

def eval_poly_ps(x, coefficients, cryptoContext):
    n = degree(coefficients)
    f2 = np.copy(coefficients)

    # Make sure the coefficients do not have the dominant terms zero
    if coefficients[-1] == 0:
        f2.resize(n + 1, refcheck=False)

    k, m = ComputeDegreesPS(n)

    # set the indices for the powers of x that need to be computed to 1
    indices = [0] * k
    for i in range(k, 0, -1):
        if not (i & (i - 1)): # if i is a power of 2
            indices[i - 1] = 1
        else: # non-power of 2
            indices[i - 1] = 1
            powerOf2 = 1 << int(np.floor(np.log2(i)))
            rem = i % powerOf2
            if indices[rem - 1] == 0:
                indices[rem - 1] = 1
            # while rem is not a power of 2
            # set indices required to compute rem to 1
            while rem & (rem - 1):
                powerOf2 = 1 << int(np.floor(np.log2(rem)))
                rem = rem % powerOf2
                if indices[rem - 1] == 0:
                    indices[rem - 1] = 1

    # computes all powers up to k for x
    powers = [x.deep_copy()]
    for i in range(2, k + 1):
        if not (i & (i - 1)): # if i is a power of two
            t = fhe.homo_square(powers[i // 2 - 1], cryptoContext)
            t = fhe.align_to(t, fhe.CipherState(t.cur_limbs - (1), t.noise_deg - (1)), cryptoContext)
            powers.append(t)
        elif indices[i - 1] == 1: # non-power of 2
            powerOf2 = 1 << int(np.floor(np.log2(i)))
            rem = i % powerOf2
            t1 = powers[powerOf2 - 1]
            t2 = powers[rem - 1]
            t2 = fhe.align_to(t2, fhe.CipherState(t1.cur_limbs, t1.noise_deg, t1.scaling_factor), cryptoContext)
            t = fhe.homo_mul(t1, t2, cryptoContext)
            t = fhe.align_to(t, fhe.CipherState(t.cur_limbs - (1), t.noise_deg - (1)), cryptoContext)
            powers.append(t)
        else:
            pass
            # powers.append(None)

    # brings all powers of x to the same level
    for i in range(1, k):
        if indices[i - 1] == 1:
            powers[i - 1] = fhe.align_to(powers[i - 1], fhe.CipherState(powers[k - 1].cur_limbs, powers[k - 1].noise_deg, powers[k - 1].scaling_factor), cryptoContext)

    # computes powers of form k*2^i for x
    powers2 = [powers[k - 1].deep_copy()]
    for i in range(1, m):
        t = fhe.homo_square(powers2[i - 1], cryptoContext)
        t = fhe.align_to(t, fhe.CipherState(t.cur_limbs - (1), t.noise_deg - (1)), cryptoContext)
        powers2.append(t)

    # computes the product of the powers in power2, that yield x^{k(2*m - 1)}
    power2km1 = powers2[0].deep_copy()
    for i in range(1, m):
        tmp = fhe.homo_mul(power2km1, powers2[i], cryptoContext)
        tmp = fhe.align_to(tmp, fhe.CipherState(tmp.cur_limbs - (1), tmp.noise_deg - (1)), cryptoContext)
        power2km1 = tmp

    # Compute k*2^{m-1}-k because we use it a lot
    k2m2k = int(k * (1 << (m - 1)) - k)
    # Add x^{k(2^m - 1)} to the polynomial that has to be evaluated
    f2.resize(2 * k2m2k + k + 1, refcheck=False)
    f2[-1] = 1

    # Divide f2 by x^{k*2^{m-1}}
    xkm = np.zeros(k2m2k + k + 1)
    xkm[-1] = 1
    divqr_q, divqr_r = long_division_poly(f2, xkm)

    r2 = np.copy(divqr_r)
    if k2m2k - degree(divqr_r) <= 0:
        r2[k2m2k] -= 1
        r2.resize(degree(r2) + 1, refcheck=False)
    else:
        r2.resize(k2m2k + 1, refcheck=False)
        r2[-1] = -1

    divcs_q, divcs_r = long_division_poly(r2, divqr_q)

    s2 = np.copy(divcs_r)
    s2.resize(k2m2k + 1, refcheck=False)
    s2[-1] = 1

    dc = degree(divcs_q)
    flag_c = False
    if dc >= 1:
        if dc == 1:
            if divcs_q[1] != 1:
                cu = fhe.homo_mul_scalar_double(powers[0], divcs_q[1], cryptoContext)
                cu = fhe.align_to(cu, fhe.CipherState(cu.cur_limbs - (1), cu.noise_deg - (1)), cryptoContext)
            else:
                cu = powers[0].deep_copy()
        else:
            ctxs = [powers[i] for i in range(dc)]
            weights = divcs_q[1:dc + 1]
            cu = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)
        # adds the free term (at x^0)
        cu = fhe.homo_add_scalar_double(cu, divcs_q[0], cryptoContext)
        flag_c = True



    if degree(divqr_q) > k:
        qu = inner_eval_poly_ps(x, divqr_q, k, m - 1, powers, powers2, cryptoContext)
    else:
        qcopy = np.copy(divqr_q)
        qcopy.resize(k, refcheck=False)
        dq = degree(qcopy)
        if dq > 0:
            ctxs = [powers[i] for i in range(dq)]
            weights = divqr_q[1:dq + 1]
            qu = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)
            # the highest order term will always be 1 because q is monic
            qu = fhe.homo_add(qu, powers[k - 1], cryptoContext)
        else:
            qu = powers[k - 1].deep_copy()
        # adds the free term (at x^0)
        qu = fhe.homo_add_scalar_double(qu, divqr_q[0], cryptoContext)

    if np.allclose(s2, divqr_q):
        su = qu.deep_copy()
    elif degree(s2) > k:
        su = inner_eval_poly_ps(x, s2, k, m - 1, powers, powers2,cryptoContext)
    else:
        # ds = k from construction
        # perform scalar multiplication for all other terms and sum them up if there are non-zero coefficients
        scopy = np.copy(s2)
        scopy.resize(k, refcheck=False)
        ds = degree(scopy)
        if ds > 0:
            ctxs = [powers[i] for i in range(ds)]
            weights = s2[1:ds + 1]
            su = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)
            # the highest order term will always be 1 because q is monic
            su = fhe.homo_add(su, powers[k - 1], cryptoContext)
        else:
            su = powers[k - 1].deep_copy()

        # adds the free term (at x^0)
        su = fhe.homo_add_scalar_double(su, s2[0], cryptoContext)

    if flag_c:
        result = fhe.homo_add(powers2[m - 1], cu, cryptoContext)
    else:
        result = fhe.homo_add_scalar_double(powers2[m - 1], divcs_q[0], cryptoContext)

    result = fhe.homo_mul(result, qu, cryptoContext)
    result = fhe.align_to(result, fhe.CipherState(result.cur_limbs - (1), result.noise_deg - (1)), cryptoContext)
    result = fhe.homo_add(result, su, cryptoContext)
    result = fhe.homo_sub(result, power2km1, cryptoContext)

    return result


def inner_eval_poly_ps(x, coefficients, k, m, powers, powers2, cryptoContext):
    k2m2k = int(k * (1 << (m - 1)) - k)

    xkm = np.zeros(k2m2k + k + 1)
    xkm[-1] = 1
    divqr_q, divqr_r = long_division_poly(coefficients, xkm)

    # Subtract x^{k(2^{m-1} - 1)} from r
    r2 = np.copy(divqr_r)
    if k2m2k - degree(divqr_r) <= 0:
        r2[k2m2k] -= 1
        r2.resize(degree(r2) + 1, refcheck=False)
    else:
        r2.resize(k2m2k + 1, refcheck=False)
        r2[-1] = -1

    divcs_q, divcs_r = long_division_poly(r2, divqr_q)

    s2 = np.copy(divcs_r)
    s2.resize(k2m2k + 1, refcheck=False)
    s2[-1] = 1

    dc = degree(divcs_q)
    flag_c = False
    if dc >= 1:
        if dc == 1:
            if divcs_q[1] != 1:
                cu = fhe.homo_mul_scalar_double(powers[0], divcs_q[1], cryptoContext)
                cu = fhe.align_to(cu, fhe.CipherState(cu.cur_limbs - (1), cu.noise_deg - (1)), cryptoContext)
            else:
                cu = powers[0].deep_copy()
        else:
            ctxs = [powers[i] for i in range(dc)]
            weights = divcs_q[1:dc + 1]
            cu = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)

        # adds the free term (at x^0)
        cu = fhe.homo_add_scalar_double(cu, divcs_q[0], cryptoContext)
        if _uses_manual_rescale(cryptoContext):
            cu = fhe.align_to(cu, fhe.CipherState(powers2[m - 1].cur_limbs, powers2[m - 1].noise_deg, powers2[m - 1].scaling_factor), cryptoContext)
        flag_c = True

    # Evaluate q and s2 at u. If their degrees are larger than k, then recursively apply the Paterson-Stockmeyer algorithm.
    if degree(divqr_q) > k:
        qu = inner_eval_poly_ps(x, divqr_q, k, m - 1, powers, powers2, cryptoContext)
    else:
        # dq = k from construction
        # perform scalar multiplication for all other terms and sum them up if there are non-zero coefficients
        qcopy = np.copy(divqr_q)
        qcopy.resize(k, refcheck=False)
        dq = degree(qcopy)
        if dq > 0:
            ctxs = [powers[i] for i in range(dq)]
            weights = divqr_q[1:dq + 1]
            qu = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)
            # the highest order term will always be 1 because q is monic
            qu = fhe.homo_add(qu, powers[k - 1], cryptoContext)
        else:
            qu = powers[k - 1]

        # adds the free term (at x^0)
        qu = fhe.homo_add_scalar_double(qu, divqr_q[0], cryptoContext)

    if np.allclose(s2, divqr_q):
        su = qu.deep_copy()
    elif degree(s2) > k:
        su = inner_eval_poly_ps(x, s2, k, m - 1, powers, powers2, cryptoContext)
    else:
        # ds = k from construction
        # perform scalar multiplication for all other terms and sum them up if there are non-zero coefficients
        scopy = np.copy(s2)
        scopy.resize(k, refcheck=False)
        ds = degree(scopy)
        if ds > 0:
            ctxs = [powers[i] for i in range(ds)]
            weights = s2[1:ds + 1]
            su = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)
            su = fhe.homo_add(su, powers[k - 1], cryptoContext)
        else:
            su = powers[k - 1]
        su = fhe.homo_add_scalar_double(su, s2[0], cryptoContext)
        if _uses_manual_rescale(cryptoContext):
            su = fhe.align_to(su, fhe.CipherState(su.cur_limbs - 1, 1, None), cryptoContext)

    if flag_c:
        result = fhe.homo_add(powers2[m - 1], cu, cryptoContext)
    else:
        result = fhe.homo_add_scalar_double(powers2[m - 1], divcs_q[0], cryptoContext)

    result = fhe.homo_mul(result, qu, cryptoContext)
    result = fhe.align_to(result, fhe.CipherState(result.cur_limbs - (1), result.noise_deg - (1)), cryptoContext)
    result = fhe.homo_add(result, su, cryptoContext)

    return result
