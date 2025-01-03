import time, os
from .ciphertext import Cipher
from .ciphertext import Plaintext as Plaintext
from .client import client
from .context import *
from .bs_context import *
from . import functional as F
from . import homo_ops
from . import hoisting_keyswitch
from . import utils
import torch.profiler
from torch.profiler import ProfilerActivity, tensorboard_trace_handler

Tensor = torch.Tensor
NORMAL_CIPHER_SIZE = 2
BASE_NUM_LEVELS_TO_DROP = 1
R_UNIFORM = 6  # number of double-angle iterations in CKKS bootstrapping. Must be static because it is used in a static function.
R_SPARSE = 3  # number of double-angle iterations in CKKS bootstrapping. Must be static because it is used in a static function.
m_correctionFactor = (
    0  # correction factor, which we scale the message by to improve precision
)

# @profile_python_function
def adjust_ciphertext(ciphertext, correction, L0, cryptoContext):
    rescale_tech = cryptoContext.rescaleTech

    if rescale_tech == "FLEXIBLEAUTO" or rescale_tech == "FLEXIBLEAUTOEXT":
        lvl = 0 if rescale_tech == "FLEXIBLEAUTO" else 1
        if cryptoContext.L != L0:
            # Print error message and raise an exception to stop the program
            print("cryptoContext.L != L0")
            raise Exception("Error: cryptoContext.L != L0")
        target_sf = cryptoContext.GetScalingFactorReal(cur_limbs = (L0 - lvl))
        source_sf = ciphertext.scaling_factor
        num_towers = len(ciphertext.cv)
        mod_to_drop = float(cryptoContext.moduliQ[num_towers - 1])
        # in the case of FLEXIBLEAUTO, we need to bring the ciphertext to the right scale using a
        # a scaling multiplication. Note the at currently FLEXIBLEAUTO is only supported for NATIVEINT = 64.
        # So the other branch is for future purposes (in case we decide to add add the FLEXIBLEAUTO support
        # for NATIVEINT = 128.
        # Scaling down the message by a correction factor to emulate using a larger q0.
        # This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        adjustment_factor = (target_sf / source_sf) * (mod_to_drop / source_sf) * math.pow(2, -correction) # if NATIVEINT != 128
        ciphertext = homo_ops.homo_mul_scalar_double(ciphertext, adjustment_factor, cryptoContext)
        ciphertext = homo_ops.homo_rescale(ciphertext, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
        ciphertext.scaling_factor = target_sf

    else:
        # Scaling down the message by a correction factor to emulate using a larger q0.
        # This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        cnst = math.pow(2, -correction)
        ciphertext = homo_ops.homo_mul_scalar_double(ciphertext, cnst, cryptoContext)
        ciphertext = homo_ops.homo_rescale(ciphertext, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
    return ciphertext


# @profile_python_function
def eval_linear_wsum_mutable(ciphertexts, constants, cryptoContext: Context):
    input_size = len(constants)

    if cryptoContext.rescaleTech != "FIXEDMANUAL":
        # Check to see if input ciphertexts are of same level
        # and adjust if needed to the max level among them
        minLimbs = ciphertexts[0].cur_limbs
        minIdx = 0
        for i in range(1, input_size):
            if (ciphertexts[i].cur_limbs < minLimbs or
                    (ciphertexts[i].cur_limbs == minLimbs)
                    and ciphertexts[i].noise_deg ==2):
                minLimbs = ciphertexts[i].cur_limbs
                minIdx = i
        for i in range(minIdx):
            ciphertexts[i], ciphertexts[minIdx] = homo_ops.adjust_levels_and_depth(ciphertexts[i], ciphertexts[minIdx], cryptoContext)
        for i in range(minIdx + 1, input_size):
            ciphertexts[i], ciphertexts[minIdx] = homo_ops.adjust_levels_and_depth(ciphertexts[i], ciphertexts[minIdx], cryptoContext)

        if ciphertexts[minIdx].noise_deg == 2:
            for i in range(0, input_size):
                ciphertexts[i] = homo_ops.homo_rescale(ciphertexts[i], BASE_NUM_LEVELS_TO_DROP, cryptoContext)

    wsum = homo_ops.homo_mul_scalar_double(ciphertexts[0], constants[0], cryptoContext)
    for i in range(1, input_size):
        tmp = homo_ops.homo_mul_scalar_double(ciphertexts[i], constants[i], cryptoContext)
        wsum = homo_ops.homo_add(wsum, tmp, cryptoContext)
    wsum = homo_ops.homo_rescale(wsum, 1, cryptoContext) if cryptoContext.rescaleTech == "FIXEDMANUAL" else wsum
    return wsum

def is_not_equal_one(val):
    PREC = math.pow(2, -20)
    return val <= 1 - PREC or val >= 1 + PREC

def degree(coefficients):
    coefficients_size = len(coefficients)
    indx = coefficients_size
    # indx becomes negative (-1) only when all coefficients are zeroes. in this case we return 0
    while True:
        indx -= 1
        if indx < 0:
            return 0
        if coefficients[indx] != 0:
            break
    return indx

# f and g are vectors of Chebyshev interpolation coefficients of the two polynomials.
# We assume their dominant coefficient is not zero. LongDivisionChebyshev returns the
# vector of Chebyshev interpolation coefficients for the quotient and remainder of the
# division f/g. longDiv is a struct that contains the vectors of coefficients for the
# quotient and rest. We assume that the zero-th coefficient is c0, not c0/2 and returns
# the same format.
def long_division_chebyshev(f, g):
    n = degree(f)
    k = degree(g)

    if n != len(f) - 1:
        raise Exception("LongDivisionChebyshev: The dominant coefficient of the dividend is zero.")
    if k != len(g) - 1:
        raise Exception("LongDivisionChebyshev: The dominant coefficient of the divisor is zero.")
    if n < k:
        return np.array([1.0]), np.array(f)

    q = np.zeros(n - k + 1)
    r = np.copy(f)
    d = np.zeros(len(g) + n)

    while n > k:
        d.resize(n + 1, refcheck=False)
        d.fill(0)  # 替换 '@' 为 0
        q[n - k] = 2 * r[-1]
        if is_not_equal_one(g[k]):
            q[n - k] /= g[-1]

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

        if is_not_equal_one(r[-1]):
            # d *= f[n]
            d *= r[-1]
        if is_not_equal_one(g[-1]):
            # d /= g[k]
            d /= g[-1]

        # f -= d
        r = r - d
        if len(r) > 1:
            n = degree(r)
            r.resize(n + 1, refcheck = False)

    if n == k:
        d = np.copy(g)
        q[0] = r[-1]
        if is_not_equal_one(g[-1]):
            q[0] /= g[-1]
        if is_not_equal_one(r[-1]):
            d *= r[-1]
        if is_not_equal_one(g[-1]):
            d /= g[-1]
        r = r - d
        if len(r) > 1:
            n = degree(r)
            r.resize(n + 1, refcheck = False)

    q[0] *= 2
    return q, r



# @profile_python_function
def inner_eval_chebyshev_ps(coefficients,
                            k, m, T, T2, cryptoContext: Context):
    # Compute k * 2^(m-1) - k
    k2m2k = k * (1 << (m - 1)) - k

    # Divide coefficients by T^{k*2^{m-1}}
    Tkm = np.zeros(int(k2m2k + k) + 1)
    Tkm[-1] = 1.0  # Tkm.back() = 1
    divqr_q, divqr_r= long_division_chebyshev(coefficients, Tkm)

    # Subtract x^(k(2^(m-1) - 1)) from r
    r2=np.copy(divqr_r)
    if (int(k2m2k - degree(divqr_r))<=0):
        r2[k2m2k]-=1
        r2.resize(degree(r2)+1, refcheck=False)
    else:
        r2.resize(k2m2k+1,refcheck=False)
        r2[-1]=-1

    # Divide r2 by q
    divcs_q, divcs_r = long_division_chebyshev(r2, divqr_q)

    # Add x^(k(2^(m-1) - 1)) to s
    s2 = np.copy(divcs_r)
    s2.resize(k2m2k+1, refcheck= False)
    s2[-1] = 1.0

    # Evaluate c at u
    dc = degree(divcs_q)
    flag_c = False
    if dc >= 1:
        if dc == 1:
            if divcs_q[1] != 1:
                cu = homo_ops.homo_mul_scalar_double(T[0], divcs_q[1], cryptoContext)
                cu = homo_ops.homo_rescale(cu, 1, cryptoContext) if cryptoContext.rescaleTech == "FIXEDMANUAL" else cu
            else:
                cu = T[0]
        else:
            ctxs = [T[i] for i in range(dc)]
            weights = divcs_q[1:dc + 1]
            cu = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)

        # adds the free term (at x^0)
        cu = homo_ops.homo_add_scalar_double(cu, divcs_q[0] / 2, cryptoContext)
        # Need to reduce levels up to the level of T2[m-1].
        cu = homo_ops.cipher_level_reduce(cu, cu.cur_limbs - T2[m - 1].cur_limbs)
        flag_c = True

    # Evaluate q and s2 at u
    if degree(divqr_q) > k:
        qu = inner_eval_chebyshev_ps(divqr_q, k, m - 1, T, T2, cryptoContext)
    else:
        qcopy=np.copy(divqr_q)
        qcopy.resize(k, refcheck=False)
        deg_qcopy = degree(qcopy)
        if deg_qcopy > 0:
            ctxs = [T[i] for i in range(deg_qcopy)]
            weights = divqr_q[1:deg_qcopy + 1]
            qu = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)
            sum = T[k - 1]
            for i in range(int(math.log2(divqr_q[-1]))):
                sum = homo_ops.homo_add(sum, sum, cryptoContext)
            qu = homo_ops.homo_add(qu, sum, cryptoContext)
        else:
            sum = T[k - 1]
            for i in range(int(math.log2(divqr_q[- 1]))):
                sum = homo_ops.homo_add(sum, sum, cryptoContext)
            qu = sum

        qu = homo_ops.homo_add_scalar_double(qu, divqr_q[0] / 2, cryptoContext)

    # Evaluate s2 at u
    if degree(s2) > k:
        su = inner_eval_chebyshev_ps(s2, k, m - 1, T, T2, cryptoContext)
    else:
        scopy=np.copy(s2)
        scopy.resize(k, refcheck=False)
        deg_scopy = degree(scopy)
        if deg_scopy > 0:
            ctxs = [T[i] for i in range(deg_scopy)]
            weights = s2[1:deg_scopy + 1]
            su = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)
            su = homo_ops.homo_add(su, T[k - 1], cryptoContext)
        else:
            su = T[k - 1]

        su = homo_ops.homo_add_scalar_double(su, s2[0] / 2, cryptoContext)
        su = homo_ops.cipher_level_reduce(su, 1)

    if flag_c:
        result = homo_ops.homo_add(T2[m - 1], cu, cryptoContext)
    else:
        result = homo_ops.homo_add_scalar_double(T2[m - 1], divcs_q[0] / 2, cryptoContext)

    result = homo_ops.homo_mul(result, qu, cryptoContext)
    result = homo_ops.homo_rescale(result, 1, cryptoContext) if cryptoContext.rescaleTech == "FIXEDMANUAL" else result
    result = homo_ops.homo_add(result, su, cryptoContext)

    return result

def PopulateParameterPS(upper_bound_degree):
    # Initialize the mlist array with zeros
    mlist = np.zeros(upper_bound_degree, dtype=np.int32)

    # Define the degree ranges and corresponding m values
    # Each tuple is (start_n, end_n, m)
    ranges = [
        (1, 2, 1),        # n in [1,2], m = 1
        (3, 11, 2),       # n in [3,11], m = 2
        (12, 13, 3),      # n in [12,13], m = 3
        (14, 17, 2),      # n in [14,17], m = 2
        (18, 55, 3),      # n in [18,55], m = 3
        (56, 59, 4),      # n in [56,59], m = 4
        (60, 76, 3),      # n in [60,76], m = 3
        (77, 239, 4),     # n in [77,239], m = 4
        (240, 247, 5),    # n in [240,247], m = 5
        (248, 284, 4),    # n in [248,284], m = 4
        (285, 991, 5),    # n in [285,991], m = 5
        (992, 1007, 6),   # n in [992,1007], m = 6
        (1008, 1083, 5),  # n in [1008,1083], m = 5
        (1084, 2015, 6),  # n in [1084,2015], m = 6
        (2016, 2031, 7),  # n in [2016,2031], m = 7
        (2032, 2204, 6)    # n in [2032,2204], m = 6
    ]

    for start, end, m in ranges:
        if upper_bound_degree < start:
            # If the upper bound is less than the start of the current range, no need to continue
            break
        # Determine the actual end for slicing to avoid exceeding upper_bound_degree
        actual_end = min(end, upper_bound_degree)
        # Set the value m for the slice [start-1, actual_end)
        # In Python, slicing is end-exclusive
        mlist[start-1 : actual_end] = m

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

# @profile_python_function
# note: EvalChebyshevSeriesPS in ckksrns-advancedshe.cpp
def eval_chebyshev_series_ps(x, coefficients, a, b, cryptoContext):
    rescaleTech=cryptoContext.rescaleTech
    n = degree(coefficients)
    f2 = np.copy(coefficients)
    # Make sure the coefficients do not have the zero dominant terms
    if coefficients[- 1] == 0:
        f2.resize(n+1, refcheck=False)

    degs = ComputeDegreesPS(n)
    k = degs[0]
    m = degs[1]

    # computes linear transformation y = -1 + 2 (x-a)/(b-a)
    # consumes one level when a <> -1 && b <> 1
    T = [0 for _ in range(k)]
    if (a - round(a) < 1e-10) and (b - round(b) < 1e-10) and \
       (round(a) == -1) and (round(b) == 1):
        T[0]=x
    else:
        alpha = 2 / (b - a)
        beta = 2 * a / (b - a)
        T[0] = homo_ops.homo_mul_scalar_double(x, alpha, cryptoContext)
        T[0] = homo_ops.homo_rescale(T[0], 1, cryptoContext) if cryptoContext.rescaleTech == "FIXEDMANUAL" else T[0]
        T[0] = homo_ops.homo_add_scalar_double(T[0], -1.0 - beta, cryptoContext)

    # Computes Chebyshev polynomials up to degree k
    # for y: T_1(y) = y, T_2(y), ... , T_k(y)
    # uses binary tree multiplication
    for i in range(2, k + 1):
        if i & (i - 1) == 0:  # i is a power of 2
            # compute T_{2i}(y) = 2*T_i(y)^2 - 1
            square = homo_ops.homo_square(T[i // 2 - 1], cryptoContext)
            T[i - 1] = homo_ops.homo_add(square, square, cryptoContext)
            T[i - 1] = homo_ops.homo_rescale(T[i - 1], 1, cryptoContext) if cryptoContext.rescaleTech == "FIXEDMANUAL" else T[i - 1]
            T[i - 1] = homo_ops.homo_add_scalar_double(T[i - 1], -1.0, cryptoContext)
        else: # non-power of 2
            if i % 2 == 1:  # i is odd
                # compute T_{2i+1}(y) = 2*T_i(y)*T_{i+1}(y) - y
                prod = homo_ops.homo_mul(T[i // 2 - 1], T[i // 2], cryptoContext)
                T[i - 1] = homo_ops.homo_add(prod, prod, cryptoContext)
                T[i - 1] = homo_ops.homo_rescale(T[i - 1], 1, cryptoContext) if cryptoContext.rescaleTech == "FIXEDMANUAL" else T[i-1]
                T[i - 1] = homo_ops.homo_sub(T[i - 1], T[0], cryptoContext)

            else:  # i is even but not power of 2
                # compute T_{2i}(y) = 2*T_i(y)^2 - 1
                square = homo_ops.homo_square(T[i // 2 - 1], cryptoContext)
                T[i - 1] = homo_ops.homo_add(square, square, cryptoContext)
                T[i - 1] = homo_ops.homo_rescale(T[i - 1], 1, cryptoContext) if cryptoContext.rescaleTech == "FIXEDMANUAL" else T[i-1]
                T[i - 1] = homo_ops.homo_add_scalar_double(T[i - 1], -1.0, cryptoContext)

    if rescaleTech =="FIXEDMANUAL":
        # brings all powers of x to the same curlimbs, different to bringing to same level in openfhe
        for i in range(1, k):
            level_diff = T[i - 1].cur_limbs - T[k - 1].cur_limbs
            T[i - 1] = homo_ops.cipher_level_reduce(T[i - 1], level_diff)
    else:
        for i in range(1, k):
            T[i - 1], T[k - 1] = homo_ops.adjust_levels_and_depth(T[i - 1], T[k - 1], cryptoContext)

    # Compute the Chebyshev polynomials T_k(y), T_{2k}(y), T_{4k}(y), ... , T_{2^{m-1}k}(y)
    # T2[0] is used as a placeholder
    T2 = [0 for _ in range(m)]
    T2[0] = T[-1]
    for i in range(1, m):
        square = homo_ops.homo_square(T2[i - 1], cryptoContext)
        T2[i] = homo_ops.homo_add(square, square, cryptoContext)
        T2[i] = homo_ops.homo_rescale(T2[i], 1, cryptoContext) if cryptoContext.rescaleTech == "FIXEDMANUAL" else T2[i]
        T2[i] = homo_ops.homo_add_scalar_double(T2[i], -1.0, cryptoContext)

    # computes T_{k(2*m - 1)}(y)
    T2km1 = T2[0]
    for i in range(1, m):
        # compute T_{k(2*m - 1)} = 2*T_{k(2^{m-1}-1)}(y)*T_{k*2^{m-1}}(y) - T_k(y)
        prod = homo_ops.homo_mul(T2km1, T2[i], cryptoContext)
        T2km1 = homo_ops.homo_add(prod, prod, cryptoContext)
        T2km1 = homo_ops.homo_rescale(T2km1, 1, cryptoContext) if cryptoContext.rescaleTech == "FIXEDMANUAL" else T2km1
        T2km1 = homo_ops.homo_sub(T2km1, T2[0], cryptoContext)

    # Compute k*2^{m-1}-k because we use it a lot
    k2m2k = k * (1 << (m - 1)) - k

    f2.resize(2 * k2m2k + k + 1, refcheck=False)
    f2[-1]=1

    # Divide f2 by T^{k*2^{m-1}}
    Tkm = np.zeros(k2m2k + k + 1)
    Tkm[- 1] = 1

    divqr_q, divqr_r = long_division_chebyshev(f2, Tkm)

    r2 = np.copy(divqr_r)
    if k2m2k - degree(divqr_r) <= 0:
        r2[k2m2k]-=1
        r2.resize(degree(r2)+1, refcheck = False)
    else:
        r2.resize(k2m2k+1, refcheck = False)
        r2[-1] = -1

    # Divide r2 by q
    divcs_q, divcs_r = long_division_chebyshev(r2, divqr_q)

    # Add x^{k(2^{m-1} - 1)} to s
    s2 = np.copy(divcs_r)
    s2.resize(k2m2k + 1, refcheck=False)
    s2[-1] = 1

    # Evaluate c at u
    cu = None
    dc = degree(divcs_q)
    flag_c = False

    if dc >= 1:
        if dc == 1:
            if divcs_q[1] != 1:
                cu = homo_ops.homo_mul_scalar_double(T[0], divcs_q[1], cryptoContext)
                cu = homo_ops.homo_rescale(cu, 1, cryptoContext)if cryptoContext.rescaleTech == "FIXEDMANUAL" else cu
            else:
                cu = T[0]
        else:
            ctxs = [T[i] for i in range(dc)]
            weights = divcs_q[1:dc + 1]
            cu = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)

        # adds the free term (at x^0)
        cu = homo_ops.homo_add_scalar_double(cu, divcs_q[0] / 2, cryptoContext)
        flag_c = True

    # Evaluate q and s2 at u. If their degrees are larger than k, then recursively apply the Paterson-Stockmeyer algorithm.
    qu = None
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
            sum = homo_ops.homo_add(T[k - 1], T[k - 1], cryptoContext)
            qu = homo_ops.homo_add(qu, sum, cryptoContext)
        else:
            qu = T[k - 1]
            for _ in range(1, divqr_q[- 1]):
                qu = homo_ops.homo_add(qu, T[k - 1], cryptoContext)

        # adds the free term (at x^0)
        qu = homo_ops.homo_add_scalar(qu, divqr_q[0] / 2, cryptoContext)
        # The number of levels of qu is the same as the number of levels of T[k-1] + 1.
        # Will only get here when m = 2, so the number of levels of qu and T2[m-1] will be the same.


    # Evaluate s2 at u
    su = None
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
            su = homo_ops.homo_add(su, T[k - 1], cryptoContext)
        else:
            su = T[k - 1]
        # adds the free term (at x^0)
        su = homo_ops.homo_add_scalar(su, s2[0] / 2, cryptoContext)
        # The number of levels of su is the same as the number of levels of T[k-1] + 1.
        # Will only get here when m = 2, so need to reduce the number of levels by 1.

    if flag_c:
        result = homo_ops.homo_add(T2[m - 1], cu, cryptoContext)
    else:
        result = homo_ops.homo_add_scalar(T2[m - 1], divcs_q[0] / 2, cryptoContext)

    result = homo_ops.homo_mul(result, qu, cryptoContext)
    result = homo_ops.homo_rescale(result, 1, cryptoContext) if cryptoContext.rescaleTech == "FIXEDMANUAL" else result
    result = homo_ops.homo_add(result, su, cryptoContext)
    result = homo_ops.homo_sub(result, T2km1, cryptoContext)

    return result


# @profile_python_function
def apply_double_angle_iterations(ciphertext, cryptoContext):
    if cryptoContext.secretKeyDist == "UNIFORM_TERNARY":
        r = R_UNIFORM
    elif cryptoContext.secretKeyDist == "SPARSE_TERNARY":
        r = R_SPARSE
    else:
        raise ValueError("set secretKeyDist first!")

    for j in range(1, r + 1):
        ciphertext = homo_ops.homo_square(ciphertext, cryptoContext)
        ciphertext = homo_ops.homo_add(ciphertext, ciphertext, cryptoContext)
        scalar = -1.0 / math.pow((2.0 * math.pi), math.pow(2.0, j - r))
        ciphertext = homo_ops.homo_add_scalar_double(ciphertext, scalar, cryptoContext)
        ciphertext = homo_ops.homo_rescale(ciphertext, 1,
                                           cryptoContext) if cryptoContext.rescaleTech == "FIXEDMANUAL" else ciphertext
    return ciphertext


def merged_function(A, ctxt, cryptoContext, flag_rem, rot_in, rot_out, config):
    def key_switch_ext(cipher, add_first, cryptoContext):
        curr_limbs = cipher.cur_limbs

        cv0 = torch.zeros(((curr_limbs + cryptoContext.K) << cryptoContext.logN), dtype=torch.uint64, device="cuda").reshape(-1, cryptoContext.N)
        cv1 = torch.zeros(((curr_limbs + cryptoContext.K) << cryptoContext.logN), dtype=torch.uint64, device="cuda").reshape(-1, cryptoContext.N)
        if add_first:
            cv0[:curr_limbs, :] = F.cv_mul_scalar(cipher.cv[0], cryptoContext.PModq_cuda, cryptoContext.moduliQ_cuda,
                                                  cryptoContext.q_mu_cuda, curr_limbs)

        cv1[:curr_limbs, :] = F.cv_mul_scalar(cipher.cv[1], cryptoContext.PModq_cuda, cryptoContext.moduliQ_cuda,
                                              cryptoContext.q_mu_cuda, curr_limbs)
        return Cipher([cv0, cv1], curr_limbs, cipher.scaling_factor, cipher.noise_deg, cipher.slots)
    #todo: it is ct*pt in extent form, refactor?
    def eval_mult_ext(cipher, pt, cryptoContext):
        cur_limbs = cipher.cur_limbs
        if cipher.slots != pt.slots:
            warnings.warn(f"slots unequal! cipher.slots = {cipher.slots}, pt.slots = {pt.slots}",
                          Warning)
        moduli = cryptoContext.BsContext.QplusP_map[cur_limbs]
        mu = cryptoContext.BsContext.QmuplusPmu_map[cur_limbs]
        limbsExt = cur_limbs + cryptoContext.K
        cv0 = F.cv_mul(cipher.cv[0], pt.mx.reshape(-1, cryptoContext.N), moduli, mu, limbsExt)
        cv1 = F.cv_mul(cipher.cv[1], pt.mx.reshape(-1, cryptoContext.N), moduli, mu, limbsExt)
        return Cipher([cv0, cv1], cur_limbs, cipher.scaling_factor*pt.scaling_factor, cipher.noise_deg+pt.noise_deg, cipher.slots)

    def eval_add_ext(cipher0, cipher1, cryptoContext):
        assert cipher0.cur_limbs == cipher1.cur_limbs
        if cipher0.slots != cipher1.slots:
            warnings.warn(f"slots unequal! cipher0.slots = {cipher0.slots}, cipher1.slots = {cipher1.slots}",
                          Warning)
        limbsExt = cipher0.cur_limbs + cryptoContext.K
        moduli = cryptoContext.BsContext.QplusP_map[cipher0.cur_limbs]
        cv = [
            F.cv_add(cv0, cv1, moduli, limbsExt)
            for cv0, cv1 in zip(cipher0.cv, cipher1.cv)
        ]
        return Cipher(cv, cipher0.cur_limbs, cipher0.scaling_factor, cipher0.noise_deg, cipher0.slots)

    # @profile_python_function
    def cv_add_ext(in0, in1, cur_limbs, cryptoContext):
        moduli = cryptoContext.BsContext.QplusP_map[cur_limbs]
        res = F.cv_add(in0, in1, moduli, in0.shape[0])
        return res

    special_limbs = cryptoContext.K
    logN = cryptoContext.logN

    # Set up configuration
    loop_direction = config["loop_direction"]
    cipher_mod_levels = config["cipher_mod_levels"]
    eval_fast_rotation_reshape = config["eval_fast_rotation_reshape"]
    key_switch_ext_size = config["key_switch_ext_size"]
    params = config["params"]
    start = config["start"]
    stop = config["stop"]

    level_budget = params.level_budget
    num_rotations = params.num_rotations
    b = params.baby_step
    g = params.giant_step
    num_rotations_rem = params.num_rotations_rem
    g_rem = params.giant_step_rem
    b_rem = params.baby_step_rem

    result = Cipher([ctxt.cv[0].clone(), ctxt.cv[1].clone()], ctxt.cur_limbs, ctxt.scaling_factor, ctxt.noise_deg, ctxt.slots)

    # Determine loop range based on direction
    if loop_direction == "forward":
        loop_range = range(start, stop)
    else:
        loop_range = range(start, stop, -1)
    
    for s in loop_range:
        if (loop_direction == "forward" and s != 0) or (loop_direction == "backward" and s != level_budget -1):
            result = homo_ops.homo_rescale(result, cipher_mod_levels, cryptoContext)
        
        curr_limbs = result.cur_limbs
        limbs_ext = curr_limbs + special_limbs
        len_ext = limbs_ext << logN

        digits = hoisting_keyswitch.eval_fast_rotation_precompute(result.cv[1], result.cur_limbs, cryptoContext)

        fast_rotation_ext = [None for _ in range(g)]
        
        for j in range(g):
            if rot_in[s][j] != 0:
                fast_rotation_ext[j] = hoisting_keyswitch.eval_fast_rotation_ext(result, digits, rot_in[s][j], True,
                                                                                 cryptoContext)
            else:
                fast_rotation_ext[j] = key_switch_ext(result, True, cryptoContext)
        
        for i in range(b):
            G = g * i
            inner = eval_mult_ext(fast_rotation_ext[0], A[s][G], cryptoContext)
            
            for j in range(1, g):
                if (G + j) != num_rotations:
                    tmp_ext = eval_mult_ext(fast_rotation_ext[j], A[s][G + j], cryptoContext)
                    inner = eval_add_ext(inner, tmp_ext, cryptoContext)
            
            if i == 0:
                first = F.cv_moddown(inner.cv[0], curr_limbs, cryptoContext)
                F.cv_set_zero(inner.cv[0], len_ext)
                outer = inner
            else:
                if rot_out[s][i] != 0:
                    inner_ks_down = hoisting_keyswitch.key_switch_down(inner.cv[0], inner.cv[1], curr_limbs,
                                                                       inner.scaling_factor, inner.noise_deg,
                                                                       inner.slots, cryptoContext)
                    auto_index = cryptoContext.find_auto_index(rot_out[s][i])

                    first_current = F.cv_automorphism_transform(
                        inner_ks_down.cv[0], curr_limbs, auto_index, cryptoContext)
                    first = cv_add_ext(first, first_current, curr_limbs, cryptoContext)
                    
                    inner_digits = hoisting_keyswitch.eval_fast_rotation_precompute(
                        inner_ks_down.cv[1], inner_ks_down.cur_limbs, cryptoContext
                    )
                    
                    inner_ks_down_ext = hoisting_keyswitch.eval_fast_rotation_ext(inner_ks_down, inner_digits, rot_out[s][i],
                                                                                  False, cryptoContext)
                    outer = eval_add_ext(outer, inner_ks_down_ext, cryptoContext)
                else:
                    tmp_first = F.cv_moddown(inner.cv[0], curr_limbs, cryptoContext)
                    first = cv_add_ext(first, tmp_first, curr_limbs, cryptoContext)
                    F.cv_set_zero(inner.cv[0], len_ext)
                    outer = eval_add_ext(outer, inner, cryptoContext)
        
        result = hoisting_keyswitch.key_switch_down(outer.cv[0], outer.cv[1], curr_limbs, outer.scaling_factor,
                                                    outer.noise_deg, outer.slots, cryptoContext)
        result.cv[0] = cv_add_ext(result.cv[0], first, curr_limbs, cryptoContext)
    
    if flag_rem:
        result = homo_ops.homo_rescale(result, cipher_mod_levels, cryptoContext)
        curr_limbs = result.cur_limbs
        limbs_ext = curr_limbs + special_limbs
        len_ext = limbs_ext << logN

        digits = hoisting_keyswitch.eval_fast_rotation_precompute(result.cv[1], result.cur_limbs, cryptoContext)
        
        fast_rotation_ext = [None for _ in range(g_rem)]
        
        s = stop if loop_direction == "backward" else level_budget - flag_rem
        
        for j in range(g_rem):
            if rot_in[s][j] != 0:
                fast_rotation_ext[j] = hoisting_keyswitch.eval_fast_rotation_ext(result, digits, True,
                                                                                 cryptoContext, )
            else:
                fast_rotation_ext[j] = key_switch_ext(result, True, cryptoContext)
        
        for i in range(b_rem):
            G = g_rem * i
            inner = eval_mult_ext(fast_rotation_ext[0], A[s][G], cryptoContext)
            
            for j in range(1, g_rem):
                if (G + j) != num_rotations_rem:
                    tmp_ext = eval_mult_ext(fast_rotation_ext[j], A[s][G + j], cryptoContext)
                    inner = eval_add_ext(inner, tmp_ext, cryptoContext)
            
            if i == 0:
                first = F.cv_moddown(inner.cv[0], curr_limbs, cryptoContext)
                F.cv_set_zero(inner.cv[0], len_ext)
                outer = inner
            else:
                if rot_out[s][i] != 0:
                    inner_ks_down = hoisting_keyswitch.key_switch_down(inner.cv[0], inner.cv[1], curr_limbs,
                                                                       inner.scaling_factor, inner.noise_deg,
                                                                       inner.slots, cryptoContext)
                    auto_index = cryptoContext.find_auto_index(rot_out[s][i])

                    first_current = F.cv_automorphism_transform(
                        inner_ks_down.cv[0], curr_limbs, auto_index, cryptoContext)
                    first = cv_add_ext(first, first_current, curr_limbs, cryptoContext)
                    
                    inner_digits = hoisting_keyswitch.eval_fast_rotation_precompute(
                        inner_ks_down.cv[1], inner_ks_down.cur_limbs, cryptoContext
                    )
                    
                    inner_ks_down_ext = hoisting_keyswitch.eval_fast_rotation_ext(inner_ks_down, inner_digits, False,
                                                                                  cryptoContext, )
                    outer = eval_add_ext(outer, inner_ks_down_ext, cryptoContext)
                else:
                    tmp_first = F.cv_moddown(inner.cv[0], curr_limbs, cryptoContext)
                    first = cv_add_ext(first, tmp_first, curr_limbs, cryptoContext)
                    F.cv_set_zero(inner.cv[0], len_ext)
                    outer = eval_add_ext(outer, inner, cryptoContext)
        
        result = hoisting_keyswitch.key_switch_down(outer.cv[0], outer.cv[1], curr_limbs, outer.scaling_factor,
                                                    outer.noise_deg, outer.slots, cryptoContext)
        result.cv[0] = cv_add_ext(result.cv[0], first, curr_limbs, cryptoContext)
    
    return result



# @profile_python_function
def eval_coeffs_to_slots(A, ctxt, cryptoContext):

    precom = cryptoContext.BsContext

    stop = 0 if precom.paramsEnc.layers_rem != 0 else -1
    flag_rem = 1 if precom.paramsEnc.layers_rem != 0 else 0

    config = {
        "loop_direction": "backward",
        "cipher_mod_levels": 1,
        "eval_fast_rotation_reshape": True,
        "key_switch_ext_size": 2,
        "params": precom.paramsEnc,
        "start": precom.paramsEnc.level_budget - 1,
        "stop": stop
    }

    return merged_function(A, ctxt, cryptoContext, flag_rem, cryptoContext.BsContext.C2S_rot_in, cryptoContext.BsContext.C2S_rot_out, config)

# @profile_python_function
def eval_slots_to_coeffs(A, ctxt, cryptoContext):

    precom = cryptoContext.BsContext
    flag_rem = 1 if precom.paramsDec.layers_rem != 0 else 0

    config = {
        "loop_direction": "forward",
        "cipher_mod_levels": BASE_NUM_LEVELS_TO_DROP,
        "eval_fast_rotation_reshape": False,
        "key_switch_ext_size": NORMAL_CIPHER_SIZE,
        "params": precom.paramsDec,
        "start": 0,
        "stop": precom.paramsDec.level_budget - flag_rem
    }

    return merged_function(A, ctxt, cryptoContext, flag_rem, cryptoContext.BsContext.S2C_rot_in, cryptoContext.BsContext.S2C_rot_out, config)

# @profile_python_function
def eval_linear_transform(A, ct, scheme):
    # TODO: to be implemented
    pass

# @profile_python_function
def cipher_mod_raise(cipher, L0, cryptoContext):
    cv0 = F.cv_switch_modulus_with_intt_ntt(cipher.cv[0], L0, cryptoContext)
    cv1 = F.cv_switch_modulus_with_intt_ntt(cipher.cv[1], L0, cryptoContext)
    return Cipher([cv0, cv1], L0, cipher.scaling_factor, cipher.noise_deg, cipher.slots)

# @profile_python_function
def cipher_mult_by_monomial_and_equal(cipher, monomial_degree, cryptoContext):
    l = cipher.cur_limbs
    cipher.cv[0] = F.cv_mul_by_monomial(cipher.cv[0], l, monomial_degree, cryptoContext)
    cipher.cv[1] = F.cv_mul_by_monomial(cipher.cv[1], l, monomial_degree, cryptoContext)
    return cipher


# @profile_python_function
# note: EvalBootstrap in ckksrns-fhe.cpp
def eval_bootstrap(ciphertext, L0, slots, cryptoContext):
    M = cryptoContext.M
    N = cryptoContext.N
    # cryptoContext.slots = slots #fixme: bad assignment!
    precom = cryptoContext.BsContext
    moduliQ = cryptoContext.moduliQ
    rescaleTech = cryptoContext.rescaleTech

    # note: FLEXIBLEAUTOEXT is not implemented yet
    assert rescaleTech == "FIXEDMANUAL" or rescaleTech == "FLEXIBLEAUTO"

    if (rescaleTech=="FLEXIBLEAUTOEXT"):
        pass
        # For FLEXIBLEAUTOEXT we raised ciphertext does not include extra modulus
        # as it is multiplied by auxiliary plaintext
        #todo: to be implemented, should raise less modulus

    q = moduliQ[0]
    q_double = float(q)

    p = cryptoContext.logp  # Equivalent to dcrbits in OpenFHE
    powP = 2**p
    deg = round(math.log2(q_double / powP))

    correction = (
        cryptoContext.correctionFactor - deg
    )  # fixme: originally a uint32_t in OpenFHE
    post = 2**deg
    pre = 1.0 / post
    scalar = round(post)

    # -------------------
    # raising the modulus
    # -------------------
    # In FLEXIBLEAUTO, raising the ciphertext to a larger number
    # of towers is a bit more complex, because we need to adjust
    # it's scaling factor to the one that corresponds to the level
    # it's being raised to.
    # Increasing the modulus

    tmp = ciphertext
    tmp = homo_ops.homo_rescale(tmp, tmp.noise_deg-1, cryptoContext)
    tmp = adjust_ciphertext(tmp, correction, L0, cryptoContext)

    # We only use the level 0 ciphertext here. All other towers are automatically ignored to make
    # CKKS bootstrapping faster.
    raised = cipher_mod_raise(tmp, L0, cryptoContext)

    constantEvalMult = pre * (1.0 / (precom.k * N))
    raised = homo_ops.homo_mul_scalar_double(raised, constantEvalMult, cryptoContext)

    ctxtDec = None  # Initialize decrypted ciphertext
    # todo: align with openfhe, but should be refactored. since when only one lb=1, none of them go into EvalLinearTransform.
    isLTBootstrap = (precom.paramsEnc.level_budget == 1) and (precom.paramsDec.level_budget == 1)

    if slots == M // 4: # FULLY PACKED CASE
        # need to call internal modular reduction so it also works for FLEXIBLEAUTO
        raised = homo_ops.homo_rescale(raised, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if isLTBootstrap:
            ctxtEnc = eval_linear_transform(precom.m_U0hatTPre, raised, cryptoContext)
        else:
            ctxtEnc = eval_coeffs_to_slots(precom.m_U0hatTPreFFT, raised, cryptoContext)

        conj = homo_ops.homo_conjugate(ctxtEnc, cryptoContext)
        ctxtEncI = homo_ops.homo_sub(ctxtEnc, conj, cryptoContext)
        ctxtEnc = homo_ops.homo_add(ctxtEnc, conj, cryptoContext)
        ctxtEncI = cipher_mult_by_monomial_and_equal(ctxtEncI, 3 * M // 4, cryptoContext)

        if rescaleTech == "FIXEDMANUAL":
            while(ctxtEnc.noise_deg>1):
                ctxtEnc = homo_ops.homo_rescale(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
                ctxtEncI = homo_ops.homo_rescale(ctxtEncI, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
        else:
            if ctxtEnc.noise_deg==2:
                ctxtEnc = homo_ops.homo_rescale(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
                ctxtEncI = homo_ops.homo_rescale(ctxtEncI, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        # ---------------------------------
        # Running Approximate Mod Reduction
        # ---------------------------------
        # Evaluate Chebyshev series for the sine wave
        ctxtEnc = eval_chebyshev_series_ps(ctxtEnc, precom.coefficients, -1, 1, cryptoContext)
        ctxtEncI = eval_chebyshev_series_ps(ctxtEncI, precom.coefficients, -1, 1, cryptoContext)


        if rescaleTech != "FIXEDMANUAL":
            ctxtEnc = homo_ops.homo_rescale(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
            ctxtEncI = homo_ops.homo_rescale(ctxtEncI, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
        ctxtEnc = apply_double_angle_iterations(ctxtEnc, cryptoContext)
        ctxtEncI = apply_double_angle_iterations(ctxtEncI, cryptoContext)

        ctxtEncI = cipher_mult_by_monomial_and_equal(ctxtEncI, M // 4, cryptoContext)
        ctxtEnc = homo_ops.homo_add(ctxtEnc, ctxtEncI, cryptoContext)

        # scale the message back up after Chebyshev interpolation
        ctxtEnc = homo_ops.homo_mul_scalar_int(ctxtEnc, scalar, cryptoContext)

        # --------------------
        # Running SlotToCoeff
        # --------------------

        # In the case of FLEXIBLEAUTO, we need one extra tower
        # openfhetodo: See if we can remove the extra level in FLEXIBLEAUTO
        if rescaleTech != "FIXEDMANUAL":
            ctxtEnc = homo_ops.homo_rescale(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if isLTBootstrap:
            ctxtDec = eval_linear_transform(precom.m_U0Pre, ctxtEnc, cryptoContext)
        else:
            ctxtDec = eval_slots_to_coeffs(precom.m_U0PreFFT, ctxtEnc, cryptoContext)

    else: # SPARSELY PACKED CASE
        # -------------------
        # Running PartialSum
        # -------------------
        for step in range(int(math.log2(N // (2 * slots)))):
            temp = homo_ops.homo_rotate(raised, (1 << step) * slots, cryptoContext)
            raised = homo_ops.homo_add(raised, temp, cryptoContext)

        # ---------------------
        # Running CoeffsToSlots
        # ---------------------
        raised = homo_ops.homo_rescale(raised, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if isLTBootstrap:
            ctxtEnc = eval_linear_transform(precom.m_U0hatTPre, raised, cryptoContext)
        else:
            ctxtEnc = eval_coeffs_to_slots(precom.m_U0hatTPreFFT, raised, cryptoContext)


        conj = homo_ops.homo_conjugate(ctxtEnc, cryptoContext)
        ctxtEnc = homo_ops.homo_add(ctxtEnc, conj, cryptoContext)

        if rescaleTech == "FIXEDMANUAL":
            while ctxtEnc.noise_deg>1:
                ctxtEnc = homo_ops.homo_rescale(ctxtEnc, 1, cryptoContext)
        else:
            if ctxtEnc.noise_deg ==2 :
                ctxtEnc = homo_ops.homo_rescale(ctxtEnc, 1, cryptoContext)

        # ---------------------------------
        # Running Approximate Mod Reduction
        # ---------------------------------

        # Evaluate Chebyshev series for the sine wave
        ctxtEnc = eval_chebyshev_series_ps(ctxtEnc, precom.coefficients, -1, 1, cryptoContext)

        if rescaleTech != "FIXEDMANUAL":
            ctxtEnc = homo_ops.homo_rescale(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
        ctxtEnc = apply_double_angle_iterations(ctxtEnc, cryptoContext)

        # scale the message back up after Chebyshev interpolation
        ctxtEnc = homo_ops.homo_mul_scalar_int(ctxtEnc, scalar, cryptoContext)

        # --------------------
        # Running SlotToCoeff
        # --------------------
        # In the case of FLEXIBLEAUTO, we need one extra tower
        # openfhetodo: See if we can remove the extra level in FLEXIBLEAUTO
        if rescaleTech != "FIXEDMANUAL":
            ctxtEnc = homo_ops.homo_rescale(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if isLTBootstrap:
            ctxtDec = eval_linear_transform(precom.m_U0Pre, ctxtEnc, cryptoContext)
        else:
            ctxtDec = eval_slots_to_coeffs(precom.m_U0PreFFT, ctxtEnc, cryptoContext)


        ctxtDec_rot = homo_ops.homo_rotate(ctxtDec, slots, cryptoContext)
        ctxtDec = homo_ops.homo_add(ctxtDec, ctxtDec_rot, cryptoContext)

    # 64-bit only: scale back the message to its original scale.
    corFactor = 1 << round(correction)
    ctxtDec = homo_ops.homo_mul_scalar_int(ctxtDec, corFactor, cryptoContext)
    if rescaleTech == "FIXEDMANUAL":  # added by yhh. FLEXIBLEAUTO can handle noise_deg=2, therefore no need to rescale
        ctxtDec = homo_ops.homo_rescale(ctxtDec, ctxtDec.noise_deg-1, cryptoContext)

    return ctxtDec


def eval_bootstrap_setup(context, level_budget, dim1, numslots, correction_factor):

    m_U0hatTPreFFT_dim1 = len(context.m_U0hatTPreFFT_dim)
    m_U0hatTPreFFT_dim2 = context.m_U0hatTPreFFT_dim
    m_U0hatTPreFFT_limbs = context.m_U0hatTPreFFT_limbs
    mx_len = context.N
    mx_slots = numslots
    m_U0PreFFT_dim1 = len(context.m_U0PreFFT_dim)
    m_U0PreFFT_dim2 = context.m_U0PreFFT_dim
    m_U0PreFFT_limbs = context.m_U0PreFFT_limbs

    M = context.M
    slots = M // 4 if numslots == 0 else numslots
    rescale_tech = context.rescaleTech
    precom = context.BsContext

    # 设置 correction_factor
    if correction_factor == 0:
        if (
            rescale_tech == "FLEXIBLEAUTO"
            or rescale_tech == "FLEXIBLEAUTOEXT"
        ):
            # 实验结果得出的最佳精度对应的默认 correction factors
            tmp = round(-0.265 * (2 * math.log2(M / 2) + math.log2(slots)) + 19.1)
            if tmp < 7:
                context.correctionFactor = 7
            elif tmp > 13:
                context.correctionFactor = 13
            else:
                context.correctionFactor = int(tmp)
        else:
            context.correctionFactor = 9
    else:
        context.correctionFactor = correction_factor

    precom.m_slots = slots
    precom.m_dim1 = dim1[0]

    log_slots = math.log2(slots)

    # 检查 level budget 并计算参数
    new_budget = [level_budget[0], level_budget[1]]

    if level_budget[0] > log_slots:
        print(
            f"\nWarning, the level budget for encoding cannot be this large. "
            f"The budget was changed to {int(log_slots)}"
        )
        new_budget[0] = int(log_slots)
    if level_budget[0] < 1:
        print(
            f"\nWarning, the level budget for encoding has to be at least 1. "
            f"The budget was changed to 1"
        )
        new_budget[0] = 1

    if level_budget[1] > log_slots:
        print(
            f"\nWarning, the level budget for decoding cannot be this large. "
            f"The budget was changed to {int(log_slots)}"
        )
        new_budget[1] = int(log_slots)
    if level_budget[1] < 1:
        print(
            f"\nWarning, the level budget for decoding has to be at least 1. "
            f"The budget was changed to 1"
        )
        new_budget[1] = 1

    precom.m_params_enc = context.BsContext.GetCollapsedFFTParams(
        slots, new_budget[0], dim1[0]
    )
    precom.m_params_dec = context.BsContext.GetCollapsedFFTParams(
        slots, new_budget[1], dim1[1]
    )

    if level_budget[0] == 1 and level_budget[1] == 1:
        pass
        # # todo: to be implemented, need to get from openfhe
        # precom.m_U0Pre = [None] * LTMatrix_Row
        # precom.m_U0hatTPre = [None] * LTMatrix_Row
        # for i in range(LTMatrix_Row):
        #     # precom.m_U0hatTPre
        #     m_U0hatTPre_len = LTMatrix_mx_len * m_U0hatTPre_limbs
        #     m_U0hatTPre = [m_U0hatTPre_mx[i * m_U0hatTPre_len + j] for j in range(m_U0hatTPre_len)]
        #     precom.m_U0hatTPre[i] = Plaintext(m_U0hatTPre, LTMatrix_mx_len, LTMatrix_Column, m_U0hatTPre_limbs)
        #
        #     # precom.m_U0Pre
        #     m_U0Pre_len = LTMatrix_mx_len * m_U0Pre_limbs
        #     m_U0Pre = [m_U0Pre_mx[i * m_U0Pre_len + j] for j in range(m_U0Pre_len)]
        #     precom.m_U0Pre[i] = Plaintext(m_U0Pre, LTMatrix_mx_len, LTMatrix_Column, m_U0Pre_limbs)
    else:
        RHScnt = 0
        precom.m_U0hatTPreFFT = [[0] * i for i in m_U0hatTPreFFT_dim2]
        cnt = 0
        for i in range(0, m_U0hatTPreFFT_dim1):
            j_len = m_U0hatTPreFFT_dim2[i]
            limbs = m_U0hatTPreFFT_limbs[i]
            m_U0hatTPreFFT_len = mx_len * limbs
            for j in range(j_len):
                m_U0hatTPreFFT = np.zeros(m_U0hatTPreFFT_len, dtype=np.uint64)
                LHScnt = 0
                for k in range(limbs):
                    for l in range(mx_len):
                        m_U0hatTPreFFT[LHScnt] = context.m_U0hatTPreFFT_mx[RHScnt]
                        LHScnt += 1
                        RHScnt += 1

                m_U0hatTPreFFT = torch.tensor(
                    m_U0hatTPreFFT, dtype=torch.uint64, device="cuda"
                )
                precom.m_U0hatTPreFFT[i][j] = Plaintext(m_U0hatTPreFFT, mx_len, mx_slots, limbs,
                                                        context.m_U0hatTPreFFT_scaling_factor[cnt], 1)
                cnt+=1

        cnt=0
        RHScnt = 0
        precom.m_U0PreFFT = [[0] * i for i in m_U0PreFFT_dim2]
        for i in range(m_U0PreFFT_dim1):
            j_len = m_U0PreFFT_dim2[i]
            limbs = m_U0PreFFT_limbs[i]
            m_U0PreFFT_len = mx_len * limbs
            for j in range(j_len):
                m_U0PreFFT = np.zeros(m_U0PreFFT_len, dtype=np.uint64)
                LHScnt = 0
                for k in range(limbs):
                    for l in range(mx_len):
                        m_U0PreFFT[LHScnt] = context.m_U0PreFFT_mx[RHScnt]
                        LHScnt += 1
                        RHScnt += 1
                m_U0PreFFT = torch.tensor(m_U0PreFFT, dtype=torch.uint64, device="cuda")
                precom.m_U0PreFFT[i][j] = Plaintext(m_U0PreFFT, mx_len, mx_slots, limbs,
                                                    context.m_U0PreFFT_scaling_factor[cnt], 1)
                cnt+=1



def BootstrapTest_N65536L26lB44(
    logN=14,
    logSlots=12,
    maxLevelsRemaining=3,
    levelBudget=[2, 2],
    dnum=3,
    dcrtBits=59,
    firstMod=60,
    approxModDepth=9,
    rescaleTech = "FLEXIBLEAUTO"# "FLEXIBLEAUTO" # "FIXEDMANUAL"
):
    load_from_file = True
    dim1 = [0, 0]
    if load_from_file:
        save_path = "torch/fhe/data/{}.pkl".format(rescaleTech)
        cryptoContext, openfhe_context = utils.load_context(save_path)

    else:
        openfhe_context, cryptoContext = client.gen_contexts(
            logN=logN,
            logSlots=logSlots, # possible slots value of runtime ciphertext #todo: should be a list?
            maxLevelsRemaining=maxLevelsRemaining,
            levelBudget=levelBudget,
            dnum=dnum,
            dcrtBits=dcrtBits,
            firstMod=firstMod,
            approxModDepth=approxModDepth,
            rotate_index=[],
            secretKeyDist="UNIFORM_TERNARY",
            rescaleTech=rescaleTech,
            dim1 = dim1,
        )

        save_path="torch/fhe/data/{}.pkl".format(cryptoContext.rescaleTech)
        utils.save_context(cryptoContext, openfhe_context, save_path)
        cryptoContext, _ = utils.load_context(save_path)

    eval_bootstrap_setup(
        cryptoContext, cryptoContext.levelBudget, dim1, (1<<logSlots), 0
    )

    # Test the correctness of the bootstrapping
    values = [0.111111, 0.222222, 0.333333, 0.444444, 0.555555, 0.666666, 0.777777, 0.888888]
    x = np.array([values[i % len(values)] for i in range((1<<logSlots))])
    x = torch.tensor(x, device="cuda")
    cipher = openfhe_context.encrypt(x)
    cipher.cv[0] = cipher.cv[0][:2]
    cipher.cv[1] = cipher.cv[1][:2]
    cipher.cur_limbs = 2

    result = eval_bootstrap(cipher, L0=cryptoContext.L, slots=(1<<logSlots), cryptoContext=cryptoContext)
    after_boot = openfhe_context.decrypt(result)
    after_boot = after_boot.cpu().numpy().reshape(-1)
    print(after_boot[:10])
    x = x.cpu().numpy().reshape(-1)
    if(np.any(np.abs(after_boot - x) > 3e-2)):
        print("Error is too large!")
        print("Error is too large!")
        print("Error is too large!")
    else:
        print("BootstrapTest_N65536L26lB44: Test passed!")
        print("BootstrapTest_N65536L26lB44: Test passed!")
        print("BootstrapTest_N65536L26lB44: Test passed!")

    measure_execution_time = True
    if measure_execution_time:
        start = time.time()
        result = eval_bootstrap(cipher, L0=cryptoContext.L, slots=(1<<logSlots), cryptoContext=cryptoContext)
        end = time.time()
        print("time", end - start)

        # Print the accumulated execution times
        # print("\nTotal execution time for each function:")
        # sorted_execution_times = sorted(utils.execution_times.items(), key=lambda x: x[1], reverse=True)
        # for func_name, total_time in sorted_execution_times:
        #     print(f"{func_name}: {total_time:.6f} seconds")

        pytorch_profiling = False
        if pytorch_profiling:
            # Set up the profiler
            with torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    "/home/zrji/log"
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as profiler:
                # Start profiling specific functions with torch.profiler.record_function()
                result = eval_bootstrap(cipher, L0=cryptoContext.L, slots=(1<<logSlots),
                                        cryptoContext=cryptoContext)

            # Get the profiling results
            profiler_results = profiler.key_averages()

            # Print the profiling summary in a table format
            print(profiler_results.table(sort_by="self_cpu_time_total"))



