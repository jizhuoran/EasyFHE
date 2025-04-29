import time
from .context import *
from .bs_context import *
from . import homo_ops
import numpy as np

BASE_NUM_LEVELS_TO_DROP = 1 #todo: to be removed, or move to cryptoContext



def eval_linear_wsum_mutable(ciphertexts, constants, cryptoContext: Context):
    if cryptoContext.rescaleTech != "FIXEDMANUAL":
        target_idx = min(range(len(ciphertexts)), key=lambda i: ciphertexts[i].cur_limbs - ciphertexts[i].noise_deg)
        if ciphertexts[target_idx].noise_deg == 2:
            ciphertexts[target_idx] = homo_ops.force_rescale(ciphertexts[target_idx], 1, cryptoContext)
        for i in range(len(ciphertexts)):
            ciphertexts[i] = homo_ops.adjust_to(
                ciphertexts[i], ciphertexts[target_idx].cur_limbs, ciphertexts[target_idx].noise_deg, ciphertexts[target_idx].scaling_factor, cryptoContext
            )

    wsum = homo_ops.homo_mul_scalar_double(ciphertexts[0], constants[0], cryptoContext)
    for i in range(1, len(constants)):
        tmp = homo_ops.homo_mul_scalar_double(ciphertexts[i], constants[i], cryptoContext)
        wsum = homo_ops.homo_add(wsum, tmp, cryptoContext)
    wsum = homo_ops.homo_rescale(wsum, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
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
                            k, m, T, T2, cryptoContext: Context):
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
                cu = homo_ops.homo_mul_scalar_double(T[0], divcs_q[1], cryptoContext)
                cu = homo_ops.homo_rescale(cu, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
            else:
                cu = T[0]
        else:
            ctxs = [T[i] for i in range(dc)]
            weights = divcs_q[1:dc + 1]
            cu = eval_linear_wsum_mutable(ctxs, weights, cryptoContext)

        # adds the free term (at x^0)
        cu = homo_ops.homo_add_scalar_double(cu, divcs_q[0] / 2, cryptoContext)
        # Need to reduce levels up to the level of T2[m-1].
        if cryptoContext.rescaleTech == "FIXEDMANUAL":
            cu = homo_ops.adjust_to(cu, T2[m - 1].cur_limbs, T2[m - 1].noise_deg, T2[m - 1].scaling_factor, cryptoContext)
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
            sum = homo_ops.homo_mul_scalar_int(T[k - 1], 2 ** math.floor(math.log2(divqr_q[-1])), cryptoContext)
            # for i in range(int(math.log2(divqr_q[-1]))):
            # sum = homo_ops.homo_add(sum, sum, cryptoContext)
            qu = homo_ops.homo_add(qu, sum, cryptoContext)
        else:
            sum = T[k - 1]
            sum = homo_ops.homo_mul_scalar_int(T[k - 1], 2 ** math.floor(math.log2(divqr_q[-1])), cryptoContext)
            # for i in range(int(math.log2(divqr_q[-1]))):
            # sum = homo_ops.homo_add(sum, sum, cryptoContext)
            qu = sum

        qu = homo_ops.homo_add_scalar_double(qu, divqr_q[0] / 2, cryptoContext)

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
            su = homo_ops.homo_add(su, T[k - 1], cryptoContext)
        else:
            su = T[k - 1]

        su = homo_ops.homo_add_scalar_double(su, s2[0] / 2, cryptoContext)
        if cryptoContext.rescaleTech == "FIXEDMANUAL":
            su = homo_ops.adjust_to(su, su.cur_limbs - 1, 1, None, cryptoContext)

    if flag_c:
        result = homo_ops.homo_add(T2[m - 1], cu, cryptoContext)
    else:
        result = homo_ops.homo_add_scalar_double(T2[m - 1], divcs_q[0] / 2, cryptoContext)

    result = homo_ops.homo_mul(result, qu, cryptoContext)
    result = homo_ops.homo_rescale(result, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
    result = homo_ops.homo_add(result, su, cryptoContext)

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
def eval_bootstrapping_chebyshev(x, a, b, cryptoContext):


    coefficientsSparse = np.array(
            [
                -0.18646470117093214, 0.036680543700430925, -0.20323558926782626, 0.029327390306199311,
                -0.24346234149506416, 0.011710240188138248, -0.27023281815251715, -0.017621188001030602,
                -0.21383614034992021, -0.048567932060728937, -0.013982336571484519, -0.051097367628344978,
                0.24300487324019346, 0.0016547743046161035, 0.23316923792642233, 0.060707936480887646,
                -0.18317928363421143, 0.0076878773048247966, -0.24293447776635235, -0.071417413140564698,
                0.37747441314067182, 0.065154496937795681, -0.24810721693607704, -0.033588418808958603,
                0.10510660697380972, 0.012045222815124426, -0.032574751830745423, -0.0032761730196023873,
                0.0078689491066424744, 0.00070965574480802061, -0.0015405394287521192, -0.00012640521062948649,
                0.00025108496615830787, 0.000018944629154033562, -0.000034753284216308228, -2.4309868106111825e-6,
                4.1486274737866247e-6, 2.7079833113674568e-7, -4.3245388569898879e-7, -2.6482744214856919e-8,
                3.9770028771436554e-8, 2.2951153557906580e-9, -3.2556026220554990e-9, -1.7691071323926939e-10,
                2.5459052150406730e-10
            ],
            dtype=np.float64,
        )

    coefficientsUniform = np.array(
        [
            0.15421426400235561,
            -0.0037671538417132409,
            0.16032011744533031,
            -0.0034539657223742453,
            0.17711481926851286,
            -0.0027619720033372291,
            0.19949802549604084,
            -0.0015928034845171929,
            0.21756948616367638,
            0.00010729951647566607,
            0.21600427371240055,
            0.0022171399198851363,
            0.17647500259573556,
            0.0042856217194480991,
            0.086174491919472254,
            0.0054640252312780444,
            -0.046667988130649173,
            0.0047346914623733714,
            -0.17712686172280406,
            0.0016205080004247200,
            -0.22703114241338604,
            -0.0028145845916205865,
            -0.13123089730288540,
            -0.0056345646688793190,
            0.078818395388692147,
            -0.0037868875028868542,
            0.23226434602675575,
            0.0021116338645426574,
            0.13985510526186795,
            0.0059365649669377071,
            -0.13918475289368595,
            0.0018580676740836374,
            -0.23254376365752788,
            -0.0054103844866927788,
            0.056840618403875359,
            -0.0035227192748552472,
            0.25667909012207590,
            0.0055029673963982112,
            -0.073334392714092062,
            0.0027810273357488265,
            -0.24912792167850559,
            -0.0069524866497120566,
            0.21288810409948347,
            0.0017810057298691725,
            0.088760951809475269,
            0.0055957188940032095,
            -0.31937177676259115,
            -0.0087539416335935556,
            0.34748800245527145,
            0.0075378299617709235,
            -0.25116537379803394,
            -0.0047285674679876204,
            0.13970502851683486,
            0.0023672533925155220,
            -0.063649401080083698,
            -0.00098993213448982727,
            0.024597838934816905,
            0.00035553235917057483,
            -0.0082485030307578155,
            -0.00011176184313622549,
            0.0024390574829093264,
            0.000031180384864488629,
            -0.00064373524734389861,
            -7.8036008952377965e-6,
            0.00015310015145922058,
            1.7670804180220134e-6,
            -0.000033066844379476900,
            -3.6460909134279425e-7,
            6.5276969021754105e-6,
            6.8957843666189918e-8,
            -1.1842811187642386e-6,
            -1.2015133285307312e-8,
            1.9839339947648331e-7,
            1.9372045971100854e-9,
            -3.0815418032523593e-8,
            -2.9013806338735810e-10,
            4.4540904298173700e-9,
            4.0505136697916078e-11,
            -6.0104912807134771e-10,
            -5.2873323696828491e-12,
            7.5943206779351725e-11,
            6.4679566322060472e-13,
            -9.0081200925539902e-12,
            -7.4396949275292252e-14,
            1.0057423059167244e-12,
            8.1701187638005194e-15,
            -1.0611736208855373e-13,
            -8.9597492970451533e-16,
            1.1421575296031385e-14,
        ],
        dtype=np.float64,
    )

    # Coefficients of the Chebyshev series interpolating 1/(2 Pi) Sin(2 Pi K x)
    if cryptoContext.secretKeyDist == "SPARSE_TERNARY":
        coefficients = coefficientsSparse
    else:
        coefficients = coefficientsUniform
            
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
        T[0] = homo_ops.homo_mul_scalar_double(x, alpha, cryptoContext)
        T[0] = homo_ops.homo_rescale(T[0], 1, cryptoContext)
    beta = 2 * a / (b - a)
    if not math.isclose(beta, -1.0):
        T[0] = homo_ops.homo_add_scalar_double(T[0], -1.0 - beta, cryptoContext)

    for i in range(2, k + 1):
        prod = homo_ops.homo_mul(T[i // 2 - 1], T[(i + 1) // 2 - 1], cryptoContext)
        tmp = homo_ops.homo_add(prod, prod, cryptoContext)
        tmp = homo_ops.homo_rescale(tmp, 1, cryptoContext)
        if i & 1 == 1:  # i is odd
            tmp = homo_ops.homo_sub(tmp, T[0], cryptoContext)
        else:
            tmp = homo_ops.homo_add_scalar_double(tmp, -1.0, cryptoContext)
        T.append(tmp)

    for i in range(k):
        T[i] = homo_ops.adjust_to(T[i], T[-1].cur_limbs, T[-1].noise_deg, T[-1].scaling_factor, cryptoContext)

    # Compute the Chebyshev polynomials T_k(y), T_{2k}(y), T_{4k}(y), ... , T_{2^{m-1}k}(y)
    # T2[0] is used as a placeholder
    T2 = [T[-1]]
    for i in range(1, m):
        tmp = homo_ops.homo_square(T2[i - 1], cryptoContext)
        tmp = homo_ops.homo_add(tmp, tmp, cryptoContext)
        tmp = homo_ops.homo_rescale(tmp, 1, cryptoContext)
        tmp = homo_ops.homo_add_scalar_double(tmp, -1.0, cryptoContext)
        T2.append(tmp)



    # computes T_{k(2*m - 1)}(y)
    T2km1 = T2[0]
    for i in range(1, m):
        # compute T_{k(2*m - 1)} = 2*T_{k(2^{m-1}-1)}(y)*T_{k*2^{m-1}}(y) - T_k(y)
        prod = homo_ops.homo_mul(T2km1, T2[i], cryptoContext)
        T2km1 = homo_ops.homo_add(prod, prod, cryptoContext)
        T2km1 = homo_ops.homo_rescale(T2km1, 1, cryptoContext)
        T2km1 = homo_ops.homo_sub(T2km1, T2[0], cryptoContext)



    dc = degree(divcs_q)
    flag_c = False
    if dc >= 1:
        if dc == 1:
            if divcs_q[1] != 1:
                cu = homo_ops.homo_mul_scalar_double(T[0], divcs_q[1], cryptoContext)
                cu = homo_ops.homo_rescale(cu, 1, cryptoContext)
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
        qu = homo_ops.homo_add_scalar_double(qu, divqr_q[0] / 2, cryptoContext)
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
            su = homo_ops.homo_add(su, T[k - 1], cryptoContext)
        else:
            su = T[k - 1]
        # adds the free term (at x^0)
        su = homo_ops.homo_add_scalar_double(su, s2[0] / 2, cryptoContext)
        # The number of levels of su is the same as the number of levels of T[k-1] + 1.
        # Will only get here when m = 2, so need to reduce the number of levels by 1.

    if flag_c:
        result = homo_ops.homo_add(T2[m - 1], cu, cryptoContext)
    else:
        result = homo_ops.homo_add_scalar_double(T2[m - 1], divcs_q[0] / 2, cryptoContext)



    result = homo_ops.homo_mul(result, qu, cryptoContext)
    result = homo_ops.homo_rescale(result, 1, cryptoContext)
    result = homo_ops.homo_add(result, su, cryptoContext)


    result = homo_ops.homo_sub(result, T2km1, cryptoContext)



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
