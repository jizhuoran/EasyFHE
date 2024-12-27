import math, time
import torch
import numpy as np
from .Ciphertext import Cipher
from .context import *
from . import functional as F
from . import homo_ops
import pickle
import torch.profiler
from torch.profiler import ProfilerActivity, tensorboard_trace_handler
from .client import client as client


# Global dictionary to accumulate execution time for each function
execution_times = {}

def profile_python_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Calculate the execution time for this call
        exec_time = end_time - start_time
        
        # Update the global dictionary with the accumulated time for this function
        if func.__name__ not in execution_times:
            execution_times[func.__name__] = 0
        execution_times[func.__name__] += exec_time

        # print(f"Function {func.__name__} executed in {exec_time:.6f} seconds")
        return result
    return wrapper


def profile_pytorch_function(func):
    def wrapper(*args, **kwargs):
        # Set up the profiler
        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler('/home/zrji/log'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as profiler:
            result = func(*args, **kwargs)
        return result
    return wrapper


BS_CONST_DIR = "/home/zrji/GPU-FHE/torch/fhe/data/"

Tensor = torch.Tensor
NORMAL_CIPHER_SIZE = 2
BASE_NUM_LEVELS_TO_DROP = 1
# ENCRYPTION = 0
# MULTIPLICATION = 1
CONJUGATION = 2
R_UNIFORM = 6  # number of double-angle iterations in CKKS bootstrapping. Must be static because it is used in a static function.
R_SPARSE = 3  # number of double-angle iterations in CKKS bootstrapping. Must be static because it is used in a static function.
m_correctionFactor = 0  # correction factor, which we scale the message by to improve precision


# @profile_python_function
def degree(coefficients, poly_degree):
    deg = 1
    for i in range(poly_degree - 1, 0, -1):
        if coefficients[i] == 0:
            deg += 1
        else:
            break
    return poly_degree - deg


PREC = math.pow(2, -20)

# @profile_python_function
def is_not_equal_one(val):
    return val < 1 - PREC or val > 1 + PREC

# @profile_python_function
def long_division_chebyshev(f, f_len, g, g_len):
    n = degree(f, f_len)
    k = degree(g, g_len)

    if n != f_len - 1:
        raise ValueError("LongDivisionChebyshev: The dominant coefficient of the dividend is zero.")

    if k != g_len - 1:
        raise ValueError("LongDivisionChebyshev: The dominant coefficient of the divisor is zero.")

    r_len = f_len
    r = np.copy(f)  # Copy of f
    q_len = max(0, n - k + 1)
    q = np.zeros(q_len, dtype=np.float64)
    if (n - k) >= 0:
        q2_len = n - k + 1
        # q_len = q2_len
        q = np.zeros(q2_len)
        # q2 = np.zeros(q2_len)
        while n - k > 0:
            q[n - k] = 2 * r[r_len - 1]
            if is_not_equal_one(g[k]):
                q[n - k] /= g[g_len - 1]

            d_len = n + 1
            d = np.zeros(d_len)

            if k == n - k:
                d[0] = 2 * g[n - k]
                for i in range(1, 2 * k + 1):
                    d[i] = g[abs(n - k - i)]
            elif k > n - k:
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

            if is_not_equal_one(r[r_len - 1]):
                d *= r[r_len - 1]

            if is_not_equal_one(g[g_len - 1]):
                d /= g[g_len - 1]

            if r_len < d_len:
                raise ValueError("error: r_len < d_len!")

            r -= d[:r_len]  # Element-wise subtraction

            if r_len > 1:
                n = degree(r, r_len)
                r_len = n + 1
                r = r[:n + 1]  # Resize r

        if n == k:
            q[0] = r[r_len - 1]
            if is_not_equal_one(g[g_len - 1]):
                q[0] /= g[g_len - 1]

            d_len = g_len
            d = np.copy(g)
            if is_not_equal_one(r[r_len - 1]):
                d *= r[r_len - 1]
            if is_not_equal_one(g[g_len - 1]):
                d /= g[g_len - 1]

            if r_len < d_len:
                raise ValueError("error: r_len < d_len!")

            r -= d[:r_len]  # Element-wise subtraction

            if r_len > 1:
                n = degree(r, r_len)
                r_len = n + 1
                r = r[:n + 1]  # Resize r
        q[0] *= 2  # Adjust the first coefficient
    else:
        q_len = 1
        q = np.zeros(q_len)
        if r_len < f_len:
            raise ValueError("error: r_len < d_len!")
        r = np.copy(f)

    return q, q_len, r, r_len

# @profile_python_function
def eval_linear_wsum_mutable(ciphertexts, ciphertexts_num, constants, cryptoContext: Context):
    minLevel = ciphertexts[0].cur_limbs
    minIdx = 0
    for i in range(1, ciphertexts_num):
        if ciphertexts[i].cur_limbs < minLevel:
            minLevel = ciphertexts[i].cur_limbs
            minIdx = i
    for i in range(minIdx):
        if ciphertexts[i].cur_limbs < minLevel:
            ciphertexts[i] = homo_ops.cipher_level_reduce(ciphertexts[i], ciphertexts[i].cur_limbs-minLevel)
    for i in range(minIdx + 1, ciphertexts_num):
        if ciphertexts[i].cur_limbs < minLevel:
            ciphertexts[i] = homo_ops.cipher_level_reduce(ciphertexts[i], ciphertexts[i].cur_limbs-minLevel)
    wsum = eval_mult_in_place(ciphertexts[0], constants[0],
                              cryptoContext)
    for i in range(1, ciphertexts_num):
        tmp = eval_mult_in_place(ciphertexts[i], constants[i],
                                 cryptoContext)
        wsum = homo_ops.cipher_add(wsum, tmp, cryptoContext)
    wsum = homo_ops.cipher_mod_reduce(wsum, 1, cryptoContext)
    return wsum

# @profile_python_function
def check_and_adjust_level(ct1: Cipher, ct2: Cipher, cryptoContext: Context):
    rct1 = Cipher([ct1.cv[0].clone(), ct1.cv[1].clone()], ct1.cur_limbs)
    rct2 = Cipher([ct2.cv[0].clone(), ct2.cv[1].clone()], ct2.cur_limbs)

    if rct1.cur_limbs > rct2.cur_limbs:
        rct1=homo_ops.cipher_level_reduce(rct1, rct1.cur_limbs - rct2.cur_limbs)
    elif rct1.cur_limbs < rct2.cur_limbs:
        rct2=homo_ops.cipher_level_reduce(rct2, rct2.cur_limbs - rct1.cur_limbs)
    return rct1, rct2

# @profile_python_function
def inner_eval_chebyshev_ps(coefficients, coefficients_len,
                            k, m, T, T2, cryptoContext: Context):
    # Compute k * 2^(m-1) - k
    k2m2k = k * (1 << (m - 1)) - k

    # Initialize Tkm
    Tkm_len = int(k2m2k + k) + 1
    Tkm = np.zeros(Tkm_len)
    Tkm[-1] = 1.0  # Tkm.back() = 1

    # Divide coefficients by T^k*2^(m-1)
    divqr_q, divqr_q_len, divqr_r, divqr_r_len = long_division_chebyshev(coefficients, coefficients_len, Tkm, Tkm_len)

    # Subtract x^(k(2^(m-1) - 1)) from r
    if int(k2m2k - degree(divqr_r, divqr_r_len)) <= 0:
        divqr_r[int(k2m2k)] -= 1
        r2_len = degree(divqr_r, divqr_r_len) + 1
        r2 = np.zeros(r2_len)
        r2[:min(divqr_r_len, r2_len)] = divqr_r[:min(divqr_r_len, r2_len)]
        divqr_r[int(k2m2k)] += 1
    else:
        r2_len = int(k2m2k + 1)
        r2 = np.zeros(r2_len)
        r2[:min(divqr_r_len, r2_len)] = divqr_r[:min(divqr_r_len, r2_len)]
        r2[-1] = -1

    # Divide r2 by q
    divcs_q, divcs_q_len, divcs_r, divcs_r_len = long_division_chebyshev(r2, r2_len, divqr_q, divqr_q_len)

    # Add x^(k(2^(m-1) - 1)) to s
    s2_len = int(k2m2k + 1)
    s2 = np.zeros(s2_len)
    s2[:min(divcs_r_len, s2_len)] = divcs_r[:min(divcs_r_len, s2_len)]
    s2[-1] = 1.0

    # Evaluate c at u
    dc = degree(divcs_q, divcs_q_len)
    flag_c = False
    if dc >= 1:
        if dc == 1:
            if divcs_q[1] != 1:
                cu = eval_mult_in_place(T[0], divcs_q[1], cryptoContext)
                cu = homo_ops.cipher_mod_reduce(cu, 1, cryptoContext)
            else:
                cu = T[0]
        else:
            ctxs = [T[i] for i in range(dc)]
            weights = divcs_q[1:dc + 1]
            cu = eval_linear_wsum_mutable(ctxs, dc, weights, cryptoContext)

        cu = homo_ops.homo_add_scalar_double(cu, divcs_q[0] / 2, cryptoContext)
        cu = homo_ops.cipher_level_reduce(cu, cu.cur_limbs - T2[m - 1].cur_limbs)
        flag_c = True

    # Evaluate q and s2 at u
    if degree(divqr_q, divqr_q_len) > k:
        qu = inner_eval_chebyshev_ps(divqr_q, divqr_q_len, k, m - 1, T, T2, cryptoContext)
    else:
        qcopy = np.zeros(k)
        qcopy_len = k
        qcopy[:min(divqr_q_len, qcopy_len)] = divqr_q[:min(divqr_q_len, qcopy_len)]
        deg_qcopy = degree(qcopy, qcopy_len)
        if deg_qcopy > 0:
            ctxs = [T[i] for i in range(deg_qcopy)]
            weights = divqr_q[1:deg_qcopy + 1]
            qu = eval_linear_wsum_mutable(ctxs, deg_qcopy, weights, cryptoContext)
            sum = T[k - 1]
            for i in range(int(math.log2(divqr_q[divqr_q_len - 1]))):
                sum = homo_ops.cipher_add(sum, sum, cryptoContext)
            qu, sum = check_and_adjust_level(qu, sum, cryptoContext)
            qu = homo_ops.cipher_add(qu, sum, cryptoContext)
        else:
            sum = T[k - 1]
            for i in range(int(math.log2(divqr_q[divqr_q_len - 1]))):
                sum = homo_ops.cipher_add(sum, sum, cryptoContext)
            qu = sum

        qu = homo_ops.homo_add_scalar_double(qu, divqr_q[0] / 2, cryptoContext)

    # Evaluate s2 at u
    if degree(s2, s2_len) > k:
        su = inner_eval_chebyshev_ps(s2, s2_len, k, m - 1, T, T2, cryptoContext)
    else:
        scopy_len = k
        scopy = np.zeros(scopy_len)
        scopy[:min(s2_len, scopy_len)] = s2[:min(s2_len, scopy_len)]
        deg_scopy = degree(scopy, scopy_len)
        if deg_scopy > 0:
            ctxs = [T[i] for i in range(deg_scopy)]
            weights = s2[1:deg_scopy + 1]
            su = eval_linear_wsum_mutable(ctxs, deg_scopy, weights, cryptoContext)
            if T[k - 1].cur_limbs > su.cur_limbs:
                su, tmp_T = check_and_adjust_level(su, T[k - 1], cryptoContext)
                su = homo_ops.cipher_add(su, tmp_T, cryptoContext)
            else:
                su, T[k - 1] = check_and_adjust_level(su, T[k - 1], cryptoContext)
                su = homo_ops.cipher_add(su, T[k - 1], cryptoContext)
        else:
            su = T[k - 1]

        su = homo_ops.homo_add_scalar_double(su, s2[0] / 2, cryptoContext)
        su = homo_ops.cipher_level_reduce(su, 1)

    if flag_c:
        T2[m - 1], cu = check_and_adjust_level(T2[m - 1], cu, cryptoContext)
        result = homo_ops.cipher_add(T2[m - 1], cu, cryptoContext)
    else:
        result = homo_ops.homo_add_scalar_double(T2[m - 1], divcs_q[0] / 2, cryptoContext)

    result, qu = check_and_adjust_level(result, qu, cryptoContext)
    result = homo_ops.homo_mul(result, qu, cryptoContext)
    result = homo_ops.cipher_mod_reduce(result, 1, cryptoContext)
    result, su = check_and_adjust_level(result, su, cryptoContext)
    result = homo_ops.cipher_add(result, su, cryptoContext)

    return result

# @profile_python_function
def eval_chebyshev_series_ps(x, coefficients, a, b, cryptoContext):

    coefficients_len = len(coefficients)
    deg = 1
    for i in range(coefficients_len - 1, 0, -1):
        if coefficients[i] == 0:
            deg += 1
        else:
            break
    n = coefficients_len - deg

    f2 = np.copy(coefficients)
    f2_len = coefficients_len

    if coefficients[coefficients_len - 1] == 0:
        f2_len = n + 1

    klist = []
    mlist = []

    sqn2 = math.sqrt(n / 2)

    for k in range(1, n + 1):
        for m in range(1, math.ceil(math.log2(n / k)) + 2):
            if n - k * ((1 << m) - 1) < 0:
                if -2 <= k - sqn2 <= 2:
                    klist.append(k)
                    mlist.append(m)

    min_index = mlist.index(min(mlist)) if mlist else -1

    degs = [klist[min_index], mlist[min_index]] if min_index != -1 else []

    k, m = degs[0], degs[1]

    # Linear transformation
    if a == -1 and b == 1:
        y = x  # y == T[0]
    else:
        alpha = 2 / (b - a)
        beta = 2 * a / (b - a)
        y = eval_mult_in_place(x, alpha, cryptoContext)
        y = homo_ops.cipher_mod_reduce(y, 1, cryptoContext)
        y = homo_ops.homo_add_scalar_double(y, -1.0 - beta, cryptoContext)

    T = [0 for _ in range(k)]
    T[0] = y
    # T = [Cipher([x.cv[0].clone(), x.cv[1].clone()], x.cur_limbs) for _ in range(k)]
    # T[0] = y

    for i in range(2, k + 1):
        if i & (i - 1) == 0:  # i is a power of 2
            square = homo_ops.homo_square(T[i // 2 - 1], cryptoContext)
            T[i - 1] = homo_ops.homo_add(square, square, cryptoContext)
            T[i - 1] = homo_ops.cipher_mod_reduce(T[i - 1], 1, cryptoContext)
            T[i - 1] = homo_ops.homo_add_scalar_double(T[i - 1], -1.0, cryptoContext)

        else:
            if i % 2 == 1:  # i is odd
                tmpct1, tmpct2 = check_and_adjust_level(T[i // 2 - 1], T[i // 2], cryptoContext)
                prod = homo_ops.homo_mul(tmpct1, tmpct2, cryptoContext)
                T[i - 1] = homo_ops.homo_add(prod, prod, cryptoContext)
                T[i - 1] = homo_ops.cipher_mod_reduce(T[i - 1], 1, cryptoContext)
                T[i - 1], tmpct2 = check_and_adjust_level(T[i - 1], y, cryptoContext)
                T[i - 1] = homo_ops.homo_sub(T[i - 1], tmpct2, cryptoContext)

            else:  # i is even
                square = homo_ops.homo_square(T[i // 2 - 1], cryptoContext)
                T[i - 1] = homo_ops.homo_add(square, square, cryptoContext)
                T[i - 1] = homo_ops.cipher_mod_reduce(T[i - 1], 1, cryptoContext)
                T[i - 1] = homo_ops.homo_add_scalar_double(T[i - 1], -1.0, cryptoContext)

    # Adjust levels of T
    for i in range(1, k):
        level_diff = T[i - 1].cur_limbs - T[k - 1].cur_limbs
        T[i - 1] = homo_ops.cipher_level_reduce(T[i - 1], level_diff)

    # T2 = [Cipher([T[0].cv[0].clone(), T[0].cv[1].clone()], T[0].cur_limbs) for _ in range(m)]
    T2 = [0 for _ in range(m)]
    T2[0] = T[k - 1]

    for i in range(1, m):
        square = homo_ops.homo_square(T2[i - 1], cryptoContext)
        T2[i] = homo_ops.homo_add(square, square, cryptoContext)
        T2[i] = homo_ops.cipher_mod_reduce(T2[i], 1, cryptoContext)
        T2[i] = homo_ops.homo_add_scalar_double(T2[i], -1.0, cryptoContext)

    T2km1 = T2[0]
    for i in range(1, m):
        tmpct1, tmpct2 = check_and_adjust_level(T2km1, T2[i], cryptoContext)
        prod = homo_ops.homo_mul(tmpct1, tmpct2, cryptoContext)
        T2km1 = homo_ops.homo_add(prod, prod, cryptoContext)
        T2km1 = homo_ops.cipher_mod_reduce(T2km1, 1, cryptoContext)
        T2km1, tmpct2 = check_and_adjust_level(T2km1, T2[0], cryptoContext)
        T2km1 = homo_ops.homo_sub(T2km1, tmpct2, cryptoContext)

    # Compute k*2^{m-1}-k
    k2m2k = k * (1 << (m - 1)) - k

    # Initialize f2
    new_f2_len = 2 * k2m2k + k + 1
    if f2_len < new_f2_len:
        new_f2 = np.zeros(new_f2_len)
        for i in range(f2_len):
            new_f2[i] = f2[i]
        # f2 = None
        for i in range(f2_len, new_f2_len):
            new_f2[i] = 0.0
        new_f2[new_f2_len - 1] = 1
        f2_len = new_f2_len
        f2 = new_f2
    else:
        f2_len = new_f2_len
        f2[f2_len - 1] = 1

    # Divide f2 by T^{k*2^{m-1}}
    Tkm_len = k2m2k + k + 1
    Tkm = np.zeros(Tkm_len)
    Tkm[Tkm_len - 1] = 1

    divqr_q, divqr_q_len, divqr_r, divqr_r_len = long_division_chebyshev(f2, f2_len, Tkm, Tkm_len)
    TKm = None
    f2 = None

    if k2m2k - degree(divqr_r, divqr_r_len) <= 0:
        divqr_r[k2m2k] -= 1
        r2_len = degree(divqr_r, divqr_r_len) + 1
        r2 = np.zeros(r2_len)
        r2[:min(len(divqr_r), r2_len)] = divqr_r[:min(len(divqr_r), r2_len)]
        divqr_r[k2m2k] += 1
    else:
        r2_len = k2m2k + 1
        r2 = np.zeros(r2_len)
        r2[:min(len(divqr_r), r2_len)] = divqr_r[:min(len(divqr_r), r2_len)]
        r2[r2_len - 1] = -1

    # Divide r2 by q
    divcs_q, divcs_q_len, divcs_r, divcs_r_len = long_division_chebyshev(r2, r2_len, divqr_q, divqr_q_len)
    r2 = None
    # Add x^{k(2^{m-1} - 1)} to s
    s2_len = k2m2k + 1
    s2 = np.zeros(k2m2k + 1)
    s2[:min(len(divcs_r), s2_len)] = divcs_r[:min(len(divcs_r), s2_len)]
    s2[-1] = 1

    # Evaluate c at u
    cu = None
    dc = degree(divcs_q, divcs_q_len)
    flag_c = False

    if dc >= 1:
        if dc == 1:
            if divcs_q[1] != 1:
                cu = eval_mult_in_place(T[0], divcs_q[1], cryptoContext)
                cu = homo_ops.cipher_mod_reduce(cu, 1, cryptoContext)
            else:
                cu = T[0]
        else:
            ctxs = [T[i] for i in range(dc)]
            weights = divcs_q[1:dc + 1]
            cu = eval_linear_wsum_mutable(ctxs, dc, weights, cryptoContext)
            ctxs = None
            weights = None
        cu = homo_ops.homo_add_scalar_double(cu, divcs_q[0] / 2, cryptoContext)
        flag_c = True

    # Evaluate q and s2 at u
    qu = None
    if degree(divqr_q, divqr_q_len) > k:
        qu = inner_eval_chebyshev_ps(divqr_q, divqr_q_len, k, m - 1, T, T2, cryptoContext)
    else:
        qcopy_len = k
        qcopy = np.zeros(qcopy_len)
        qcopy[:min(len(divqr_q), qcopy_len)] = divqr_q[:qcopy_len]

        deg_qcopy = degree(qcopy, qcopy_len)
        if deg_qcopy > 0:
            ctxs = [T[i] for i in range(deg_qcopy)]
            weights = divqr_q[1:deg_qcopy + 1]
            qu = eval_linear_wsum_mutable(ctxs, deg_qcopy, weights, cryptoContext)
            sum = homo_ops.cipher_add(T[k - 1], T[k - 1], cryptoContext)
            qu, sum, = check_and_adjust_level(qu, sum, cryptoContext)
            qu = homo_ops.cipher_add(qu, sum, cryptoContext)
        else:
            qu = T[k - 1]
            for _ in range(1, divqr_q[divqr_q_len - 1]):
                qu = homo_ops.cipher_add(qu, T[k - 1], cryptoContext)
        qu = homo_ops.cipher_add_scalar(qu, divqr_q[0] / 2, cryptoContext)

    # Evaluate s2 at u
    su = None
    deg_s2 = degree(s2, s2_len)
    if deg_s2 > k:
        su = inner_eval_chebyshev_ps(s2, s2_len, k, m - 1, T, T2, cryptoContext)
    else:
        scopy_len = k
        scopy = np.zeros(scopy_len)
        scopy[:min(len(s2), scopy_len)] = s2[:scopy_len]

        deg_scopy = degree(scopy, scopy_len)
        if deg_scopy > 0:
            ctxs = [T[i] for i in range(deg_scopy)]
            weights = s2[1:deg_scopy + 1]
            su = eval_linear_wsum_mutable(ctxs, deg_scopy, weights, cryptoContext)
            if T[k - 1].cur_limbs > su.cur_limbs:
                su, tmp_T = check_and_adjust_level(su, T[k - 1], cryptoContext)
                su = homo_ops.cipher_add(su, tmp_T, cryptoContext)
            else:
                su, T[k - 1] = check_and_adjust_level(su, T[k - 1], cryptoContext)
                su = homo_ops.cipher_add(su, T[k - 1], cryptoContext)

        else:
            su = T[k - 1]

        su = homo_ops.cipher_add_scalar(su, s2[0] / 2, cryptoContext)
        scopy = None

    # Final result computation
    if flag_c:
        T2[m - 1], cu = check_and_adjust_level(T2[m - 1], cu, cryptoContext)
        result = homo_ops.cipher_add(T2[m - 1], cu, cryptoContext)
    else:
        result = homo_ops.cipher_add_scalar(T2[m - 1], divcs_q[0] / 2, cryptoContext)

    result, qu = check_and_adjust_level(result, qu, cryptoContext)
    result = homo_ops.homo_mul(result, qu, cryptoContext)
    result = homo_ops.cipher_mod_reduce(result, 1, cryptoContext)
    result, su = check_and_adjust_level(result, su, cryptoContext)
    result = homo_ops.homo_add(result, su, cryptoContext)
    result, T2km1 = check_and_adjust_level(result, T2km1, cryptoContext)
    result = homo_ops.homo_sub(result, T2km1, cryptoContext)

    return result


# @profile_python_function
def eval_fast_rotation_precompute(input, curr_limbs, cryptoContext):
    res = F.cv_modup(input, curr_limbs, cryptoContext)
    return res.clone()

def eval_fast_key_switch_core_ext(d2Tilde, auto_index, key_map, expand_length, beta, curr_limbs, cryptoContext):
    swk = cryptoContext.left_rot_key_map[str(auto_index)]
    swk_bx = swk[0][:beta, :, :]
    swk_ax = swk[1][:beta, :, :]
    
    res = F.cv_innerproduct(
        d2Tilde.reshape(-1),
        curr_limbs=curr_limbs,
        context_cuda=cryptoContext,
        swk_bx=swk_bx,
        swk_ax=swk_ax
    )
    return res[1], res[0]

# @profile_python_function
def eval_fast_rotation_ext(bx, digits, curr_limbs, index, add_first, cryptoContext):
    alpha = cryptoContext.K
    logN = cryptoContext.logN
    K = cryptoContext.K
    beta = int(np.ceil(curr_limbs / alpha))  # Calculate beta as per the original C++ code

    # Find the automorphism index that corresponds to rotation index.
    auto_index = cryptoContext.BsContext.auto_index[index]

    expand_limbs = curr_limbs + K
    expand_length = expand_limbs << logN

    # Inner Product
    sumaxmult, sumbxmult = eval_fast_key_switch_core_ext(digits, auto_index, cryptoContext.left_rot_key_map,
                                                         expand_length, beta, curr_limbs, cryptoContext)

    if (add_first):
        cMult = F.cv_mul_scalar(bx, cryptoContext.PModq_cuda, cryptoContext.moduliQ_cuda,
                                cryptoContext.q_mu_cuda, curr_limbs)
        sumbxmult = F.cv_add(sumbxmult, cMult, cryptoContext.moduliQ_cuda, curr_limbs, inplace=True)

    cv0 = F.cv_automorphism_transform(sumbxmult, expand_limbs, auto_index, cryptoContext)
    cv1 = F.cv_automorphism_transform(sumaxmult, expand_limbs, auto_index, cryptoContext)
    return Cipher([cv0, cv1], curr_limbs)

# @profile_python_function
def key_switch_ext(cipher, cipher_size, add_first, cryptoContext):

    assert cipher_size == 2 # Only 2-dim ciphertexts are supported
    curr_limbs = cipher.cur_limbs
    N = cryptoContext.N
    logN = cryptoContext.logN
    K = cryptoContext.K

    if add_first:
        cv0 = F.cv_mul_scalar(cipher.cv[0], cryptoContext.PModq_cuda, cryptoContext.moduliQ_cuda,
                                       cryptoContext.q_mu_cuda,
                                       curr_limbs)
    else:
        # If not adding the first, we ensure bx is zero-initialized
        cv0 = torch.zeros(((curr_limbs + K) << logN), dtype=torch.uint64, device="cuda").reshape(-1, N)

    cv1 = F.cv_mul_scalar(cipher.cv[1], cryptoContext.PModq_cuda, cryptoContext.moduliQ_cuda,
                                   cryptoContext.q_mu_cuda,
                                   curr_limbs)
    return Cipher([cv0, cv1], curr_limbs)

# @profile_python_function
def eval_mult_ext(cipher, pt, cryptoContext):
    cur_limbs = cipher.cur_limbs
    # Perform the multiplication on ax and bx components
    moduli = cryptoContext.BsContext.QplusP_map[cur_limbs]
    mu = cryptoContext.BsContext.QmuplusPmu_map[cur_limbs]
    cv1 = F.cv_mul(cipher.cv[1], pt.mx.reshape(-1, cryptoContext.N), moduli, mu, cipher.cv[0].shape[0])
    cv0 = F.cv_mul(cipher.cv[0], pt.mx.reshape(-1, cryptoContext.N), moduli, mu, cipher.cv[0].shape[0])
    return Cipher([cv0, cv1], cur_limbs)
#
# @profile_python_function
def eval_add_ext(cipher0, cipher1, cryptoContext):
    assert cipher0.cur_limbs == cipher1.cur_limbs
    cur_limbs = min(cipher0.cv[0].shape[0], cipher1.cv[0].shape[0])
    moduli = cryptoContext.BsContext.QplusP_map[cipher0.cur_limbs]
    cv = [
        F.cv_add(cv0, cv1, moduli, cur_limbs, inplace=True)
        for cv0, cv1 in zip(cipher0.cv, cipher1.cv)
    ]
    return Cipher(cv, cipher0.cur_limbs)

# @profile_python_function
def key_switch_down(sumaxmult, sumbxmult, curr_limbs, cryptoContext):
    res_ax = F.cv_moddown(sumaxmult, curr_limbs, cryptoContext)
    res_bx = F.cv_moddown(sumbxmult, curr_limbs, cryptoContext)
    return Cipher([res_bx, res_ax], curr_limbs)

# @profile_python_function
def add_and_equal(in0, in1, cur_limbs, cryptoContext):
    # moduli = torch.from_numpy(
        # np.concatenate((cryptoContext.moduliQ[0:cur_limbs], cryptoContext.moduliP[0:cryptoContext.K]))).cuda()
    moduli = cryptoContext.BsContext.QplusP_map[cur_limbs]
    res = F.cv_add(in0, in1, moduli, in0.shape[0])
    return res

def merged_function(A, ctxt, cryptoContext, flag_rem, rot_in, rot_out, config):

    special_limbs = cryptoContext.K
    logN = cryptoContext.logN
    N = cryptoContext.N
    M = N << 1

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

    result = Cipher([ctxt.cv[0].clone(), ctxt.cv[1].clone()], ctxt.cur_limbs)

    # Determine loop range based on direction
    if loop_direction == "forward":
        loop_range = range(start, stop)
    else:
        loop_range = range(start, stop, -1)
    
    for s in loop_range:
        if (loop_direction == "forward" and s != 0) or (loop_direction == "backward" and s != level_budget -1):
            result = homo_ops.cipher_mod_reduce(result, cipher_mod_levels, cryptoContext)
        
        curr_limbs = result.cur_limbs
        limbs_ext = curr_limbs + special_limbs
        len_ext = limbs_ext << logN
        len_ = curr_limbs << logN
        alpha = cryptoContext.K
        beta = (curr_limbs + alpha - 1) // alpha
        
        digits_len = beta * len_ext
        digits = eval_fast_rotation_precompute(result.cv[1], result.cur_limbs, cryptoContext)
        
        
        fast_rotation_ext = [None for _ in range(g)]
        
        for j in range(g):
            if rot_in[s][j] != 0:
                cv0 = result.cv[0].reshape(-1, cryptoContext.N) if eval_fast_rotation_reshape else result.cv[0]
                fast_rotation_ext[j] = eval_fast_rotation_ext(
                    cv0, digits, result.cur_limbs, rot_in[s][j], True, cryptoContext
                )
            else:
                fast_rotation_ext[j] = key_switch_ext(result, key_switch_ext_size, True, cryptoContext)
        
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
                    inner_ks_down = key_switch_down(inner.cv[1], inner.cv[0], curr_limbs, cryptoContext)
                    auto_index = cryptoContext.BsContext.auto_index[rot_out[s][i]]

                    first_current = F.cv_automorphism_transform(
                        inner_ks_down.cv[0], curr_limbs, auto_index, cryptoContext)
                    first = add_and_equal(first, first_current, curr_limbs, cryptoContext)
                    
                    inner_digits = eval_fast_rotation_precompute(
                        inner_ks_down.cv[1], inner_ks_down.cur_limbs, cryptoContext
                    )
                    
                    inner_ks_down_ext = eval_fast_rotation_ext(
                        None, inner_digits, inner_ks_down.cur_limbs, rot_out[s][i], False, cryptoContext
                    )
                    outer = eval_add_ext(outer, inner_ks_down_ext, cryptoContext)
                else:
                    tmp_first = F.cv_moddown(inner.cv[0], curr_limbs, cryptoContext)
                    first = add_and_equal(first, tmp_first, curr_limbs, cryptoContext)
                    F.cv_set_zero(inner.cv[0], len_ext)
                    outer = eval_add_ext(outer, inner, cryptoContext)
        
        result = key_switch_down(outer.cv[1], outer.cv[0], curr_limbs, cryptoContext)
        result.cv[0] = add_and_equal(result.cv[0], first, curr_limbs, cryptoContext)
    
    if flag_rem:
        result = homo_ops.cipher_mod_reduce(result, cipher_mod_levels, cryptoContext)
        curr_limbs = result.cur_limbs
        limbs_ext = curr_limbs + special_limbs
        len_ext = limbs_ext << logN
        len_ = curr_limbs << logN
        alpha = cryptoContext.K
        beta = (curr_limbs + alpha - 1) // alpha
        
        digits_len = beta * len_ext
        digits = eval_fast_rotation_precompute(result.cv[1], result.cur_limbs, cryptoContext)
        
        fast_rotation_ext = [None for _ in range(g_rem)]
        
        s = stop if loop_direction == "backward" else level_budget - flag_rem
        
        for j in range(g_rem):
            if rot_in[s][j] != 0:
                fast_rotation_ext[j] = eval_fast_rotation_ext(
                    result.cv[0], digits, result.cur_limbs, rot_in[s][j], True, cryptoContext
                )
            else:
                fast_rotation_ext[j] = key_switch_ext(result, key_switch_ext_size, True, cryptoContext)
        
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
                    inner_ks_down = key_switch_down(inner.cv[1], inner.cv[0], curr_limbs, cryptoContext)
                    auto_index = cryptoContext.BsContext.auto_index[rot_out[s][i]]
                    map_tensor = cryptoContext.BsContext.precompute_auto_map[auto_index]

                    # map_tensor = cryptoContext.compute_auto_map(N, auto_index, None)
                    
                    first_current = F.cv_automorphism_transform(
                        inner_ks_down.cv[0], curr_limbs, auto_index, cryptoContext)
                    first = add_and_equal(first, first_current, curr_limbs, cryptoContext)
                    
                    inner_digits = eval_fast_rotation_precompute(
                        inner_ks_down.cv[1], inner_ks_down.cur_limbs, cryptoContext
                    )
                    
                    inner_ks_down_ext = eval_fast_rotation_ext(
                        None, inner_digits, inner_ks_down.cur_limbs, rot_out[s][i], False, cryptoContext
                    )
                    outer = eval_add_ext(outer, inner_ks_down_ext, cryptoContext)
                else:
                    tmp_first = F.cv_moddown(inner.cv[0], curr_limbs, cryptoContext)
                    first = add_and_equal(first, tmp_first, curr_limbs, cryptoContext)
                    F.cv_set_zero(inner.cv[0], len_ext)
                    outer = eval_add_ext(outer, inner, cryptoContext)
        
        result = key_switch_down(outer.cv[1], outer.cv[0], curr_limbs, cryptoContext)
        result.cv[0] = add_and_equal(result.cv[0], first, curr_limbs, cryptoContext)
    
    return result


# @profile_python_function
def eval_coeffs_to_slots(A, A_len, ctxt, cryptoContext):

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
def eval_slots_to_coeffs(A, A_len, ctxt, cryptoContext):

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
def get_element_for_eval_mult(factors, cur_limbs, constant, cryptoContext):
    num_towers = cur_limbs
    p = cryptoContext.p
    q_vec = cryptoContext.moduliQ  # Assuming qVec is a numpy array

    sc_factor = p

    # Assuming DoubleInteger is equivalent to Python's int (arbitrary precision)
    MAX_BITS_IN_WORD = 126

    # Compute approxFactor
    log_sf = int(math.ceil(math.log2(math.fabs(sc_factor))))
    log_valid = log_sf if log_sf <= MAX_BITS_IN_WORD else MAX_BITS_IN_WORD
    log_approx = log_sf - log_valid
    approx_factor = pow(2, log_approx)

    large = int((constant / approx_factor * sc_factor) + 0.5)
    large_abs = abs(large)
    bound = 1 << 63

    if large_abs > bound:
        for i in range(num_towers):
            reduced = large % q_vec[i]
            factors[i] = reduced + q_vec[i] if reduced < 0 else reduced
    else:
        sc_constant = int(large)
        for i in range(num_towers):
            reduced = sc_constant % int(q_vec[i])
            factors[i] = reduced + q_vec[i] if reduced < 0 else reduced
    return factors

# @profile_python_function
def eval_mult_in_place(ciphertext, constant, cryptoContext):

    # print("eval_mult_in_place", "constant", constant)

    cur_limbs = ciphertext.cur_limbs
    factors = np.zeros(cur_limbs, dtype=np.uint64)

    # Generate the factors needed for multiplication
    factors = get_element_for_eval_mult(factors, cur_limbs, constant, cryptoContext)
    factors = torch.tensor(factors, dtype=torch.uint64, device="cuda")
    cv = [
        F.cv_mul_scalar(
            cv0, factors, cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda, ciphertext.cur_limbs
        )
        for cv0 in ciphertext.cv
    ]
    return Cipher(cv, ciphertext.cur_limbs)

# @profile_python_function
def adjust_ciphertext(cryptoContext, ciphertext, correction):
    rescale_tech = cryptoContext.BsContext.rescaleTech

    if rescale_tech == 'FLEXIBLEAUTO' or rescale_tech == 'FLEXIBLEAUTOEXT':
        # TODO: to be implemented
        pass
    else:
        # Scaling down the message by a correction factor
        cnst = math.pow(2, -correction)

        ciphertext = eval_mult_in_place(ciphertext, cnst, cryptoContext)

        ciphertext = homo_ops.cipher_mod_reduce(ciphertext, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
    return ciphertext

# @profile_python_function
def eval_linear_transform(A, A_len, ct, scheme):
    # TODO: to be implemented
    pass

# @profile_python_function
def apply_double_angle_iterations(ciphertext, cryptoContext):
    # Determine r based on the scheme's secretKeyDist attribute
    if cryptoContext.BsContext.secretKeyDist == SecretKeyDist.UNIFORM_TERNARY:
        r = R_UNIFORM
    elif cryptoContext.BsContext.secretKeyDist == SecretKeyDist.SPARSE_TERNARY:
        r = R_SPARSE
    else:
        raise ValueError("set secretKeyDist first!")

    for j in range(1, r + 1):
        # Equivalent of cc->EvalSquareInPlace(ciphertext);
        ciphertext = homo_ops.homo_square(ciphertext, cryptoContext)

        # Equivalent of cc->EvalAdd(ciphertext, ciphertext);
        ciphertext = homo_ops.cipher_add(ciphertext, ciphertext, cryptoContext)

        # Equivalent of ModReduceInternalInPlace(ciphertext, 1, scheme)
        ciphertext = homo_ops.cipher_mod_reduce(ciphertext, 1, cryptoContext)

        # Calculate scalar as per the formula
        scalar = -1.0 / math.pow((2.0 * math.pi), math.pow(2.0, j - r))

        # Equivalent of cc->EvalAddInPlace(ciphertext, scalar);
        ciphertext = homo_ops.homo_add_scalar_double(ciphertext, scalar, cryptoContext)
    return ciphertext

# @profile_python_function
def mult_by_monomial_and_equal(cipher, monomial_degree, cryptoContext):
    l = cipher.cur_limbs
    cipher.cv[0] = F.cv_mul_by_monomial(cryptoContext, cipher.cv[0], l, monomial_degree)
    cipher.cv[1] = F.cv_mul_by_monomial(cryptoContext, cipher.cv[1], l, monomial_degree)
    return cipher

# @profile_python_function
def switch_modulus_with_intt_ntt(input_tensor, l, cryptoContext):
    res = F.cv_switch_modulus(cryptoContext, input_tensor, l)
    return res

# @profile_python_function
def eval_bootstrap(cryptoContext, ciphertext, num_iterations, precision, rescaleTech, secretKeyDist, L0, slots):
    M = cryptoContext.M
    N = cryptoContext.N
    cryptoContext.slots = slots
    precom = cryptoContext.BsContext
    bs_ctx = cryptoContext.BsContext
    moduliQ = cryptoContext.moduliQ
    rescaleTech = precom.rescaleTech

    assert num_iterations == 1 # Only one iteration is supported
    assert rescaleTech == ScalingTechnique.FIXEDMANUAL # Only FIXEDMANUAL is supported

    q = moduliQ[0]
    q_double = float(q)

    p = cryptoContext.logp  # Equivalent to dcrbits in OpenFHE
    powP = 2 ** p
    deg = round(math.log2(q_double / powP))

    correction = cryptoContext.correctionFactor - deg  # fixme: originally a uint32_t in OpenFHE
    post = 2 ** deg
    pre = 1. / post
    scalar = round(post)

    tmp = adjust_ciphertext(cryptoContext, ciphertext, correction)
    cv0 = switch_modulus_with_intt_ntt(tmp.cv[0], L0, cryptoContext)  # bx
    cv1 = switch_modulus_with_intt_ntt(tmp.cv[1], L0, cryptoContext)  # ax
    raised = Cipher([cv0, cv1], L0)


    constantEvalMult = pre * (1.0 / (bs_ctx.k * N))

    ctxtDec = None  # Initialize decrypted ciphertext
    isLTBootstrap = (precom.paramsEnc.level_budget == 1) and (precom.paramsDec.level_budget == 1)

    raised = homo_ops.homo_mul_scalar_double(raised, constantEvalMult, cryptoContext)

    if slots == M // 4:
        raised = homo_ops.cipher_mod_reduce(raised, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if isLTBootstrap:
            ctxtEnc = eval_linear_transform(precom.m_U0hatTPre, precom.LTMatrix_Row, raised, cryptoContext)
        else:
            ctxtEnc = eval_coeffs_to_slots(precom.m_U0hatTPreFFT, cryptoContext.slots, raised,
                                           cryptoContext)  # slots全局固定


        conj = Cipher([ctxtEnc.cv[0].clone(), ctxtEnc.cv[1].clone()], ctxtEnc.cur_limbs)
        conj = homo_ops.homo_conjugate(conj, 2 * N - 1, cryptoContext)

        ctxtEncI = homo_ops.cipher_sub(ctxtEnc, conj, cryptoContext)
        ctxtEnc = homo_ops.cipher_add(ctxtEnc, conj, cryptoContext)
        ctxtEncI = mult_by_monomial_and_equal(ctxtEncI, 3 * M // 4, cryptoContext)

        if rescaleTech == ScalingTechnique.FIXEDMANUAL:
            ctxtEnc = homo_ops.cipher_mod_reduce(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
            ctxtEncI = homo_ops.cipher_mod_reduce(ctxtEncI, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        ctxtEnc_copy = Cipher([ctxtEnc.cv[0].clone(), ctxtEnc.cv[1].clone()], ctxtEnc.cur_limbs)  # ctxtEnc.copy()
        ctxtEncI_copy = Cipher([ctxtEncI.cv[0].clone(), ctxtEncI.cv[1].clone()], ctxtEncI.cur_limbs)  # ctxtEncI.copy()
        ctxtEnc = eval_chebyshev_series_ps(ctxtEnc_copy, bs_ctx.coefficients, -1, 1, cryptoContext)
        ctxtEncI = eval_chebyshev_series_ps(ctxtEncI_copy, bs_ctx.coefficients, -1, 1,
                                            cryptoContext)

        if secretKeyDist == SecretKeyDist.UNIFORM_TERNARY:
            if rescaleTech != ScalingTechnique.FIXEDMANUAL:
                ctxtEnc = homo_ops.cipher_mod_reduce(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
                ctxtEncI = homo_ops.cipher_mod_reduce(ctxtEncI, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
            ctxtEnc = apply_double_angle_iterations(ctxtEnc, cryptoContext)
            ctxtEncI = apply_double_angle_iterations(ctxtEncI, cryptoContext)

        ctxtEncI = mult_by_monomial_and_equal(ctxtEncI, M // 4, cryptoContext)
        ctxtEnc = homo_ops.cipher_add(ctxtEnc, ctxtEncI, cryptoContext)
        ctxtEnc = homo_ops.homo_mul_scalar_int(ctxtEnc, scalar, cryptoContext)

        if rescaleTech != ScalingTechnique.FIXEDMANUAL:
            ctxtEnc = homo_ops.cipher_mod_reduce(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if isLTBootstrap:
            ctxtDec = eval_linear_transform(precom.m_U0Pre, ctxtEnc, cryptoContext)
        else:
            ctxtDec = eval_slots_to_coeffs(precom.m_U0PreFFT, slots, ctxtEnc, cryptoContext)

    else:
        for step in range(int(math.log2(N // (2 * slots)))):
            auto_index = cryptoContext.BsContext.auto_index[(1 << step) * slots]
            temp = homo_ops.homo_rotate(raised, auto_index, cryptoContext)
            raised = homo_ops.cipher_add(raised, temp, cryptoContext)
        raised = homo_ops.cipher_mod_reduce(raised, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if isLTBootstrap:
            ctxtEnc = eval_linear_transform(precom.m_U0hatTPre, raised, cryptoContext)
        else:
            ctxtEnc = eval_coeffs_to_slots(precom.m_U0hatTPreFFT, cryptoContext.slots, raised, cryptoContext)


        conj = homo_ops.homo_conjugate(ctxtEnc, 2 * N - 1, cryptoContext)
        ctxtEnc = homo_ops.cipher_add(ctxtEnc, conj, cryptoContext)

        if rescaleTech == ScalingTechnique.FIXEDMANUAL:
            ctxtEnc = homo_ops.cipher_mod_reduce(ctxtEnc, 1, cryptoContext)

        ctxtEnc_copy = Cipher([ctxtEnc.cv[0].clone(), ctxtEnc.cv[1].clone()], ctxtEnc.cur_limbs)  # ctxtEnc.copy()
        ctxtEnc = eval_chebyshev_series_ps(ctxtEnc_copy, bs_ctx.coefficients, -1, 1, cryptoContext)

        if secretKeyDist == SecretKeyDist.UNIFORM_TERNARY:
            if rescaleTech != ScalingTechnique.FIXEDMANUAL:
                ctxtEnc = homo_ops.cipher_mod_reduce(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
            ctxtEnc = apply_double_angle_iterations(ctxtEnc, cryptoContext)

        ctxtEnc = homo_ops.homo_mul_scalar_int(ctxtEnc, scalar, cryptoContext)

        if rescaleTech != ScalingTechnique.FIXEDMANUAL:
            ctxtEnc = homo_ops.cipher_mod_reduce(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if isLTBootstrap:
            ctxtDec = eval_linear_transform(precom.m_U0Pre, ctxtEnc, cryptoContext)
        else:
            ctxtDec = eval_slots_to_coeffs(precom.m_U0PreFFT, slots, ctxtEnc, cryptoContext)


        auto_index = cryptoContext.BsContext.auto_index[slots]
        ctxtDec_rot = homo_ops.homo_rotate(ctxtDec, auto_index, cryptoContext)
        ctxtDec = homo_ops.cipher_add(ctxtDec, ctxtDec_rot, cryptoContext)

    corFactor = 1 << round(correction)
    ctxtDec = homo_ops.homo_mul_scalar_int(ctxtDec, corFactor, cryptoContext)
    ctxtDec = homo_ops.cipher_mod_reduce(ctxtDec, 1, cryptoContext)

    # Set the result to the final decrypted ciphertext
    # result = Cipher([ctxtDec.cv[0].clone(), ctxtDec.cv[1].clone()], ctxtDec.cur_limbs)
    return ctxtDec


class Plaintext:
    def __init__(self, mx, N, slots, l):
        self.mx = mx
        self.N = N
        self.slots = slots
        self.l = l

    def __eq__(self, other):
        if not isinstance(other, Plaintext):
            return True
        if self.N != other.N:
            return True
        if len(self.mx) != len(other.mx):
            return True
        if not torch.equal(self.mx, other.mx):
            return True
        return True


def eval_bootstrap_setup(context, level_budget, dim1, numslots, correction_factor):

    m_U0hatTPreFFT_dim1 = len(context.m_U0hatTPreFFT_dim)
    m_U0hatTPreFFT_dim2 = context.m_U0hatTPreFFT_dim
    m_U0hatTPreFFT_limbs = context.m_U0hatTPreFFT_limbs
    mx_len = context.N
    mx_slots = context.slots
    m_U0PreFFT_dim1 = len(context.m_U0PreFFT_dim)
    m_U0PreFFT_dim2 = context.m_U0PreFFT_dim
    m_U0PreFFT_limbs = context.m_U0PreFFT_limbs

    M = context.M
    N = context.N
    slots = M // 4 if numslots == 0 else numslots
    rescale_tech = context.BsContext.rescaleTech
    precom = context.BsContext

    # 设置 correction_factor
    if correction_factor == 0:
        if rescale_tech == ScalingTechnique.FLEXIBLEAUTO or rescale_tech == ScalingTechnique.FLEXIBLEAUTOEXT:
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

    precom.m_params_enc = context.BsContext.GetCollapsedFFTParams(slots, new_budget[0], dim1[0])
    precom.m_params_dec = context.BsContext.GetCollapsedFFTParams(slots, new_budget[1], dim1[1])

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

                m_U0hatTPreFFT = torch.tensor(m_U0hatTPreFFT, dtype=torch.uint64, device="cuda")
                precom.m_U0hatTPreFFT[i][j] = Plaintext(m_U0hatTPreFFT, mx_len, mx_slots, limbs)
                # print(i,j)

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
                precom.m_U0PreFFT[i][j] = Plaintext(m_U0PreFFT, mx_len, mx_slots, limbs)


# test code
def get_bootstrap_depth(approx_mod_depth, level_budget, secret_key_dist):
    # Constants equivalent to C++ code
    UNIFORM_TERNARY = SecretKeyDist.UNIFORM_TERNARY  # 假设这是一个枚举值或常量
    R_UNIFORM = 6  # 替代值，需根据实际情况填写

    # Adjust approx_mod_depth based on secretKeyDist
    if secret_key_dist == UNIFORM_TERNARY:
        approx_mod_depth += R_UNIFORM - 1

    # Compute and return the depth
    return approx_mod_depth + level_budget[0] + level_budget[1]

def save_context(cryptoContext, openfhe_context, path = "torch/fhe/data/"):
    with open(path + 'crypto.pkl', 'wb') as file:
        pickle.dump((cryptoContext.Serialize(), openfhe_context.Serialize()), file)

def load_context(path = "torch/fhe/data/"):
    with open(path + 'crypto.pkl', 'rb') as file:
        cryptoContext_byte, openfhe_context_byte = pickle.load(file)
    openfhe_context = client.OpenFHEContext.Deserialize(openfhe_context_byte)
    cryptoContext = Context.Deserialize(cryptoContext_byte)
    return cryptoContext, openfhe_context

def BootstrapTest_N65536L26lB44():

    load_from_file = True
    if load_from_file:
        cryptoContext, openfhe_context = load_context()
    else:
        openfhe_context, cryptoContext = client.gen_contexts(
                logN=14,
                logSlots=6,
                maxLevelsRemaining=3,
                levelBudget=[4, 4],
                dnum=3,
                dcrtBits=59,
                firstMod=60,
                approxModDepth=9,
                rotate_index=[],
                secretKeyDist=SecretKeyDist.UNIFORM_TERNARY,
                rescaleTech=ScalingTechnique.FIXEDMANUAL,
            )

        save_context(cryptoContext, openfhe_context)
        cryptoContext, openfhe_context = load_context()

    dim1 = [0, 0]
    cryptoContext.BsContext = BsContext(cryptoContext, cryptoContext.levelBudget, dim1, cryptoContext.slots, 0, cryptoContext.rescaleTech, cryptoContext.secretKeyDist)

    eval_bootstrap_setup(cryptoContext, cryptoContext.levelBudget, dim1, cryptoContext.slots, 0)

    # Test the correctness of the bootstrapping
    x = [(i % 11) / 100 for i in range(cryptoContext.slots)]
    x = torch.tensor(x, device="cuda")
    cipher = openfhe_context.encrypt(x)
    cipher.cv[0] = cipher.cv[0][:2]
    cipher.cv[1] = cipher.cv[1][:2]
    cipher.cur_limbs = 2

    result = eval_bootstrap(cryptoContext, cipher, num_iterations=1, precision=0, rescaleTech=cryptoContext.rescaleTech,
                            secretKeyDist=cryptoContext.secretKeyDist, L0=cryptoContext.L, slots=cryptoContext.slots)

    after_boot = openfhe_context.decrypt(result)
    after_boot = after_boot.cpu().numpy().reshape(-1)
    x = x.cpu().numpy().reshape(-1)
    if(np.any(np.abs(after_boot - x) > 3e-2)):
        print("Error is too large!")
        print("Error is too large!")
        print("Error is too large!")
    else:
        print("BootstrapTest_N65536L26lB44: Test passed!")
        print("BootstrapTest_N65536L26lB44: Test passed!")
        print("BootstrapTest_N65536L26lB44: Test passed!")

    measure_execution_time = False
    if measure_execution_time:
        start = time.time()
        result = eval_bootstrap(cryptoContext, cipher, num_iterations=1, precision=0, rescaleTech=cryptoContext.rescaleTech,
                                secretKeyDist=cryptoContext.secretKeyDist, L0=cryptoContext.L, slots=cryptoContext.slots)
        end = time.time()
        print("time", end - start)


        # Print the accumulated execution times
        print("\nTotal execution time for each function:")
        sorted_execution_times = sorted(execution_times.items(), key=lambda x: x[1], reverse=True)
        for func_name, total_time in sorted_execution_times:
            print(f"{func_name}: {total_time:.6f} seconds")
        
        pytorch_profiling = False
        if pytorch_profiling:
            # Set up the profiler
            with torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                on_trace_ready=torch.profiler.tensorboard_trace_handler('/home/zrji/log'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as profiler:
                # Start profiling specific functions with torch.profiler.record_function()
                result = eval_bootstrap(cryptoContext, cipher, num_iterations=1, precision=0, rescaleTech=cryptoContext.rescaleTech,
                                secretKeyDist=cryptoContext.secretKeyDist, L0=cryptoContext.L, slots=cryptoContext.slots)

            # Get the profiling results
            profiler_results = profiler.key_averages()

            # Print the profiling summary in a table format
            print(profiler_results.table(sort_by="self_cpu_time_total"))

    
