import math
import torch
import numpy as np
from .Ciphertext import Cipher
from .context import *
from . import functional as F
from . import homo_ops
from .data import m_U0PreFFT_mx
from .data import m_U0hatTPreFFT_mx

Tensor = torch.Tensor
NORMAL_CIPHER_SIZE = 2
BASE_NUM_LEVELS_TO_DROP = 1
ENCRYPTION = 0
MULTIPLICATION = 1
CONJUGATION = 2
K_UNIFORM = 512
R_UNIFORM = 6  # number of double-angle iterations in CKKS bootstrapping. Must be static because it is used in a static function.
R_SPARSE = 3  # number of double-angle iterations in CKKS bootstrapping. Must be static because it is used in a static function.
m_correctionFactor = 0  # correction factor, which we scale the message by to improve precision

coefficientsSparse = np.array([
    0, -0.0190665676962401, 0, -0.0181773905007824, 0, -0.0162862756167401, 0, -0.0131970301188482,
    0, -0.00869599648960049, 0, -0.00266512292674043, 0, 0.00475378458365385, 0, 0.0129619218183744,
    0, 0.0207345065018299, 0, 0.0261987740118010, 0, 0.0271237206149663, 0, 0.0216632442529301,
    0, 0.00952467756531695, 0, -0.00682586258643841, 0, -0.0217665193289893, 0, -0.0279850481505861,
    0, -0.0202671538394630, 0, -0.000311697041869291, 0, 0.0210206341691402, 0, 0.0282597848811002,
    0, 0.0130902946902468, 0, -0.0144903750619968, 0, -0.0292119597624053, 0, -0.0133436971840822,
    0, 0.0187762764821447, 0, 0.0284541504148807, 0, -0.000489726742355156, 0, -0.0298222811587479,
    0, -0.0127584877864399, 0, 0.0267192319192248, 0, 0.0186624682104780, 0, -0.0261495713329483,
    0, -0.0179030470013594, 0, 0.0303046477803535, 0, 0.00859965792435869, 0, -0.0352157135816712,
    0, 0.0127788627989003, 0, 0.0264211888837408, 0, -0.0374200640582086, 0, 0.0132393631154040,
    0, 0.0219435428661135, 0, -0.0444788687151216, 0, 0.0477866972698431, 0, -0.0383304915060382,
    0, 0.0252513113739573, 0, -0.0142806559093283, 0, 0.00711359650506429, 0, -0.00317433716746386,
    0, 0.00128436605459822, 0, -0.000475515283653384, 0, 0.000162257517416398, 0, -0.0000513272589524132,
    0, 0.0000151253840421986, 0, -4.16938339926456e-6, 0, 1.07891901728700e-6, 0, -2.62909460240295e-7,
    0, 6.04943494968095e-8, 0, -1.31757718513370e-8, 0, 2.72234854083432e-9, 0, -5.34663845707394e-10,
    0, 9.99938555825121e-11, 0, -1.78377633651571e-11, 0, 3.03978611829284e-12, 0, -4.95680040223255e-13,
    0, 7.73718537798400e-14, 0, -1.14402314781930e-14, 0, 1.69000615970718e-15, 0
], dtype=np.float64)

coefficientsUniform = np.array([
    0.15421426400235561, -0.0037671538417132409, 0.16032011744533031, -0.0034539657223742453,
    0.17711481926851286, -0.0027619720033372291, 0.19949802549604084, -0.0015928034845171929,
    0.21756948616367638, 0.00010729951647566607, 0.21600427371240055, 0.0022171399198851363,
    0.17647500259573556, 0.0042856217194480991, 0.086174491919472254, 0.0054640252312780444,
    -0.046667988130649173, 0.0047346914623733714, -0.17712686172280406, 0.0016205080004247200,
    -0.22703114241338604, -0.0028145845916205865, -0.13123089730288540, -0.0056345646688793190,
    0.078818395388692147, -0.0037868875028868542, 0.23226434602675575, 0.0021116338645426574,
    0.13985510526186795, 0.0059365649669377071, -0.13918475289368595, 0.0018580676740836374,
    -0.23254376365752788, -0.0054103844866927788, 0.056840618403875359, -0.0035227192748552472,
    0.25667909012207590, 0.0055029673963982112, -0.073334392714092062, 0.0027810273357488265,
    -0.24912792167850559, -0.0069524866497120566, 0.21288810409948347, 0.0017810057298691725,
    0.088760951809475269, 0.0055957188940032095, -0.31937177676259115, -0.0087539416335935556,
    0.34748800245527145, 0.0075378299617709235, -0.25116537379803394, -0.0047285674679876204,
    0.13970502851683486, 0.0023672533925155220, -0.063649401080083698, -0.00098993213448982727,
    0.024597838934816905, 0.00035553235917057483, -0.0082485030307578155, -0.00011176184313622549,
    0.0024390574829093264, 0.000031180384864488629, -0.00064373524734389861, -7.8036008952377965e-6,
    0.00015310015145922058, 1.7670804180220134e-6, -0.000033066844379476900, -3.6460909134279425e-7,
    6.5276969021754105e-6, 6.8957843666189918e-8, -1.1842811187642386e-6, -1.2015133285307312e-8,
    1.9839339947648331e-7, 1.9372045971100854e-9, -3.0815418032523593e-8, -2.9013806338735810e-10,
    4.4540904298173700e-9, 4.0505136697916078e-11, -6.0104912807134771e-10, -5.2873323696828491e-12,
    7.5943206779351725e-11, 6.4679566322060472e-13, -9.0081200925539902e-12, -7.4396949275292252e-14,
    1.0057423059167244e-12, 8.1701187638005194e-15, -1.0611736208855373e-13, -8.9597492970451533e-16,
    1.1421575296031385e-14
], dtype=np.float64)


def degree(coefficients, poly_degree):
    deg = 1
    for i in range(poly_degree - 1, 0, -1):
        if coefficients[i] == 0:
            deg += 1
        else:
            break
    return poly_degree - deg


PREC = math.pow(2, -20)


def is_not_equal_one(val):
    return val < 1 - PREC or val > 1 + PREC


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


def eval_linear_wsum_mutable(ciphertexts, ciphertexts_num, constants, cryptoContext: Context):
    minLevel = ciphertexts[0].cur_limbs
    minIdx = 0
    for i in range(1, ciphertexts_num):
        if ciphertexts[i].cur_limbs < minLevel:
            minLevel = ciphertexts[i].cur_limbs
            minIdx = i
    for i in range(minIdx):
        if ciphertexts[i].cur_limbs < minLevel:
            mod_down_to_and_equal(ciphertexts[i], minLevel, cryptoContext)
    for i in range(minIdx + 1, ciphertexts_num):
        if ciphertexts[i].cur_limbs < minLevel:
            mod_down_to_and_equal(ciphertexts[i], minLevel, cryptoContext)
    wsum = eval_mult_in_place(ciphertexts[0], constants[0],
                              cryptoContext)
    for i in range(1, ciphertexts_num):
        tmp = eval_mult_in_place(ciphertexts[i], constants[i],
                                 cryptoContext)
        wsum = homo_ops.cipher_add(wsum, tmp, cryptoContext)
    wsum = homo_ops.cipher_mod_reduce(wsum, 1, cryptoContext)
    return wsum


def mod_down_and_equal(a, l, dl, logN):
    ra = torch.tensor([0] * ((l - dl) << logN), dtype=torch.uint64,
                      device="cuda").reshape((l - dl), -1)  # Create a new array of the required size
    ra[:(l - dl)][:] = a[:(l - dl)][:]  # Copy values from a to ra
    return ra  # Return the new array


def mod_down_by_and_equal(cipher: Cipher, dl, cryptoContext: Context):
    cipher.cv[0] = mod_down_and_equal(cipher.cv[0], cipher.cur_limbs, dl, cryptoContext.logN)
    cipher.cv[1] = mod_down_and_equal(cipher.cv[1], cipher.cur_limbs, dl, cryptoContext.logN)
    cipher.cur_limbs -= dl
    return cipher


def mod_down_to_and_equal(cipher: Cipher, l, cryptoContext: Context):
    dl = cipher.cur_limbs - l
    mod_down_by_and_equal(cipher, dl, cryptoContext)
    return cipher


def check_and_adjust_level(ct1: Cipher, ct2: Cipher, cryptoContext: Context):
    rct1 = Cipher([ct1.cv[0].clone(), ct1.cv[1].clone()], ct1.cur_limbs)
    rct2 = Cipher([ct2.cv[0].clone(), ct2.cv[1].clone()], ct2.cur_limbs)

    if rct1.cur_limbs > rct2.cur_limbs:
        mod_down_to_and_equal(rct1, rct2.cur_limbs, cryptoContext)
    elif rct1.cur_limbs < rct2.cur_limbs:
        mod_down_to_and_equal(rct2, rct1.cur_limbs, cryptoContext)
    return rct1, rct2


def my_mult_and_equal(cipher0, cipher1, cryptoContext):
    axbx1 = F.cv_add(cipher0.cv[1], cipher0.cv[0], cryptoContext.moduliQ_cuda, cipher0.cur_limbs)
    axbx2 = F.cv_add(cipher1.cv[1], cipher1.cv[0], cryptoContext.moduliQ_cuda, cipher0.cur_limbs)
    axbx1 = F.cv_mul(axbx1, axbx2, cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda, cipher0.cur_limbs)
    bxbx = F.cv_mul(cipher0.cv[0], cipher1.cv[0], cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda,
                    cipher0.cur_limbs)
    axax = F.cv_mul(cipher0.cv[1], cipher1.cv[1], cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda,
                    cipher0.cur_limbs)
    axbx1 = F.cv_sub(axbx1, axax, cryptoContext.moduliQ_cuda, cipher0.cur_limbs)
    axbx1 = F.cv_sub(axbx1, bxbx, cryptoContext.moduliQ_cuda, cipher0.cur_limbs)

    curr_limbs = cipher0.cur_limbs
    beta = math.ceil((curr_limbs * 1.0 / cryptoContext.K))
    swk_ax = []
    swk_bx = []
    for i in range(beta):
        swk_auto_index = cryptoContext.key_map[str(MULTIPLICATION * cryptoContext.dnum + i)]
        swk_ax.append(swk_auto_index[1])
        swk_bx.append(swk_auto_index[0])
    swk_bx = torch.cat(swk_bx, dim=0).reshape(beta, -1, cryptoContext.N)
    swk_ax = torch.cat(swk_ax, dim=0).reshape(beta, -1, cryptoContext.N)
    res = F.cv_keyswitch(axax, curr_limbs, swk_bx, swk_ax, cryptoContext)

    sumaxmult = F.cv_add(res[1], axbx1, cryptoContext.moduliQ_cuda, curr_limbs)
    sumbxmult = F.cv_add(res[0], bxbx, cryptoContext.moduliQ_cuda, curr_limbs)
    return Cipher([sumbxmult, sumaxmult], curr_limbs)


def inner_eval_chebyshev_ps(x: Cipher, coefficients, coefficients_len,
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
    cu = Cipher([T[0].cv[0].clone(), T[0].cv[1].clone()], T[0].cur_limbs)
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
        cu = mod_down_to_and_equal(cu, T2[m - 1].cur_limbs, cryptoContext)
        flag_c = True

    # Evaluate q and s2 at u
    if degree(divqr_q, divqr_q_len) > k:
        qu = inner_eval_chebyshev_ps(x, divqr_q, divqr_q_len, k, m - 1, T, T2, cryptoContext)
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
    deg_s2 = degree(s2, s2_len)
    if deg_s2 > k:
        su = inner_eval_chebyshev_ps(x, s2, s2_len, k, m - 1, T, T2, cryptoContext)
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
        su = mod_down_by_and_equal(su, 1, cryptoContext)

    if flag_c:
        T2[m - 1], cu = check_and_adjust_level(T2[m - 1], cu, cryptoContext)
        result = homo_ops.cipher_add(T2[m - 1], cu, cryptoContext)
    else:
        result = homo_ops.homo_add_scalar_double(T2[m - 1], divcs_q[0] / 2, cryptoContext)

    result, qu = check_and_adjust_level(result, qu, cryptoContext)
    result = my_mult_and_equal(result, qu, cryptoContext)
    result = homo_ops.cipher_mod_reduce(result, 1, cryptoContext)
    result, su = check_and_adjust_level(result, su, cryptoContext)
    result = homo_ops.cipher_add(result, su, cryptoContext)

    return result


def eval_chebyshev_series_ps(x: Cipher, coefficients, a, b, coefficients_len, cryptoContext: Context):
    # n = degree(coefficients, coefficients_len)
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

    T = [Cipher([x.cv[0].clone(), x.cv[1].clone()], x.cur_limbs) for _ in range(k)]
    T[0] = y

    for i in range(2, k + 1, 1):
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
        T[i - 1] = mod_down_by_and_equal(T[i - 1], level_diff, cryptoContext)

    T2 = [Cipher([T[0].cv[0].clone(), T[0].cv[1].clone()], T[0].cur_limbs) for _ in range(m)]
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
        qu = inner_eval_chebyshev_ps(x, divqr_q, divqr_q_len, k, m - 1, T, T2, cryptoContext)
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
        su = inner_eval_chebyshev_ps(x, s2, s2_len, k, m - 1, T, T2, cryptoContext)
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
    result = my_mult_and_equal(result, qu, cryptoContext)
    result = homo_ops.cipher_mod_reduce(result, 1, cryptoContext)
    result, su = check_and_adjust_level(result, su, cryptoContext)
    result = homo_ops.homo_add(result, su, cryptoContext)
    result, T2km1 = check_and_adjust_level(result, T2km1, cryptoContext)
    result = homo_ops.homo_sub(result, T2km1, cryptoContext)

    return result


def reduce_rotation(index, slots):
    islots = int(slots)
    index = int(index)

    # if slots is a power of 2
    if (int(slots) & int(slots - 1)) == 0:
        n = int(math.log2(slots))
        if index >= 0:
            return index - ((index >> n) << n)
        return index + islots + ((abs(index) >> n) << n)

    return (islots + index % islots) % islots


def eval_fast_rotation_precompute(input, curr_limbs, cryptoContext):
    res = F.cv_modup(input, curr_limbs, cryptoContext)
    return res.clone()


def find_automorphism_index_2n_complex(i, m):
    if i == 0:
        return 1

    # Conjugation automorphism
    if i == m - 1:
        return i

    # Generator
    if i < 0:
        g0 = inv_mod(5, m)
        g0 = (g0 * 5) % m
    else:
        g0 = 5

    i_unsigned = abs(i)
    g = g0

    for j in range(1, int(i_unsigned)):
        g = (g * g0) % m

    return g


def inv_mod(a, m):
    # Extended Euclidean algorithm for modular inverse
    m0, x0, x1 = m, 0, 1
    if m == 1:
        return 0
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    if x1 < 0:
        x1 += m0
    return x1


def eval_fast_key_switch_core_ext(d2Tilde, type_, key_map, expand_length, beta, curr_limbs, cryptoContext):
    swk_ax = []
    swk_bx = []
    for i in range(beta):
        swk_auto_index = cryptoContext.left_rot_key_map[str(type_ * cryptoContext.dnum + i)]
        swk_ax.append(swk_auto_index[1])
        swk_bx.append(swk_auto_index[0])
    swk_bx = torch.cat(swk_bx, dim=0).reshape(beta, -1, cryptoContext.N)
    swk_ax = torch.cat(swk_ax, dim=0).reshape(beta, -1, cryptoContext.N)

    res = F.cv_innerproduct(
        d2Tilde.reshape(-1),
        curr_limbs=curr_limbs,
        context_cuda=cryptoContext,
        swk_bx=swk_bx,
        swk_ax=swk_ax
    )
    return res[1], res[0]


def set_zero(array, start, length):
    array[start:start + length] = 0


def reverse_bits(num, num_bits):
    rev = 0
    for i in range(num_bits):
        rev = (rev << 1) | (num & 1)
        num >>= 1
    return rev


def automorphism_transform(a, l, N, i, precomp_vec, cryptoContext):
    ra = F.cv_automorphism_transform(cryptoContext, a, int(l), int(N), int(i), precomp_vec)
    return ra


def eval_fast_rotation_ext_add_first_true(bx, digits, curr_limbs, index, cryptoContext):
    N = cryptoContext.N
    M = N << 1
    alpha = cryptoContext.K
    logN = cryptoContext.logN
    K = cryptoContext.K
    beta = int(np.ceil(curr_limbs / alpha))  # Calculate beta as per the original C++ code

    # Find the automorphism index that corresponds to rotation index.
    auto_index = find_automorphism_index_2n_complex(index, M)

    expand_limbs = curr_limbs + K
    expand_length = expand_limbs << logN

    # Inner Product
    sumaxmult, sumbxmult = eval_fast_key_switch_core_ext(digits, auto_index, cryptoContext.left_rot_key_map,
                                                         expand_length, beta, curr_limbs, cryptoContext)

    cMult = F.cv_mul_scalar(bx, cryptoContext.PModq_cuda, cryptoContext.moduliQ_cuda,
                            cryptoContext.q_mu_cuda, curr_limbs)

    sumbxmult = F.cv_add(sumbxmult, cMult, cryptoContext.moduliQ_cuda, curr_limbs, inplace=True)

    vec_len = N
    vec = np.zeros(vec_len, dtype=np.int32)
    vec_tensor = cryptoContext.precompute_auto_map(N, auto_index, vec)

    cv1 = automorphism_transform(sumaxmult, expand_limbs, N, auto_index, vec_tensor, cryptoContext)
    cv0 = automorphism_transform(sumbxmult, expand_limbs, N, auto_index, vec_tensor, cryptoContext)
    return Cipher([cv0, cv1], curr_limbs)


def eval_fast_rotation_ext_add_first_false(digits, curr_limbs, index, cryptoContext):
    N = cryptoContext.N
    M = N << 1
    alpha = cryptoContext.K
    logN = cryptoContext.logN
    K = cryptoContext.K
    beta = int(np.ceil(curr_limbs / alpha))  # Calculate the beta value

    # Find the automorphism index that corresponds to the rotation index.
    auto_index = find_automorphism_index_2n_complex(index, M)

    expand_limbs = curr_limbs + K
    expand_length = expand_limbs << logN

    # Inner Product
    sum_ax_mult, sum_bx_mult = eval_fast_key_switch_core_ext(digits, auto_index,
                                                             cryptoContext.left_rot_key_map, expand_length, beta,
                                                             curr_limbs, cryptoContext)

    vec_len = N
    vec = np.zeros(vec_len, dtype=np.int32)
    vec_tensor = cryptoContext.precompute_auto_map(N, auto_index, vec)

    cv1 = automorphism_transform(sum_ax_mult, expand_limbs, N, auto_index, vec_tensor, cryptoContext)
    cv0 = automorphism_transform(sum_bx_mult, expand_limbs, N, auto_index, vec_tensor, cryptoContext)
    return Cipher([cv0, cv1], curr_limbs)


def key_switch_ext(result, cipher, cipher_size, add_first, cryptoContext):
    """
    Performs key switching on the given ciphertext.
    """
    # Check for supported cipher size
    if cipher_size > 2:
        print(f"error, KeySwitchExt currently does not support {cipher_size}-dim ciphertext")
        return

    curr_limbs = cipher.cur_limbs
    N = cryptoContext.N
    logN = cryptoContext.logN
    K = cryptoContext.K

    # Initialize result arrays to zero
    result_bx = torch.tensor([0] * ((curr_limbs + K) << logN), dtype=torch.uint64, device="cuda").reshape(-1,
                                                                                                          cryptoContext.N)
    result_ax = torch.tensor([0] * ((curr_limbs + K) << logN), dtype=torch.uint64, device="cuda").reshape(-1,
                                                                                                          cryptoContext.N)
    result = Cipher([result_bx, result_ax], curr_limbs)

    if add_first:
        result.cv[0] = F.cv_mul_scalar(cipher.cv[0], cryptoContext.PModq_cuda, cryptoContext.moduliQ_cuda,
                                       cryptoContext.q_mu_cuda,
                                       curr_limbs)

    else:
        # If not adding the first, we ensure bx is zero-initialized
        result.cv[0][0:curr_limbs << logN] = [0] * (curr_limbs << logN)
    result.cv[1] = F.cv_mul_scalar(cipher.cv[1], cryptoContext.PModq_cuda, cryptoContext.moduliQ_cuda,
                                   cryptoContext.q_mu_cuda,
                                   curr_limbs)
    return result


def eval_mult_ext(cipher, pt, cryptoContext):
    curr_limbs = cipher.cur_limbs

    # Perform the multiplication on ax and bx components
    moduli = torch.tensor(
        np.concatenate((cryptoContext.moduliQ[0:curr_limbs], cryptoContext.moduliP[0:cryptoContext.K])),
        dtype=torch.uint64, device="cuda")
    mu = torch.tensor(
        np.concatenate((cryptoContext.q_mu[0:curr_limbs], cryptoContext.p_mu[:cryptoContext.K])), dtype=torch.uint64,
        device="cuda")
    cv1 = F.cv_mul(cipher.cv[1], pt.mx.reshape(-1, cryptoContext.N), moduli, mu, cipher.cv[0].shape[0])
    cv0 = F.cv_mul(cipher.cv[0], pt.mx.reshape(-1, cryptoContext.N), moduli, mu, cipher.cv[0].shape[0])
    return Cipher([cv0, cv1], curr_limbs)


def eval_add_ext(cipher0, cipher1, cryptoContext):
    assert cipher0.cur_limbs == cipher1.cur_limbs
    curr_limbs = min(cipher0.cv[0].shape[0], cipher1.cv[0].shape[0])

    moduli = torch.tensor(
        np.concatenate((cryptoContext.moduliQ[0:cipher0.cur_limbs], cryptoContext.moduliP[0:cryptoContext.K])),
        dtype=torch.uint64, device="cuda")
    cv = [
        F.cv_add(cv0, cv1, moduli, curr_limbs, inplace=True)
        for cv0, cv1 in zip(cipher0.cv, cipher1.cv)
    ]
    return Cipher(cv, cipher0.cur_limbs)


def key_switch_down_first_element(sumbxmult, curr_limbs, cryptoContext):
    res = F.cv_moddown(sumbxmult, curr_limbs, cryptoContext)
    return res


def key_switch_down(sumaxmult, sumbxmult, curr_limbs, cryptoContext):
    res_ax = F.cv_moddown(sumaxmult, curr_limbs, cryptoContext)
    res_bx = F.cv_moddown(sumbxmult, curr_limbs, cryptoContext)
    return Cipher([res_bx, res_ax], curr_limbs)


def add_and_equal(in0, in1, curr_limbs, cryptoContext):
    moduli = torch.from_numpy(
        np.concatenate((cryptoContext.moduliQ[0:curr_limbs], cryptoContext.moduliP[0:cryptoContext.K]))).cuda()
    res = F.cv_add(in0, in1, moduli, in0.shape[0])
    return res


def eval_coeffs_to_slots(A, A_len, ctxt, cryptoContext):
    slots = A_len
    N = cryptoContext.N
    M = cryptoContext.M
    K = cryptoContext.K
    logN = cryptoContext.logN
    special_limbs = K

    precom = cryptoContext.BsContext
    level_budget = precom.paramsEnc.level_budget
    layers_collapse = precom.paramsEnc.layers_coll
    rem_collapse = precom.paramsEnc.layers_rem
    num_rotations = precom.paramsEnc.num_rotations
    b = precom.paramsEnc.baby_step
    g = precom.paramsEnc.giant_step
    num_rotations_rem = precom.paramsEnc.num_rotations_rem
    b_rem = precom.paramsEnc.baby_step_rem
    g_rem = precom.paramsEnc.giant_step_rem

    stop = -1
    flag_rem = 0

    if rem_collapse != 0:
        stop = 0
        flag_rem = 1

    rot_in = [[] for _ in range(level_budget)]
    for i in range(level_budget):
        if flag_rem == 1 and i == 0:
            rot_in[i] = [0] * (num_rotations_rem + 1)
        else:
            rot_in[i] = [0] * (num_rotations + 1)

    rot_out = [[] for _ in range(level_budget)]
    for i in range(level_budget):
        rot_out[i] = [0] * (b + b_rem)

    for s in range(level_budget - 1, stop, -1):
        for j in range(g):
            rot_in[s][j] = reduce_rotation(
                (j - (num_rotations + 1) // 2 + 1) * (1 << ((s - flag_rem) * layers_collapse + rem_collapse)),
                slots)

        for i in range(b):
            rot_out[s][i] = reduce_rotation((g * i) * (1 << ((s - flag_rem) * layers_collapse + rem_collapse)), M // 4)

    if flag_rem:
        for j in range(g_rem):
            rot_in[stop][j] = reduce_rotation((j - (num_rotations_rem + 1) // 2 + 1), slots)

        for i in range(b_rem):
            rot_out[stop][i] = reduce_rotation((g_rem * i), M // 4)

    result = Cipher([ctxt.cv[0].clone(), ctxt.cv[1].clone()], ctxt.cur_limbs)

    for s in range(level_budget - 1, stop, -1):
        if s != level_budget - 1:
            result = homo_ops.cipher_mod_reduce(result, 1, cryptoContext)

        curr_limbs = result.cur_limbs
        limbs_ext = curr_limbs + special_limbs
        len_ext = (limbs_ext << logN)
        len = (curr_limbs << logN)
        alpha = cryptoContext.K
        beta = (curr_limbs + alpha - 1) // alpha

        digits_len = beta * len_ext

        # modup
        digits = eval_fast_rotation_precompute(result.cv[1], result.cur_limbs, cryptoContext)
        fast_rotation_ext = [[0] for _ in range(g)]

        for j in range(g):
            if rot_in[s][j] != 0:
                fast_rotation_ext[j] = eval_fast_rotation_ext_add_first_true(
                    result.cv[0].reshape(-1, cryptoContext.N),
                    digits,
                    result.cur_limbs, rot_in[s][j],
                    cryptoContext)
            else:
                fast_rotation_ext[j] = key_switch_ext(fast_rotation_ext[j], result, 2, True, cryptoContext)

        outer_ax = [0] * len_ext
        outer_bx = [0] * len_ext
        outer = Cipher([outer_ax, outer_bx], curr_limbs)

        first = [0] * len

        for i in range(b):
            G = g * i
            inner = eval_mult_ext(fast_rotation_ext[0], A[s][G], cryptoContext)

            for j in range(1, g):
                if (G + j) != num_rotations:
                    tmp_ext = eval_mult_ext(fast_rotation_ext[j], A[s][G + j], cryptoContext)
                    inner = eval_add_ext(inner, tmp_ext, cryptoContext)

            if i == 0:
                # moddown
                first = key_switch_down_first_element(inner.cv[0], curr_limbs, cryptoContext)
                set_zero(inner.cv[0], 0, len_ext)
                outer = inner
            else:
                if rot_out[s][i] != 0:
                    inner_ks_down = key_switch_down(inner.cv[1], inner.cv[0], curr_limbs,
                                                    cryptoContext)
                    auto_index = find_automorphism_index_2n_complex(rot_out[s][i], M)
                    map_len = N
                    map = np.zeros(map_len, dtype=np.int32)
                    map_tensor = cryptoContext.precompute_auto_map(N, auto_index, map)

                    first_current = automorphism_transform(inner_ks_down.cv[0], curr_limbs, N, auto_index, map_tensor,
                                                           cryptoContext)
                    first = add_and_equal(first, first_current, curr_limbs, cryptoContext)

                    inner_digits = eval_fast_rotation_precompute(inner_ks_down.cv[1], inner_ks_down.cur_limbs,
                                                                 cryptoContext)
                    inner_ks_down_ext = eval_fast_rotation_ext_add_first_false(inner_digits,
                                                                               inner_ks_down.cur_limbs, rot_out[s][i],
                                                                               cryptoContext)
                    outer = eval_add_ext(outer, inner_ks_down_ext, cryptoContext)
                else:
                    tmp_first = key_switch_down_first_element(inner.cv[0], curr_limbs, cryptoContext)
                    first = add_and_equal(first, tmp_first, curr_limbs, cryptoContext)
                    set_zero(inner.cv[0], 0, len_ext)
                    outer = eval_add_ext(outer, inner, cryptoContext)
        result = key_switch_down(outer.cv[1], outer.cv[0], curr_limbs, cryptoContext)
        result.cv[0] = add_and_equal(result.cv[0], first, curr_limbs, cryptoContext)

    if flag_rem:
        result = homo_ops.cipher_mod_reduce(result, 1, cryptoContext)
        curr_limbs = result.cur_limbs
        limbs_ext = curr_limbs + special_limbs
        len_ext = (limbs_ext << logN)
        len = (curr_limbs << logN)
        alpha = cryptoContext.K
        beta = (curr_limbs + alpha - 1) // alpha

        digits_len = beta * len_ext
        digits = eval_fast_rotation_precompute(result.cv[1], result.cur_limbs, cryptoContext)

        fast_rotation_ext = [[0] for _ in range(g_rem)]
        for j in range(g_rem):
            if rot_in[stop][j] != 0:
                fast_rotation_ext[j] = eval_fast_rotation_ext_add_first_true(result.cv[0], digits,
                                                                             result.cur_limbs, rot_in[stop][j],
                                                                             cryptoContext)
            else:
                fast_rotation_ext[j] = key_switch_ext(fast_rotation_ext[j], result, NORMAL_CIPHER_SIZE, True,
                                                      cryptoContext)
        outer_ax = torch.tensor([0] * len_ext, dtype=torch.uint64, device="cuda").reshape(-1, cryptoContext.N)
        outer_bx = torch.tensor([0] * len_ext, dtype=torch.uint64, device="cuda").reshape(-1, cryptoContext.N)
        outer = Cipher([outer_bx, outer_ax], curr_limbs)
        first = [0] * len

        for i in range(b_rem):
            G = g_rem * i
            inner = eval_mult_ext(fast_rotation_ext[0], A[stop][G], cryptoContext)

            for j in range(1, g_rem):
                if (G + j) != num_rotations_rem:
                    tmp_ext = eval_mult_ext(fast_rotation_ext[j], A[stop][G + j], cryptoContext)
                    inner = eval_add_ext(inner, tmp_ext, cryptoContext)

            if i == 0:
                first = key_switch_down_first_element(inner.cv[0], curr_limbs, cryptoContext)
                set_zero(inner.cv[0], 0, len_ext)
                outer = inner
            else:
                if rot_out[stop][i] != 0:
                    inner_ks_down = key_switch_down(inner.cv[1],
                                                    inner.cv[0], curr_limbs, cryptoContext)

                    auto_index = find_automorphism_index_2n_complex(rot_out[stop][i], M)
                    map_len = N
                    map = np.zeros(map_len, dtype=np.int32)
                    map_tensor = cryptoContext.precompute_auto_map(N, auto_index, map)

                    # first_current = [0] * len
                    first_current = automorphism_transform(inner_ks_down.cv[0], curr_limbs, N, auto_index, map_tensor,
                                                           cryptoContext)
                    first = add_and_equal(first, first_current, curr_limbs, cryptoContext)

                    # inner_digits = [0] * digits_len
                    inner_digits = eval_fast_rotation_precompute(inner_ks_down.cv[1],
                                                                 inner_ks_down.cur_limbs, cryptoContext)

                    inner_ks_down_ext = eval_fast_rotation_ext_add_first_false(inner_digits,
                                                                               inner_ks_down.cur_limbs,
                                                                               rot_out[stop][i], cryptoContext)
                    outer = eval_add_ext(outer, inner_ks_down_ext, cryptoContext)
                else:
                    # tmp_first = [0] * len
                    tmp_first = key_switch_down_first_element(inner.cv[0], curr_limbs, cryptoContext)
                    first = add_and_equal(first, tmp_first, curr_limbs, cryptoContext)
                    set_zero(inner.cv[0], 0, len_ext)
                    outer = eval_add_ext(outer, inner, cryptoContext)

        result = key_switch_down(outer.cv[1], outer.cv[0], curr_limbs, cryptoContext)
        result.cv[0] = add_and_equal(result.cv[0], first, curr_limbs, cryptoContext)

    return result


def eval_slots_to_coeffs(A, A_len, ctxt, cryptoContext):
    slots = A_len

    N = cryptoContext.N
    M = cryptoContext.M
    K = cryptoContext.K
    logN = cryptoContext.logN
    special_limbs = K

    precom = cryptoContext.BsContext
    level_budget = precom.paramsDec.level_budget
    layers_collapse = precom.paramsDec.layers_coll
    rem_collapse = precom.paramsDec.layers_rem
    num_rotations = precom.paramsDec.num_rotations
    b = precom.paramsDec.baby_step
    g = precom.paramsDec.giant_step
    num_rotations_rem = precom.paramsDec.num_rotations_rem
    b_rem = precom.paramsDec.baby_step_rem
    g_rem = precom.paramsDec.giant_step_rem

    flag_rem = 1 if rem_collapse != 0 else 0

    rot_in = []
    rot_out = []

    for i in range(level_budget):
        if flag_rem == 1 and i == (level_budget - 1):
            rot_in.append(np.zeros(num_rotations_rem + 1))

        else:
            rot_in.append(np.zeros(num_rotations + 1))
    for i in range(level_budget):
        rot_out.append(np.zeros(b + b_rem))

    for s in range(level_budget - flag_rem):
        for j in range(g):
            rot_in[s][j] = reduce_rotation((j - ((num_rotations + 1) / 2) + 1) * (1 << (s * layers_collapse)), M // 4)

        for i in range(b):
            rot_out[s][i] = reduce_rotation((g * i) * (1 << (s * layers_collapse)), M // 4)

    if flag_rem:
        s = level_budget - flag_rem
        for j in range(g_rem):
            rot_in[s][j] = reduce_rotation((j - (num_rotations_rem + 1) // 2 + 1) * (1 << (s * layers_collapse)),
                                           M // 4)

        for i in range(b_rem):
            rot_out[s][i] = reduce_rotation((g_rem * i) * (1 << (s * layers_collapse)), M // 4)

    result = ctxt

    for s in range(level_budget - flag_rem):
        if s != 0:
            result = homo_ops.cipher_mod_reduce(result, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        curr_limbs = result.cur_limbs
        limbs_ext = curr_limbs + special_limbs
        len_ext = limbs_ext << logN
        len_ = curr_limbs << logN
        alpha = cryptoContext.K
        beta = (curr_limbs + alpha - 1) // alpha

        digits_len = beta * len_ext
        digits = eval_fast_rotation_precompute(result.cv[1], result.cur_limbs, cryptoContext)

        fast_rotation_ext = [[0] for _ in range(g)]

        for j in range(g):
            if rot_in[s][j] != 0:
                fast_rotation_ext[j] = eval_fast_rotation_ext_add_first_true(result.cv[0], digits,
                                                                             result.cur_limbs, rot_in[s][j],
                                                                             cryptoContext)
            else:
                fast_rotation_ext[j] = key_switch_ext(fast_rotation_ext[j], result, NORMAL_CIPHER_SIZE, True,
                                                      cryptoContext)

        outer_ax = [0] * len_ext
        outer_bx = [0] * len_ext
        outer = Cipher([outer_ax, outer_bx], curr_limbs)

        first = [0] * len_

        for i in range(b):
            G = g * i
            inner = eval_mult_ext(fast_rotation_ext[0], A[s][G], cryptoContext)

            for j in range(1, g):
                if (G + j) != num_rotations:
                    tmp_ext = eval_mult_ext(fast_rotation_ext[j], A[s][G + j], cryptoContext)
                    inner = eval_add_ext(inner, tmp_ext, cryptoContext)

            if i == 0:
                first = key_switch_down_first_element(inner.cv[0], curr_limbs, cryptoContext)
                set_zero(inner.cv[0], 0, len_ext)
                outer = inner
            else:
                if rot_out[s][i] != 0:
                    inner_ks_down = key_switch_down(inner.cv[1],
                                                    inner.cv[0], curr_limbs, cryptoContext)

                    auto_index = find_automorphism_index_2n_complex(rot_out[s][i], M)
                    map_ = np.zeros(N, dtype=np.int32)
                    map_tensor = cryptoContext.precompute_auto_map(N, auto_index, map_)

                    first_current = automorphism_transform(inner_ks_down.cv[0], curr_limbs, N, auto_index, map_tensor,
                                                           cryptoContext)
                    first = add_and_equal(first, first_current, curr_limbs, cryptoContext)

                    inner_digits = eval_fast_rotation_precompute(inner_ks_down.cv[1], inner_ks_down.cur_limbs,
                                                                 cryptoContext)

                    inner_ks_down_ext = eval_fast_rotation_ext_add_first_false(inner_digits,
                                                                               inner_ks_down.cur_limbs, rot_out[s][i],
                                                                               cryptoContext)
                    outer = eval_add_ext(outer, inner_ks_down_ext, cryptoContext)
                else:
                    # tmp_first = [0] * len_
                    tmp_first = key_switch_down_first_element(inner.cv[1], curr_limbs, cryptoContext)
                    first = add_and_equal(first, tmp_first, curr_limbs, cryptoContext)
                    set_zero(inner.cv[0], 0, len_ext)
                    outer = eval_add_ext(outer, inner, cryptoContext)

        result = key_switch_down(outer.cv[1], outer.cv[0], curr_limbs, cryptoContext)
        result.cv[0] = add_and_equal(result.cv[0], first, curr_limbs, cryptoContext)

    if flag_rem:
        result = homo_ops.cipher_mod_reduce(result, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
        curr_limbs = result.cur_limbs
        limbs_ext = curr_limbs + special_limbs
        len_ext = limbs_ext << logN
        len_ = curr_limbs << logN
        alpha = cryptoContext.K
        beta = (curr_limbs + alpha - 1) // alpha

        digits_len = beta * len_ext
        digits = eval_fast_rotation_precompute(result.cv[1], result.cur_limbs, cryptoContext)

        fast_rotation_ext = [
            [0] for _ in
            range(g_rem)]

        s = level_budget - flag_rem
        for j in range(g_rem):

            if rot_in[s][j] != 0:
                fast_rotation_ext[j] = eval_fast_rotation_ext_add_first_true(result.cv[0], digits,
                                                                             result.cur_limbs, rot_in[s][j],
                                                                             cryptoContext)
            else:
                fast_rotation_ext[j] = key_switch_ext(fast_rotation_ext[j], result, NORMAL_CIPHER_SIZE, True,
                                                      cryptoContext)

        outer = Cipher([torch.tensor([0] * len_ext, dtype=torch.uint64, device="cuda").reshape(-1, cryptoContext.N),
                        torch.tensor([0] * len_ext, dtype=torch.uint64, device="cuda").reshape(-1, cryptoContext.N)],
                       curr_limbs)
        first = [0] * len_

        for i in range(b_rem):
            G_rem = g_rem * i
            inner = eval_mult_ext(fast_rotation_ext[0], A[s][G_rem], cryptoContext)

            for j in range(1, g_rem):
                if (G_rem + j) != num_rotations_rem:
                    tmp_ext = eval_mult_ext(fast_rotation_ext[j], A[s][G_rem + j], cryptoContext)
                    inner = eval_add_ext(inner, tmp_ext, cryptoContext)

            if i == 0:
                first = key_switch_down_first_element(inner.cv[0], curr_limbs, cryptoContext)
                set_zero(inner.cv[0], 0, len_ext)
                outer = inner
            else:
                if rot_out[s][i] != 0:
                    inner_ks_down = key_switch_down(inner.cv[1], inner.cv[0], curr_limbs,
                                                    cryptoContext)

                    auto_index = find_automorphism_index_2n_complex(rot_out[s][i], M)
                    map_ = np.zeros(N, dtype=np.int32)
                    map_tensor = cryptoContext.precompute_auto_map(N, auto_index, map_)

                    # first_current = [0] * len_
                    first_current = automorphism_transform(inner_ks_down.cv[0], curr_limbs, N, auto_index, map_tensor,
                                                           cryptoContext)
                    first = add_and_equal(first, first_current, curr_limbs, cryptoContext)

                    inner_digits = eval_fast_rotation_precompute(inner_ks_down.cv[1], inner_ks_down.cur_limbs,
                                                                 cryptoContext)

                    inner_ks_down_ext = eval_fast_rotation_ext_add_first_false(inner_digits,
                                                                               inner_ks_down.cur_limbs, rot_out[s][i],
                                                                               cryptoContext)
                    outer = eval_add_ext(outer, inner_ks_down_ext, cryptoContext)
                else:
                    # tmp_first = [0] * len_
                    tmp_first = key_switch_down_first_element(inner.cv[0], curr_limbs, cryptoContext)
                    first = add_and_equal(first, tmp_first, curr_limbs, cryptoContext)
                    set_zero(inner.cv[0], 0, len_ext)
                    outer = eval_add_ext(outer, inner, cryptoContext)

        result = key_switch_down(outer.cv[1], outer.cv[0], curr_limbs, cryptoContext)
        result.cv[0] = add_and_equal(result.cv[0], first, curr_limbs, cryptoContext)

    # 返回结果
    return result


def get_element_for_eval_mult(factors, ciphertext, constant, cryptoContext):
    num_towers = ciphertext.cur_limbs
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


def eval_mult_in_place(ciphertext, constant, cryptoContext):
    logN = cryptoContext.logN
    curr_limbs = ciphertext.cur_limbs
    factors = np.zeros(curr_limbs, dtype=np.uint64)

    # Generate the factors needed for multiplication
    factors = get_element_for_eval_mult(factors, ciphertext, constant, cryptoContext)
    factors = torch.tensor(factors, dtype=torch.uint64, device="cuda")
    cv = [
        F.cv_mul_scalar(
            cv0, factors, cryptoContext.moduliQ_cuda, cryptoContext.q_mu_cuda, ciphertext.cur_limbs
        )
        for cv0 in ciphertext.cv
    ]
    return Cipher(cv, ciphertext.cur_limbs)


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


def eval_linear_transform(A, A_len, ct, scheme):
    # TODO: to be implemented
    pass


def conjugate_demo(cipher, cryptoContext):
    curr_limbs = cipher.cur_limbs
    N = cryptoContext.N
    M = N << 1
    logN = cryptoContext.logN

    auto_index = 2 * N - 1  # 自动映射索引

    KS_input = cipher.cv[1]

    res_len = curr_limbs << logN

    beta = math.ceil((curr_limbs * 1.0 / cryptoContext.K))
    swk_ax = []
    swk_bx = []
    for i in range(beta):
        swk_auto_index = cryptoContext.left_rot_key_map[str(auto_index * cryptoContext.dnum + i)]
        swk_ax.append(swk_auto_index[1])
        swk_bx.append(swk_auto_index[0])
    swk_bx = torch.cat(swk_bx, dim=0).reshape(beta, -1, cryptoContext.N)
    swk_ax = torch.cat(swk_ax, dim=0).reshape(beta, -1, cryptoContext.N)
    # swk = cryptoContext.left_rot_key_map[str(autoIndex * cryptoContext.dnum)]
    res = F.cv_keyswitch(KS_input, curr_limbs, swk_bx, swk_ax, cryptoContext)
    res_cipher = Cipher(res, curr_limbs)

    bxrot = homo_ops.cipher_add(res_cipher, cipher, cryptoContext)

    vec_len = N
    vec = np.zeros(vec_len, dtype=np.int32)
    vec_tensor = cryptoContext.precompute_auto_map(N, auto_index, vec)  # 自动映射预计算

    cipher.cv[1] = automorphism_transform(res_cipher.cv[1], curr_limbs, N, auto_index, vec_tensor, cryptoContext)
    cipher.cv[0] = automorphism_transform(bxrot.cv[0], curr_limbs, N, auto_index, vec_tensor, cryptoContext)
    return cipher


def fast_rotate_demo(cipher, index, cryptoContext):
    curr_limbs = cipher.cur_limbs
    N = cryptoContext.N
    M = N << 1
    logN = cryptoContext.logN

    auto_index = find_automorphism_index_2n_complex(index, M)  # Equivalent to FindAutomorphismIndex2nComplex

    # KeySwitchCore operation: rotating cipher.ax to bx
    KS_input = cipher.cv[1]

    res_len = curr_limbs << logN

    # KeySwitchCore operation with scheme's leftRotKeyMap
    beta = math.ceil((curr_limbs * 1.0 / cryptoContext.K))
    swk_ax = []
    swk_bx = []
    for i in range(beta):
        swk_auto_index = cryptoContext.left_rot_key_map[str(auto_index * cryptoContext.dnum + i)]
        swk_ax.append(swk_auto_index[1])
        swk_bx.append(swk_auto_index[0])
    swk_bx = torch.cat(swk_bx, dim=0).reshape(beta, -1, cryptoContext.N)
    swk_ax = torch.cat(swk_ax, dim=0).reshape(beta, -1, cryptoContext.N)
    res = F.cv_keyswitch(KS_input, curr_limbs, swk_bx, swk_ax, cryptoContext)

    res_cipher = Cipher(res, curr_limbs)

    bxrot = homo_ops.cipher_add(cipher, res_cipher, cryptoContext)

    # Precompute the automorphism map
    vec_len = N
    vec = np.zeros(vec_len, dtype=np.int32)
    vec_tensor = cryptoContext.precompute_auto_map(N, auto_index, vec)  # Equivalent to PrecomputeAutoMap

    # Apply the AutomorphismTransform to ax and bx
    cipher.cv[1] = automorphism_transform(res_cipher.cv[1], curr_limbs, N, auto_index, vec_tensor, cryptoContext)
    cipher.cv[0] = automorphism_transform(bxrot.cv[0], curr_limbs, N, auto_index, vec_tensor, cryptoContext)

    return cipher


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


def mul_by_monomial_and_equal(a, l, monomial_deg, context):
    M = context.M  # Cyclotomic order, should be part of context
    N = context.N
    q_vec = context.q_vec  # Assumed to be an array in context
    logN = context.logN

    shift = monomial_deg % M
    if shift == 0:
        return

    # Creating the temporary array
    tmp = [[0] * N for _ in range(l)]

    if shift < N:
        for i in range(l):
            tmpi = tmp[i]
            ai = a[i * (1 << logN):(i + 1) * (1 << logN)]
            for n in range(N):
                tmpi[n] = ai[n]
    else:
        # Negate using qVec
        for i in range(l):
            tmpi = tmp[i]
            ai = a[i * (1 << logN):(i + 1) * (1 << logN)]
            for n in range(N):
                tmpi[n] = q_vec[i] - ai[n]

    shift %= N
    for i in range(l):
        tmpi = tmp[i]
        ai = a[i * (1 << logN):(i + 1) * (1 << logN)]
        for n in range(shift):
            ai[n] = q_vec[i] - tmpi[N - shift + n]
        for n in range(shift, N):
            ai[n] = tmpi[n - shift]


def mult_by_monomial_and_equal(cipher, monomial_degree, cryptoContext):
    l = cipher.cur_limbs
    qVec = cryptoContext.primes[cryptoContext.L:]
    cipher.cv[0] = F.cv_mul_by_monomial(cryptoContext, cipher.cv[0], qVec, cipher.cv[0].clone(), l, monomial_degree, l)
    cipher.cv[1] = F.cv_mul_by_monomial(cryptoContext, cipher.cv[1], qVec, cipher.cv[1].clone(), l, monomial_degree, l)
    return cipher


def switch_modulus_with_intt_ntt(input_tensor, l, cryptoContext):
    res = F.cv_switch_modulus(cryptoContext, input_tensor, l)
    return res


def eval_bootstrap(cryptoContext, ciphertext, num_iterations, precision, rescaleTech, secretKeyDist, L0, slots):
    M = cryptoContext.M
    N = cryptoContext.N
    logN = cryptoContext.logN
    cryptoContext.slots = slots
    precom = cryptoContext.BsContext
    moduliQ = cryptoContext.moduliQ
    rescaleTech = precom.rescaleTech

    if num_iterations > 1:
        # TODO: to be implemented
        pass

    if rescaleTech == ScalingTechnique.FLEXIBLEAUTOEXT:
        # TODO: to be implemented
        pass

    q = moduliQ[0]
    q_double = float(q)

    p = cryptoContext.logp  # Equivalent to dcrbits in OpenFHE
    powP = 2 ** p
    deg = round(math.log2(q_double / powP))

    correction = cryptoContext.correctionFactor - deg  # fixme: originally a uint32_t in OpenFHE
    post = 2 ** deg
    pre = 1. / post
    scalar = round(post)

    # tmp = ciphertext.copy()  # Copy ciphertext
    tmp = Cipher([ciphertext.cv[0].clone(), ciphertext.cv[1].clone()], ciphertext.cur_limbs)
    tmp = adjust_ciphertext(cryptoContext, tmp, correction)

    # raised_ax = [0] * (L0 << logN)
    # raised_bx = [0] * (L0 << logN)
    raised_cv0 = torch.tensor([0] * (L0 << logN), dtype=torch.uint64, device="cuda").reshape(-1, cryptoContext.N)
    raised_cv1 = torch.tensor([0] * (L0 << logN), dtype=torch.uint64, device="cuda").reshape(-1, cryptoContext.N)
    for i in range(tmp.cur_limbs):
        raised_cv0[i] = tmp.cv[0][i]
        raised_cv1[i] = tmp.cv[1][i]
    raised = Cipher([raised_cv0, raised_cv1], L0)
    raised.cv[0] = switch_modulus_with_intt_ntt(raised.cv[0], L0, cryptoContext)  # bx
    raised.cv[1] = switch_modulus_with_intt_ntt(raised.cv[1], L0, cryptoContext)  # ax

    print("Mod Raise done")

    # Chebyshev series coefficients for modular reduction
    if secretKeyDist == SecretKeyDist.SPARSE_TERNARY:
        coefficients = np.copy(coefficientsSparse)
        coefficients_len = len(coefficients)
        k = 1.0
    else:
        coefficients = np.copy(coefficientsUniform)
        coefficients_len = len(coefficients)
        k = K_UNIFORM

    constantEvalMult = pre * (1.0 / (k * N))

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

        print("CoeffsToSlots done")

        conj = Cipher([ctxtEnc.cv[0].clone().ctxtEnc.cv[1].clone()], ctxtEnc.cur_limbs)
        conj = conjugate_demo(conj, cryptoContext)

        ctxtEncI = homo_ops.cipher_sub(ctxtEnc, conj, cryptoContext)
        ctxtEnc = homo_ops.cipher_add(ctxtEnc, conj, cryptoContext)
        mult_by_monomial_and_equal(ctxtEncI, 3 * M // 4, cryptoContext)

        if rescaleTech == ScalingTechnique.FIXEDMANUAL:
            ctxtEnc = homo_ops.cipher_mod_reduce(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
            ctxtEncI = homo_ops.cipher_mod_reduce(ctxtEncI, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        ctxtEnc_copy = Cipher([ctxtEnc.cv[0].clone(), ctxtEnc.cv[1].clone()], ctxtEnc.cur_limbs)  # ctxtEnc.copy()
        ctxtEncI_copy = Cipher([ctxtEncI.cv[0].clone(), ctxtEncI.cv[1].clone()], ctxtEncI.cur_limbs)  # ctxtEncI.copy()
        ctxtEnc = eval_chebyshev_series_ps(ctxtEnc_copy, coefficients, -1, 1, coefficients_len, cryptoContext)
        ctxtEncI = eval_chebyshev_series_ps(ctxtEncI_copy, coefficients, -1, 1, coefficients_len,
                                            cryptoContext)

        if secretKeyDist == SecretKeyDist.UNIFORM_TERNARY:
            if rescaleTech != ScalingTechnique.FIXEDMANUAL:
                ctxtEnc = homo_ops.cipher_mod_reduce(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
                ctxtEncI = homo_ops.cipher_mod_reduce(ctxtEncI, BASE_NUM_LEVELS_TO_DROP, cryptoContext)
            ctxtEnc = apply_double_angle_iterations(ctxtEnc, cryptoContext)
            ctxtEncI = apply_double_angle_iterations(ctxtEncI, cryptoContext)

        print("Approximate Mod Reduction done")

        mult_by_monomial_and_equal(ctxtEncI, M // 4, cryptoContext)
        ctxtEnc = homo_ops.cipher_add(ctxtEnc, ctxtEncI, cryptoContext)
        ctxtEnc = homo_ops.homo_mul_scalar_int(ctxtEnc, scalar, cryptoContext)

        if rescaleTech != ScalingTechnique.FIXEDMANUAL:
            ctxtEnc = homo_ops.cipher_mod_reduce(ctxtEnc, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if isLTBootstrap:
            ctxtDec = eval_linear_transform(precom.m_U0Pre, ctxtEnc, cryptoContext)
        else:
            ctxtDec = eval_slots_to_coeffs(precom.m_U0PreFFT, ctxtEnc.slots, ctxtEnc, cryptoContext)

    else:
        j = 1
        while j < N / (2 * slots):
            temp = Cipher([raised.cv[0].clone(), raised.cv[1].clone()], raised.cur_limbs)  # raised.copy()
            temp = fast_rotate_demo(temp, j * slots, cryptoContext)
            raised = homo_ops.cipher_add(raised, temp, cryptoContext)
            j <<= 1

        raised = homo_ops.cipher_mod_reduce(raised, BASE_NUM_LEVELS_TO_DROP, cryptoContext)

        if isLTBootstrap:
            ctxtEnc = eval_linear_transform(precom.m_U0hatTPre, raised, cryptoContext)
        else:
            ctxtEnc = eval_coeffs_to_slots(precom.m_U0hatTPreFFT, cryptoContext.slots, raised, cryptoContext)

        print("CoeffsToSlots done")

        conj = Cipher([ctxtEnc.cv[0].clone(), ctxtEnc.cv[1].clone()], ctxtEnc.cur_limbs)  # ctxtEnc.copy()
        # Conjugate_KeyGen(scheme.secretKey, scheme)
        conj = conjugate_demo(conj, cryptoContext)
        ctxtEnc = homo_ops.cipher_add(ctxtEnc, conj, cryptoContext)

        if rescaleTech == ScalingTechnique.FIXEDMANUAL:
            ctxtEnc = homo_ops.cipher_mod_reduce(ctxtEnc, 1, cryptoContext)

        print("Approximate Mod Reduction done")
        ctxtEnc_copy = Cipher([ctxtEnc.cv[0].clone(), ctxtEnc.cv[1].clone()], ctxtEnc.cur_limbs)  # ctxtEnc.copy()
        ctxtEnc = eval_chebyshev_series_ps(ctxtEnc_copy, coefficients, -1, 1, coefficients_len, cryptoContext)

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
            ctxtDec_curr_limbs = ctxtEnc.cur_limbs - precom.paramsDec.level_budget + 1

            ctxtDec = eval_slots_to_coeffs(precom.m_U0PreFFT, slots, ctxtEnc, cryptoContext)

        ctxtDec_rot = Cipher([ctxtDec.cv[0].clone(), ctxtDec.cv[1].clone()], ctxtDec.cur_limbs)  # ctxtDec.copy()
        # FastRotate_KeyGen(scheme.secretKey, slots, scheme)
        ctxtDec_rot = fast_rotate_demo(ctxtDec_rot, slots, cryptoContext)
        ctxtDec = homo_ops.cipher_add(ctxtDec, ctxtDec_rot, cryptoContext)

    print("SlotsToCoeffs done")

    corFactor = 1 << round(correction)
    ctxtDec = homo_ops.homo_mul_scalar_int(ctxtDec, corFactor, cryptoContext)
    ctxtDec = homo_ops.cipher_mod_reduce(ctxtDec, 1, cryptoContext)

    # Set the result to the final decrypted ciphertext
    result = Cipher([ctxtDec.cv[0].clone(), ctxtDec.cv[1].clone()], ctxtDec.cur_limbs)
    return result


class Plaintext:
    def __init__(self, mx, N, slots, l):
        self.mx = mx
        self.N = N
        self.slots = slots
        self.l = l

    def __eq__(self, other):
        if not isinstance(other, Plaintext):
            return False
        if self.N != other.N:
            return False
        if len(self.mx) != len(other.mx):
            return False
        if not torch.equal(self.mx, other.mx):
            return False
        return True


m_U0hatTPreFFT_dim1 = 4
m_U0hatTPreFFT_dim2 = np.array([15, 3, 3, 3])
m_U0hatTPreFFT_limbs = np.array([31, 32, 33, 34])
mx_len = 65536
mx_slots = 64
m_U0PreFFT_dim1 = 4
m_U0PreFFT_dim2 = np.array([3, 3, 3, 15])
m_U0PreFFT_limbs = np.array([17, 16, 15, 14])


def eval_bootstrap_setup(context, level_budget, dim1, numslots, correction_factor):
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
        if N == 64:
            if context.L == 18 and context.K == 6:
                LTMatrix_Row = LTMatrix_Row_Q18P6
                LTMatrix_Column = LTMatrix_Column_Q18P6
                LTMatrix_mx_len = LTMatrix_mx_len_Q18P6

                m_U0hatTPre_limbs = m_U0hatTPre_limbs_Q18P6
                m_U0Pre_limbs = m_U0Pre_limbs_Q18P6

                precom.LTMatrix_Row = LTMatrix_Row_Q18P6
                precom.LTMatrix_Column = LTMatrix_Column_Q18P6
                precom.LTMatrix_mx_len = LTMatrix_mx_len_Q18P6

                precom.m_U0hatTPre_limbs = m_U0hatTPre_limbs_Q18P6
                precom.m_U0Pre_limbs = m_U0Pre_limbs_Q18P6

                m_U0hatTPre_mx = m_U0hatTPre_mx_Q18P6
                m_U0Pre_mx = m_U0Pre_mx_Q18P6
            elif context.L == 18 and context.K == 1:
                LTMatrix_Row = LTMatrix_Row_Q18P1
                LTMatrix_Column = LTMatrix_Column_Q18P1
                LTMatrix_mx_len = LTMatrix_mx_len_Q18P1

                m_U0hatTPre_limbs = m_U0hatTPre_limbs_Q18P1
                m_U0Pre_limbs = m_U0Pre_limbs_Q18P1

                precom.LTMatrix_Row = LTMatrix_Row_Q18P1
                precom.LTMatrix_Column = LTMatrix_Column_Q18P1
                precom.LTMatrix_mx_len = LTMatrix_mx_len_Q18P1

                precom.m_U0hatTPre_limbs = m_U0hatTPre_limbs_Q18P1
                precom.m_U0Pre_limbs = m_U0Pre_limbs_Q18P1

                m_U0hatTPre_mx = m_U0hatTPre_mx_Q18P1
                m_U0Pre_mx = m_U0Pre_mx_Q18P1
            else:
                raise ValueError("error! matrix for current L and K has not generated yet!\n")
        elif N == 65536:
            if context.L == 32 and context.K == 1:
                LTMatrix_Row = LTMatrix_Row_Q32P1_N65536
                LTMatrix_Column = LTMatrix_Column_Q32P1_N65536
                LTMatrix_mx_len = LTMatrix_mx_len_Q32P1_N65536

                m_U0hatTPre_limbs = m_U0hatTPre_limbs_Q32P1_N65536
                m_U0Pre_limbs = m_U0Pre_limbs_Q32P1_N65536

                precom.LTMatrix_Row = LTMatrix_Row_Q32P1_N65536
                precom.LTMatrix_Column = LTMatrix_Column_Q32P1_N65536
                precom.LTMatrix_mx_len = LTMatrix_mx_len_Q32P1_N65536

                precom.m_U0hatTPre_limbs = m_U0hatTPre_limbs_Q32P1_N65536
                precom.m_U0Pre_limbs = m_U0Pre_limbs_Q32P1_N65536

                m_U0hatTPre_mx = m_U0hatTPre_mx_Q32P1_N65536
                m_U0Pre_mx = m_U0Pre_mx_Q32P1_N65536
            else:
                raise ValueError("error! matrix for current L and K has not generated yet!\n")

        # fixme: change to on-the-fly compute
        precom.m_U0Pre = [None] * LTMatrix_Row
        precom.m_U0hatTPre = [None] * LTMatrix_Row
        for i in range(LTMatrix_Row):
            # precom.m_U0hatTPre
            m_U0hatTPre_len = LTMatrix_mx_len * m_U0hatTPre_limbs
            m_U0hatTPre = [m_U0hatTPre_mx[i * m_U0hatTPre_len + j] for j in range(m_U0hatTPre_len)]
            precom.m_U0hatTPre[i] = Plaintext(m_U0hatTPre, LTMatrix_mx_len, LTMatrix_Column, m_U0hatTPre_limbs)

            # precom.m_U0Pre
            m_U0Pre_len = LTMatrix_mx_len * m_U0Pre_limbs
            m_U0Pre = [m_U0Pre_mx[i * m_U0Pre_len + j] for j in range(m_U0Pre_len)]
            precom.m_U0Pre[i] = Plaintext(m_U0Pre, LTMatrix_mx_len, LTMatrix_Column, m_U0Pre_limbs)
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
                        m_U0hatTPreFFT[LHScnt] = m_U0hatTPreFFT_mx.m_U0hatTPreFFT_mx[RHScnt]
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
                        m_U0PreFFT[LHScnt] = m_U0PreFFT_mx.m_U0PreFFT_mx[RHScnt]
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


def BootstrapTest_N65536L26lB44():
    slots = 64
    levelsRemaining = 3
    secretKeyDist = SecretKeyDist.UNIFORM_TERNARY
    rescaleTech = ScalingTechnique.FIXEDMANUAL
    dim1 = [0, 0]
    levelBudget = [4, 4]
    depth = levelsRemaining + get_bootstrap_depth(9, levelBudget, secretKeyDist)
    L0 = depth + 1
    dnum = 3
    K = math.ceil(L0 * 1.0 / dnum)
    L = L0
    logN = 16
    logp = 59
    N = 65536

    l = 2
    cipher_cv = np.loadtxt("/home/yons/Desktop/test_data/input_cipher.txt", dtype=np.uint64)
    ax_cipher = torch.tensor(cipher_cv[1], dtype=torch.uint64, device="cuda").reshape([l, N])
    bx_cipher = torch.tensor(cipher_cv[0], dtype=torch.uint64, device="cuda").reshape([l, N])
    cipher = Cipher([ax_cipher, bx_cipher], 2)

    moduliQ26 = np.array(
        [1152921504606584833, 576460752340123649, 576460752267509761, 576460752337502209, 576460752272228353,
         576460752331210753, 576460752273801217, 576460752329900033, 576460752279306241, 576460752329506817,
         576460752284418049, 576460752329113601, 576460752286253057, 576460752328327169, 576460752289005569,
         576460752325705729, 576460752289529857, 576460752321642497, 576460752289923073, 576460752319414273,
         576460752298180609, 576460752319021057, 576460752298835969, 576460752315482113, 576460752300015617,
         576460752308273153, ])
    rootsQ26 = np.array([
        18043022392882, 9864335377277, 8953348410340, 60935135015, 308940258959, 4927150527883, 1364616692108,
        4626619421836, 5167116140063, 51256291259317, 4216999963069, 3124074488816, 13706574615761, 26898031712068,
        12481347222717, 8161815494988, 1549889294979, 8917348739478, 4426162160102, 5029855326074, 8856820954566,
        1072858004773, 3047882667676, 9939870822671, 1034043136987, 3760097055997])
    moduliP9 = np.array([
        1152921504598720513, 1152921504597016577, 1152921504595968001, 1152921504592822273, 1152921504592429057,
        1152921504589938689, 1152921504586530817, 1152921504583647233, 1152921504581419009, ])
    rootsP9 = np.array([
        800790938143, 17749908910371, 11469071954203, 21482204621753, 6744827058362, 17679085976867, 19946736815584,
        102116018653, 10353721066739, ])

    keymap_key = np.loadtxt("/home/yons/Desktop/test_data/key_map_key.txt", dtype=np.int32)
    keymap_ax = np.loadtxt("/home/yons/Desktop/test_data/key_map_ax.txt", dtype=np.uint64)
    keymap_bx = np.loadtxt("/home/yons/Desktop/test_data/key_map_bx.txt", dtype=np.uint64)

    # swk = generate_random_uint64_array(2 * dnum * (L + K) * N).reshape(2, dnum, L + K, N)
    swk_ax = keymap_ax.reshape(dnum, L + K, N)
    swk_bx = keymap_bx.reshape(dnum, L + K, N)
    swk = [swk_bx, swk_ax]

    cryptoContext = Context(logN,
                            60, 59, 59,
                            L, K,
                            moduliQ26, moduliP9, rootsQ26, rootsP9, swk)
    cryptoContext.BsContext = BsContext(cryptoContext, levelBudget, dim1, slots, 0, rescaleTech, secretKeyDist)

    keymap0_ax = np.loadtxt("/home/yons/Desktop/test_data/key_map0_ax.txt", dtype=np.uint64).reshape(-1, N)
    keymap0_bx = np.loadtxt("/home/yons/Desktop/test_data/key_map0_bx.txt", dtype=np.uint64).reshape(-1, N)
    cryptoContext.key_map['0'] = [torch.tensor(keymap0_bx, dtype=torch.uint64, device="cuda"),
                                  torch.tensor(keymap0_ax, dtype=torch.uint64, device="cuda")]
    for j in range(dnum):
        ax = torch.tensor(keymap_ax[j].reshape(-1, N), dtype=torch.uint64, device="cuda")
        bx = torch.tensor(keymap_bx[j].reshape(-1, N), dtype=torch.uint64, device="cuda")
        cryptoContext.key_map[str(keymap_key[j + 1])] = [bx, ax]

    left_rotation_keymap_key = np.loadtxt("/home/yons/Desktop/test_data/leftRotKeyMap_key.txt", dtype=np.int64)
    left_rotation_keymap_ax = np.loadtxt("/home/yons/Desktop/test_data/leftRotKeyMap_ax.txt", dtype=np.uint64)
    left_rotation_keymap_bx = np.loadtxt("/home/yons/Desktop/test_data/leftRotKeyMap_bx.txt", dtype=np.uint64)
    for i in range(left_rotation_keymap_ax.shape[0]):
        ax = torch.tensor(left_rotation_keymap_ax[i].reshape(-1, N), dtype=torch.uint64, device="cuda")
        bx = torch.tensor(left_rotation_keymap_bx[i].reshape(-1, N), dtype=torch.uint64, device="cuda")
        cryptoContext.left_rot_key_map[str(left_rotation_keymap_key[i])] = [bx, ax]

    left_rotation_keymap_key_c2s = np.loadtxt("/home/yons/Desktop/test_data/leftRotKeyMap_key_c2s.txt", dtype=np.int64)
    left_rotation_keymap_ax_c2s = np.loadtxt("/home/yons/Desktop/test_data/leftRotKeyMap_ax_c2s.txt", dtype=np.uint64)
    left_rotation_keymap_bx_c2s = np.loadtxt("/home/yons/Desktop/test_data/leftRotKeyMap_bx_c2s.txt", dtype=np.uint64)
    for i in range(left_rotation_keymap_ax_c2s.shape[0]):
        ax = torch.tensor(left_rotation_keymap_ax_c2s[i].reshape(-1, N), dtype=torch.uint64, device="cuda")
        bx = torch.tensor(left_rotation_keymap_bx_c2s[i].reshape(-1, N), dtype=torch.uint64, device="cuda")
        if left_rotation_keymap_key_c2s[i] in cryptoContext.left_rot_key_map.keys():
            if (cryptoContext.left_rot_key_map[str(left_rotation_keymap_key_c2s[i])][0] != bx or
                    cryptoContext.left_rot_key_map[str(left_rotation_keymap_key_c2s[i])][0] != ax):
                print("errorrrrr! same key left_rotation_keymap_key_c2s", i)
                return
            else:
                continue
        cryptoContext.left_rot_key_map[str(left_rotation_keymap_key_c2s[i])] = [bx, ax]

    left_rotation_keymap_key_s2c = np.loadtxt("/home/yons/Desktop/test_data/leftRotKeyMap_key_s2c.txt", dtype=np.int64)
    left_rotation_keymap_ax_s2c = np.loadtxt("/home/yons/Desktop/test_data/leftRotKeyMap_ax_s2c.txt", dtype=np.uint64)
    left_rotation_keymap_bx_s2c = np.loadtxt("/home/yons/Desktop/test_data/leftRotKeyMap_bx_s2c.txt", dtype=np.uint64)
    for i in range(left_rotation_keymap_ax_s2c.shape[0]):
        ax = torch.tensor(left_rotation_keymap_ax_s2c[i].reshape(-1, N), dtype=torch.uint64, device="cuda")
        bx = torch.tensor(left_rotation_keymap_bx_s2c[i].reshape(-1, N), dtype=torch.uint64, device="cuda")
        if left_rotation_keymap_key_s2c[i] in cryptoContext.left_rot_key_map.keys():
            if (cryptoContext.left_rot_key_map[str(left_rotation_keymap_key_s2c[i])][0] != bx or
                    cryptoContext.left_rot_key_map[str(left_rotation_keymap_key_s2c[i])][0] != ax):
                print("errorrrrr! same key left_rotation_keymap_key_s2c", i)
                return
            else:
                continue
        cryptoContext.left_rot_key_map[str(left_rotation_keymap_key_s2c[i])] = [bx, ax]

    eval_bootstrap_setup(cryptoContext, levelBudget, dim1, slots, 0)

    result = cipher
    result = eval_bootstrap(cryptoContext, cipher, num_iterations=1, precision=0, rescaleTech=rescaleTech,
                   secretKeyDist=secretKeyDist, L0=L, slots=slots)

    result_answer_ax = np.loadtxt("/home/yons/Desktop/test_data/result_ax.txt", dtype=np.uint64)
    result_answer_bx = np.loadtxt("/home/yons/Desktop/test_data/result_bx.txt", dtype=np.uint64)

    result_ax = result.cv[1].cpu().numpy().reshape(-1)
    result_bx = result.cv[0].cpu().numpy().reshape(-1)
    # for i in range(result.cur_limbs * cryptoContext.N):
    #     if(result_ax[i] != result_answer_ax[i]):
    #         print(i, result_ax[i], result_answer_ax[i])
    #         break
    compare0 = np.array_equal(result_ax, result_answer_ax)
    compare1 = np.array_equal(result_bx, result_answer_bx)
    print(f"\ntest BootstrapTest_N65536L26lB44: \nresult: ")
    print(compare0)
    print(compare1)

