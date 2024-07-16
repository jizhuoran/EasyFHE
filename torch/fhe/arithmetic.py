import numpy as np
import math


def NTT(a, N, moduli, moduli_Inv, RootScalePows):
    t = N
    logt1 = int(math.log2(N)) + 1
    q = int(moduli)  # np
    qInv = int(moduli_Inv)  # np
    m = 1
    while m < N:
        t >>= 1
        logt1 -= 1
        for i in range(m):
            j1 = i << logt1
            j2 = j1 + t - 1
            W = RootScalePows[m + i]  # np
            for j in range(j1, j2 + 1):
                T = a[j + t]  # np
                U = int(T) * int(W)  # np
                U0 = U & 0xFFFFFFFFFFFFFFFF
                U0 = np.uint64(U0)
                U1 = U >> 64
                U1 = np.uint64(U1)
                Q = int(U0) * qInv
                Q = Q & 0xFFFFFFFFFFFFFFFF
                Hx = int(Q) * q
                H = Hx >> 64
                H = np.uint64(H)
                if U1 < H:
                    V = int(U1) + q - int(H)
                    V = np.uint64(V)
                else:
                    V = int(U1) - int(H)
                    V = np.uint64(V)
                if a[j] < V:
                    tmp = int(a[j]) + q - int(V)
                    a[j + t] = np.uint64(tmp)
                else:
                    a[j + t] = np.uint64(int(a[j]) - int(V))
                tmp = int(a[j])+int(V)
                if tmp > q:
                    a[j] = np.uint64(tmp-q)
                else:
                    a[j] = np.uint64(tmp)
        m = m << 1
    return a


def iNTT(a, N, moduli, moduli_Inv, RootScalePowsInv, NScaleInvModq):
    q = moduli
    qd = q * np.uint64(2)
    qInv = moduli_Inv
    t = 1
    m = N
    while m > 1:
        j1 = 0
        h = m >> 1
        for i in range(h):
            j2 = j1 + t - 1
            W = int(RootScalePowsInv[h + i])
            W = W & 0xFFFFFFFFFFFFFFFF
            for j in range(j1, j2 + 1):
                T = a[j] + qd
                T = T & 0xFFFFFFFFFFFFFFFF
                T -= a[j + t]
                a[j] += a[j + t]
                if a[j] >= qd:
                    a[j] -= qd
                UU = int(T) * int(W)
                U0 = UU & 0xFFFFFFFFFFFFFFFF
                U0 = np.uint64(U0)
                U1 = UU >> 64
                U1 = np.uint64(U1)
                U1 = U1 & 0xFFFFFFFFFFFFFFFF
                Q = int(U0) * int(qInv)
                Q = Q & 0xFFFFFFFFFFFFFFFF
                Hx = int(Q) * int(q)
                H = Hx >> 64
                H = np.uint64(H)
                a[j + t] = U1 + q
                a[j + t] -= H
            j1 += (t << 1)
        t <<= 1
        m >>= 1

    NScale = NScaleInvModq
    NScale = NScale & 0xFFFFFFFFFFFFFFFF
    for i in range(N):
        T = a[i] if a[i] < q else a[i] - q
        T = T & 0xFFFFFFFFFFFFFFFF
        U = int(T) * int(NScale)
        U0 = U & 0xFFFFFFFFFFFFFFFF
        U0 = np.uint64(U0)
        U1 = U >> 64
        U1 = np.uint64(U1)
        Q = U0 * qInv
        Q = Q & 0xFFFFFFFFFFFFFFFF
        Hx = int(Q) * int(q)
        H = Hx >> 64
        H = np.uint64(H)
        if U1 < H:
            a[i] = U1 + q - H
        else:
            a[i] = U1 - H
    return a

def vec_mod_int(a, MOD):
    N = len(a)
    c = [0] * int(N)
    for ri in range(N):
        c[ri] = np.uint64(int(a[ri]) % int(MOD))
    return c


def vec_add_int(a, b):
    assert len(a) == len(b)
    N = len(a)
    c = [0] * int(N)
    for ri in range(N):
        c[ri] = int(a[ri])+int(b[ri])
    return c


def vec_mul_scalar_int(a, scalar):
    N = len(a)
    c = [0]*int(N)
    for ri in range(N):
        c[ri] = int(a[ri]) * int(scalar)
    return c

def vec_mod(a, MOD):
    N = len(a)
    c = [0] * int(N)
    for ri in range(N):
        c[ri] = np.uint64(int(a[ri]) % int(MOD))
    return c

def vec_mul_scalar_mod(a, scalar, MOD):
    N = len(a)
    c = [0]*int(N)
    for ri in range(N):
        c[ri] = int(a[ri]) * int(scalar) % MOD
    return c

def vec_mul_mod(a, b, MOD):
    assert a.shape == b.shape
    N = a.shape[-1]
    c = [0]*int(N)
    for ri in range(N):
        c[ri] = np.uint64(int(a[ri]) * int(b[ri]) % int(MOD))
    return c

def vec_add_mod(a, b, MOD):
    assert len(a) == len(b)
    N = len(a)
    c = [0] * int(N)
    for ri in range(N):
        c[ri] = np.uint64((int(a[ri])+int(b[ri])) % int(MOD))
    return c

def vec_sub_mod(a, b, MOD):
    assert len(a) == len(b)
    N = len(a)
    c = [0] * int(N)
    for ri in range(N):
        c[ri] = np.uint64((int(a[ri])+ int(MOD) - int(b[ri])) % int(MOD))
    return c

def vec_switch_modulus(input, new_modulus, old_modulus):
    old_modulus_by_two = int(old_modulus) >> 1
    diff = old_modulus - new_modulus if old_modulus > new_modulus else new_modulus - old_modulus

    res = np.zeros(input.shape, np.uint64)
    if new_modulus >= old_modulus:
        tmp = [diff if n > old_modulus_by_two else 0 for n in input]
        res[:] = [n + t for n, t in zip(input, tmp)]
    else:  # new_modulus < old_modulus
        tmp = [diff if n > old_modulus_by_two else 0 for n in input]
        res[:] = vec_sub_mod(input, tmp, new_modulus)

    return res
######################################################
## KS components
######################################################

def ModUp_Core(intt_a, d2Tilde,
               moduliQ, moduliP, QHatInvModq, QHatModp,
               curr_limbs, K, N):
    beta = int(math.ceil(curr_limbs / K))  # total beta groups
    for j in range(beta):
        in_C_L_index = j * K
        in_C_L_len = K if j < (beta - 1) else (curr_limbs - in_C_L_index)
        sizeP = curr_limbs - in_C_L_len + K

        a = intt_a[in_C_L_index:in_C_L_index + in_C_L_len, :]
        qi = moduliQ[in_C_L_index:in_C_L_index + in_C_L_len]

        qi_comple = np.concatenate((moduliQ[:in_C_L_index], moduliQ[in_C_L_index + in_C_L_len:]))
        moduliQP = np.concatenate((qi_comple, moduliP))

        assert moduliQP.size == sizeP, "moduliQP.size() should equal to sizeP, check again"

        sum = [[int(0) for _ in range(N)] for _ in range(sizeP)]

        for i in range(in_C_L_len):
            tmp = vec_mul_scalar_mod(a[i], QHatInvModq[j][in_C_L_len - 1][i], qi[i])
            for k in range(sizeP):
                product = vec_mul_scalar_int(tmp, QHatModp[curr_limbs - 1][j][i][k])
                sum[k] = vec_add_int(sum[k], product)

        ranges = list(range(0, in_C_L_index)) + list(range(in_C_L_index + in_C_L_len, curr_limbs + K))
        for k, i in enumerate(ranges):
            d2Tilde[j][i] = vec_mod_int(sum[k], moduliQP[k])

def ModDown_Core(intt_a,
                 pHatInvModp, pHatModq, PInvModq,
                 moduliQ, moduliP,
                 N, curr_limbs, K,):

    res = np.zeros((curr_limbs, N), dtype=np.uint64)
    tmp3 = np.zeros((K, N), dtype=np.uint64)
    tmpk = intt_a[curr_limbs:, :]
    for k in range(K):
        tmp3[k] = vec_mul_scalar_mod(tmpk[k], pHatInvModp[k], moduliP[k])

    tmpi = intt_a[:curr_limbs, :]
    for i in range(curr_limbs):
        sum = [int(0) for _ in range(N)]
        for k in range(K):
            product = vec_mul_scalar_int(tmp3[k], pHatModq[k][i])
            sum = vec_add_int(sum, product)

        res[i] = vec_mod_int(sum, moduliQ[i])
        res[i] = vec_sub_mod(tmpi[i], res[i], moduliQ[i])
        res[i] = vec_mul_scalar_mod(res[i], PInvModq[i], moduliQ[i])

    return res
