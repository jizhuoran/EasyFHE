from math import log, pi, cos, sin
from . import number_theory as nbtheory
import numpy as np
import math
import random
import warnings

def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    return f"{message}\n"

warnings.formatwarning = custom_warning_format

class Context:
    def __init__(self, logN, logq0, logqi, logp, L, K,
                 moduliQ = None, moduliP = None, rootsQ= None, rootsP=None, MULT_SWK= None,
                 h=64, sigma=32,):

        self.logN = logN
        self.logqi = logqi
        self.L = int(L)
        self.K = int(K)
        self.dnum = int(L / K)
        self.h = h
        self.sigma = sigma
        self.N = int(1 << logN)
        self.M = self.N << 1
        self.logNh = logN - 1
        self.Nh = self.N >> 1
        self.p = 1 << logqi

        self.moduliQ = [0] * L
        self.qrVec = [0] * L
        self.qTwok = [0] * L
        self.qkVec = [0] * L
        self.qdVec = [0] * L
        self.qInvVec = [0] * L
        self.qRoots = [0] * L
        self.qRootsInv = [0] * L
        self.qRootPows = [[] for _ in range(L)]
        self.qRootScalePows = [[] for _ in range(L)]
        self.qRootScalePowsOverq = [[] for _ in range(L)]
        self.qRootScalePowsInv = [[] for _ in range(L)]
        self.qRootPowsInv = [[] for _ in range(L)]
        self.NInvModq = [0] * L
        self.NScaleInvModq = [0] * L
        bnd = 1
        cnt = 1
        if moduliQ is None and rootsQ is None:
            while True:
                prime = (1 << logq0) + bnd * self.M + 1
                if nbtheory.is_prime(prime):
                    self.moduliQ[0] = prime
                    break
                bnd += 1
            # self.qRoots[i] = self.findMthRootOfUnity(self.M, self.moduliQ[i])
            self.qRoots[0] = nbtheory.root_of_unity(order=self.M, modulus=self.moduliQ[0])
            # print("moduliQ[0]", self.moduliQ[0])
            bnd = 1
            while cnt < L:
                prime1 = (1 << logqi) + bnd * self.M + 1
                if self.primeTest(prime1):
                    self.moduliQ[cnt] = prime1
                    cnt += 1
                prime2 = (1 << logqi) - bnd * self.M + 1
                if self.primeTest(prime2):
                    self.moduliQ[cnt] = prime2
                    # self.qRoots[i] = self.findMthRootOfUnity(self.M, self.moduliQ[i])
                    self.qRoots[cnt] = nbtheory.root_of_unity(order=self.M, modulus=self.moduliQ[cnt - 1])
                    cnt += 1
                bnd += 1

            if logqi - logN - 1 - math.ceil(math.log2(bnd)) < 10:
                print("ERROR: too small number of precision")
                print("TRY to use larger logqi or smaller depth")
        else:
            if moduliQ is None:
                print("moduliQ needs to be set!")
                return
            elif rootsQ is None:
                print("rootsQ needs to be set!")
                return
            for i in range(L):
                self.moduliQ[i] = moduliQ[i]
                self.qRoots[i] = rootsQ[i]

        for i in range(L):
            # print(i)
            self.qTwok[i] = 2 * (int(math.log2(self.moduliQ[i])) + 1)
            self.qrVec[i] = (1 << self.qTwok[i]) // self.moduliQ[i]
            self.qkVec[i] = ((nbtheory.mod_inv(1 << 62, self.moduliQ[i]) << 62) - 1) // self.moduliQ[i]
            self.qRootsInv[i] = nbtheory.mod_inv(self.qRoots[i], int(self.moduliQ[i]))
            self.NInvModq[i] = nbtheory.mod_inv(self.N, int(self.moduliQ[i]))
            self.NScaleInvModq[i] = self.mulMod(int(self.NInvModq[i]), int(1 << 32), int(self.moduliQ[i]))
            self.NScaleInvModq[i] = self.mulMod(int(self.NScaleInvModq[i]), int(1 << 32), int(self.moduliQ[i]))
            self.qInvVec[i] = self.inv(self.moduliQ[i])
            self.qRootPows[i] = [0] * self.N
            self.qRootPowsInv[i] = [0] * self.N
            self.qRootScalePows[i] = [0] * self.N
            self.qRootScalePowsOverq[i] = [0] * self.N
            self.qRootScalePowsInv[i] = [0] * self.N
            power = int(1)
            powerInv = int(1)
            for j in range(self.N):
                jprime = self.bitReverse(j) >> (32 - self.logN)
                self.qRootPows[i][jprime] = int(power)
                # tmp = (power << 64)
                tmp = (int(power) << 64)
                self.qRootScalePowsOverq[i][jprime] = int(tmp // int(self.moduliQ[i]))
                self.qRootScalePows[i][jprime] = int(
                    self.mulMod(int(self.qRootPows[i][jprime]), int(1 << 32), int(self.moduliQ[i])))
                self.qRootScalePows[i][jprime] = int(
                    self.mulMod(int(self.qRootScalePows[i][jprime]), int(1 << 32), int(self.moduliQ[i])))
                self.qRootPowsInv[i][jprime] = int(powerInv)
                self.qRootScalePowsInv[i][jprime] = int(
                    self.mulMod(int(self.qRootPowsInv[i][jprime]), int(1 << 32), int(self.moduliQ[i])))
                self.qRootScalePowsInv[i][jprime] = int(
                    self.mulMod(int(self.qRootScalePowsInv[i][jprime]), int(1 << 32), int(self.moduliQ[i])))
                if j < self.N - 1:
                    power = self.mulMod(int(power), int(self.qRoots[i]), int(self.moduliQ[i]))
                    powerInv = self.mulMod(powerInv, int(self.qRootsInv[i]), int(self.moduliQ[i]))
        q_mu = [] # for barret mul mod
        for mod in self.moduliQ:
            x = 2**128 // mod
            low = x & ((1 << 64) - 1)  # 取低64位
            high = x >> 64  # 取高64位
            q_mu.append([low, high])
        self.q_mu = np.array(q_mu, dtype=np.uint64)

        self.moduliP = [0] * self.K
        self.prVec = [0] * self.K
        self.pTwok = [0] * self.K
        self.pkVec = [0] * self.K
        self.pdVec = [0] * self.K
        self.pInvVec = [0] * self.K
        self.pRoots = [0] * self.K
        self.pRootsInv = [0] * self.K
        self.pRootPows = [[] for _ in range(self.K)]
        self.pRootPowsInv = [[] for _ in range(self.K)]
        self.pRootScalePows = [[] for _ in range(self.K)]
        self.pRootScalePowsOverp = [[] for _ in range(self.K)]
        self.pRootScalePowsInv = [[] for _ in range(self.K)]
        self.NInvModp = [0] * self.K
        self.NScaleInvModp = [0] * self.K

        if moduliP is None and rootsP is None:
            cnt = 0
            while cnt < self.K:
                prime1 = (1 << logp) + bnd * self.M + 1
                if self.primeTest(prime1):
                    self.moduliP[cnt] = prime1
                    self.pRoots[cnt] = nbtheory.root_of_unity(order=self.M, modulus=self.moduliP[cnt])
                    cnt += 1
                if cnt == self.K:
                    break
                prime2 = (1 << logp) - bnd * self.M + 1
                if self.primeTest(prime2):
                    self.moduliP[cnt] = prime2
                    self.pRoots[cnt] = nbtheory.root_of_unity(order=self.M, modulus=self.moduliP[cnt])
                    cnt += 1
                bnd += 1

        else:
            if moduliP is None:
                print("moduliP needs to be set")
                return
            elif rootsP is None:
                print("rootsP needs to be set")
                return
            for i in range(K):
                self.moduliP[i] = moduliP[i]
                self.pRoots[i] = rootsP[i]

        for i in range(K):
            # print(i)
            self.pTwok[i] = 2 * (int(math.log2(self.moduliP[i])) + 1)
            self.prVec[i] = (1 << self.pTwok[i]) // self.moduliP[i]
            self.pkVec[i] = ((nbtheory.mod_inv(1 << 62, self.moduliP[i]) << 62) - 1) // self.moduliP[i]
            self.pRootsInv[i] = nbtheory.mod_inv(self.pRoots[i], int(self.moduliP[i]))
            self.NInvModp[i] = nbtheory.mod_inv(self.N, int(self.moduliP[i]))
            self.NScaleInvModp[i] = self.mulMod(int(self.NInvModp[i]), int(1 << 32), int(self.moduliP[i]))
            self.NScaleInvModp[i] = self.mulMod(int(self.NScaleInvModp[i]), int(1 << 32), int(self.moduliP[i]))
            self.pInvVec[i] = self.inv(self.moduliP[i])
            self.pRootPows[i] = [0] * self.N
            self.pRootPowsInv[i] = [0] * self.N
            self.pRootScalePows[i] = [0] * self.N
            self.pRootScalePowsOverp[i] = [0] * self.N
            self.pRootScalePowsInv[i] = [0] * self.N
            power = int(1)
            powerInv = int(1)
            for j in range(self.N):
                jprime = self.bitReverse(j) >> (32 - self.logN)
                self.pRootPows[i][jprime] = int(power)
                tmp = (int(power) << 64)
                self.pRootScalePowsOverp[i][jprime] = tmp // self.moduliP[i]
                self.pRootScalePows[i][jprime] = self.mulMod(self.pRootPows[i][jprime], int(1 << 32),
                                                             int(self.moduliP[i]))
                self.pRootScalePows[i][jprime] = self.mulMod(self.pRootScalePows[i][jprime], int(1 << 32),
                                                             int(self.moduliP[i]))
                self.pRootPowsInv[i][jprime] = powerInv
                self.pRootScalePowsInv[i][jprime] = self.mulMod(self.pRootPowsInv[i][jprime], int(1 << 32),
                                                                int(self.moduliP[i]))
                self.pRootScalePowsInv[i][jprime] = self.mulMod(self.pRootScalePowsInv[i][jprime], int(1 << 32),
                                                                int(self.moduliP[i]))
                if j < self.N - 1:
                    power = self.mulMod(power, int(self.pRoots[i]), int(self.moduliP[i]))
                    powerInv = self.mulMod(powerInv, int(self.pRootsInv[i]), int(self.moduliP[i]))

        p_mu = [] # for barret mul mod
        for mod in self.moduliP:
            x = 2**128 // mod
            low = x & ((1 << 64) - 1)  # 取低64位
            high = x >> 64  # 取高64位
            p_mu.append([low, high])
        self.p_mu = np.array(p_mu, dtype=np.uint64)

        moduliPartQ = [0] * self.dnum
        for j in range(self.dnum):
            moduliPartQ[j]= int(1)
            for i in range(K*j, K*(j+1)):
                if i<L:
                    moduliPartQ[j] *= int(self.moduliQ[i])

        self.PartQlHatInvModq = [[[0 for _ in range(K)] for _ in range(K)] for _ in range(self.dnum)]
        for k in range(self.dnum):
            sizePartQk = (L - (k * K)) if (k == self.dnum - 1) else K
            modulusPartQ = moduliPartQ[k]
            for l in range(sizePartQk):
                if l > 0:
                    modulusPartQ = int(int(modulusPartQ) // int(self.moduliQ[(k + 1) * K - l]))
                for i in range(sizePartQk - l):
                    moduli = int(self.moduliQ[k * K + i])
                    QHat = modulusPartQ // moduli
                    QHatInvModqi = int(self.invMod(QHat, moduli))
                    self.PartQlHatInvModq[k][sizePartQk - l - 1][i] = QHatInvModqi

        # 初始化 PartQlHatModp
        self.PartQlHatModp = [[[[0 for _ in range(self.dnum * K)] for _ in range(K)] for _ in range(self.dnum)] for _ in range(L)]
        for l in range(L):
            beta = math.ceil((l + 1) / K)
            for k in range(beta):
                partQ_size = (L - (beta - 1) * K) if (beta == self.dnum and k == beta - 1) else K
                digitSize = K
                modulusPartQ = int(moduliPartQ[k])

                if k == beta - 1:
                    digitSize = l + 1 - k * K
                    for idx in range(digitSize, partQ_size):
                        modulusPartQ //= int(self.moduliQ[K * k + idx])

                for i in range(digitSize):
                    partQHat = modulusPartQ // int(self.moduliQ[K * k + i])

                    start_idx = k * K
                    end_idx = start_idx + digitSize
                    complBasis_vec = (
                            self.moduliQ[:start_idx] + self.moduliQ[end_idx:l + 1] + self.moduliP
                    )

                    for j, mod in enumerate(complBasis_vec):
                        QHatModpj = int(partQHat) % int(mod)
                        self.PartQlHatModp[l][k][i][j] = QHatModpj

        self.pHatModp = [0] * K  # 初始化 pHatModp 列表
        self.pHatInvModp = [0] * K  # 初始化 pHatInvModp 列表
        # 计算 pHatModp
        for k in range(K):
            self.pHatModp[k] = int(1)
            for j in list(range(k)) + list(range(k + 1, K)):
                temp = int(self.moduliP[j] % self.moduliP[k])
                self.pHatModp[k] = (self.pHatModp[k] * temp) % int(self.moduliP[k])

        # 计算 pHatInvModp # [k] qhat_k^-1 mod q_k
        for k in range(K):
            self.pHatInvModp[k] = int(self.invMod(int(self.pHatModp[k]), self.moduliP[k]))

        # 初始化 pHatModq
        self.pHatModq = [[0] * L for _ in range(K)]
        for k in range(K):
            for i in range(L):
                self.pHatModq[k][i] = int(1)
                for s in list(range(k)) + list(range(k + 1, K)):
                    temp = int(self.moduliP[s]) % int(self.moduliQ[i])
                    self.pHatModq[k][i] = self.mulMod(int(self.pHatModq[k][i]), temp, int(self.moduliQ[i]))

        self.PModq = [0] * L  # 初始化 PModq

        # 计算 PModq
        for i in range(L):
            self.PModq[i] = int(1)
            for k in range(K):
                temp = self.moduliP[k] % self.moduliQ[i]
                self.PModq[i] = self.mulMod(int(self.PModq[i]), int(temp), int(self.moduliQ[i]))

        self.PInvModq = [0] * L  # 初始化 PInvModq
        # 计算 PInvModq
        for i in range(L):
            self.PInvModq[i] = self.invMod(int(self.PModq[i]), int(self.moduliQ[i]))

        self.qInvModq = [[0 for _ in range(L)] for _ in range(L)]
        for i in range(L):
            for j in list(range(i))+list(range(i+1, L)):
                self.qInvModq[i][j] = self.invMod(int(self.moduliQ[i]), int(self.moduliQ[j]))

        # rescale param
        # sizeQ in openFHE equals to L here.
        self.QlQlInvModqlDivqlModq = [[0] * (L - 1) for _ in range(L - 1)]
        # self.QlQlInvModqlDivqlModq = [None] * (L - 1)
        for k in range(L - 1):
            l = L - (k + 1)
            # self.QlQlInvModqlDivqlModq[k] = [0] * l

            for i in range(l):
                QlInvModql = int(1)

                for j in range(l):
                    temp = self.invMod(self.moduliQ[j], self.moduliQ[l])
                    QlInvModql = self.mulMod(int(QlInvModql), int(temp), int(self.moduliQ[l]))

                modulusQ = int(1)
                for j in range(l):
                    modulusQ *= int(self.moduliQ[j])

                result = int((int(QlInvModql) * modulusQ) // int(self.moduliQ[l]))
                result %= int(self.moduliQ[i])

                self.QlQlInvModqlDivqlModq[k][i] = np.uint64(result)

        self.mult_swk = np.zeros((2, self.dnum, L + K, self.N), dtype=np.uint64)
        if MULT_SWK is None:
            warnings.warn(
                "\n------------------------\n"
                "MULT_SWK needs to be set"
                "\n------------------------\n", UserWarning)
            # todo: set data in numpy array
        else:
            self.mult_swk[0] = MULT_SWK[0]
            self.mult_swk[1] = MULT_SWK[1]

        self.moduliQ = np.array(self.moduliQ, dtype=np.uint64)
        self.qrVec = np.array(self.qrVec, dtype=np.uint64)
        self.qTwok = np.array(self.qTwok, dtype=np.uint64)
        self.qkVec = np.array(self.qkVec, dtype=np.uint64)
        self.qdVec = np.array(self.qdVec, dtype=np.uint64)
        self.moduliP = np.array(self.moduliP, dtype=np.uint64)
        self.prVec = np.array(self.prVec, dtype=np.uint64)
        self.pTwok = np.array(self.pTwok, dtype=np.uint64)
        self.pkVec = np.array(self.pkVec, dtype=np.uint64)
        self.pdVec = np.array(self.pdVec, dtype=np.uint64)
        self.qRoots = np.array(self.qRoots, dtype=np.uint64)
        self.pRoots = np.array(self.pRoots, dtype=np.uint64)

        self.qInvVec = np.array(self.qInvVec, dtype=np.uint64)
        self.pInvVec = np.array(self.pInvVec, dtype=np.uint64)
        self.qRootScalePows = np.array(self.qRootScalePows, dtype=np.uint64)
        self.pRootScalePows = np.array(self.pRootScalePows, dtype=np.uint64)
        self.qRootScalePowsInv = np.array(self.qRootScalePowsInv, dtype=np.uint64)
        self.pRootScalePowsInv = np.array(self.pRootScalePowsInv, dtype=np.uint64)
        self.NInvModq = np.array(self.NInvModq, dtype=np.uint64)
        self.NInvModp = np.array(self.NInvModp, dtype=np.uint64)
        self.NScaleInvModq = np.array(self.NScaleInvModq, dtype=np.uint64)
        self.NScaleInvModp = np.array(self.NScaleInvModp, dtype=np.uint64)
        self.QHatInvModq = np.array(self.PartQlHatInvModq, dtype=np.uint64)
        self.QHatModp = np.array(self.PartQlHatModp, dtype=np.uint64)
        self.pHatInvModp = np.array(self.pHatInvModp, dtype=np.uint64)
        self.pHatModq = np.array(self.pHatModq, dtype=np.uint64)
        self.PInvModq = np.array(self.PInvModq, dtype=np.uint64)

        self.PartQlHatInvModq = np.array(self.PartQlHatInvModq, dtype=np.uint64)
        self.PartQlHatModp = np.array(self.PartQlHatModp, dtype=np.uint64)
        self.pHatModp = np.array(self.pHatModp, dtype=np.uint64)
        self.pHatInvModp = np.array(self.pHatInvModp, dtype=np.uint64)
        self.pHatModq = np.array(self.pHatModq, dtype=np.uint64)
        self.PModq = np.array(self.PModq, dtype=np.uint64)
        self.qInvModq = np.array(self.qInvModq, dtype=np.uint64)
        self.QlQlInvModqlDivqlModq = np.array(self.QlQlInvModqlDivqlModq, dtype=np.uint64)



    def negate(self, r, a):
        r = -a

    def addMod(self, r, a, b, m):
        r = (a + b) % m

    def subMod(self, r, a, b, m):
        r = b % m
        r = (a + m - r) % m

    def mulMod(self, a, b, m):
        mul = (a % m) * (b % m)
        mul %= m
        return int(mul)

    def mulModBarrett(self, r, a, b, p, pr, twok):
        mul = (a % p) * (b % p)
        self.modBarrett(r, mul, p, pr, twok)

    def modBarrett(self, r, a, m, mr, twok):
        tmp = (a * mr) >> twok
        tmp *= m
        tmp = a - tmp
        r = tmp
        if r < m:
            return
        else:
            r -= m
            return

    def invMod(self, x, m):
        temp = int(x) % int(m)
        if self.gcd(temp, m) != 1:
            raise ValueError("Inverse doesn't exist!!!")
        else:
            return self.powMod(int(temp), (int(m) - 2), int(m))

    def powMod(self, x, y, modulus):
        res = 1
        while y > 0:
            if y & 1:
                res = self.mulMod(res, x, modulus)
            y = y >> 1
            x = self.mulMod(x, x, modulus)
        return res

    def inv(self, x):
        UINT64_MAX = 0xffffffffffffffff
        return pow(int(x), UINT64_MAX, UINT64_MAX + 1)

    def pow(self, x, y):
        res = 1
        while y > 0:
            if y & 1:
                res *= x
            y = y >> 1
            x *= x
        return res

    def bitReverse(self, n, bit_size=32):
        reversed_bits = 0
        for i in range(bit_size):
            # 将 n 的最低有效位移到 reversed_bits 的适当位置
            reversed_bits <<= 1
            reversed_bits |= (n & 1)
            n >>= 1
        return reversed_bits

    def gcd(self, a, b):
        if a == 0:
            return b
        return self.gcd(int(b) % int(a), int(a))

    def findPrimeFactors(self, s, number):
        while number % 2 == 0:
            s.add(2)
            number //= 2
        for i in range(3, int(math.sqrt(number)) + 1):
            while number % i == 0:
                s.add(i)
                number //= i
        if number > 2:
            s.add(number)

    def findPrimitiveRoot(self, modulus):
        s = set()
        phi = modulus - 1
        self.findPrimeFactors(s, phi)
        for r in range(2, phi + 1):
            flag = False
            for prime in s:
                if self.powMod(r, phi // prime, modulus) == 1:
                    flag = True
                    break
            if not flag:
                return r
        return -1

    def findMthRootOfUnity(self, M, mod):
        res = self.findPrimitiveRoot(mod)
        if (mod - 1) % M == 0:
            factor = (mod - 1) // M
            res = self.powMod(res, factor, mod)
            return res
        else:
            return -1

    # Miller-Rabin Prime Test #
    def primeTest(self, p):
        if p < 2:
            return False
        if p != 2 and p % 2 == 0:
            return False
        s = p - 1
        while s % 2 == 0:
            s //= 2
        for _ in range(200):
            temp1 = random.getrandbits(64)
            temp1 = (temp1 << 32) | random.getrandbits(32)
            temp1 = temp1 % (p - 1) + 1
            temp2 = s
            mod = self.powMod(temp1, temp2, p)
            while temp2 != p - 1 and mod != 1 and mod != p - 1:
                mod = self.mulMod(mod, mod, p)
                temp2 *= 2
            if mod != p - 1 and temp2 % 2 == 0:
                return False
        return True
    def method(self): # function to initialize variables
        pass