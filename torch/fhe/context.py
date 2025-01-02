
import numpy as np
import math
import random
import warnings
import torch
import pickle
import sympy

K_UNIFORM = 512

def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    return f"{message}\n"

class Context:
    def __init__(
        self,
        logN,
        logSlots,
        logq0, #todo: rename to firstMod
        logqi, #todo: rename to dcrtBits
        logp, #todo: rename to specialMod
        L,
        K,
        levelBudget,
        moduliQ=None,
        moduliP=None,
        rootsQ=None,
        rootsP=None,
        MULT_SWK=None,
        ROT_SWK=None,
        BOOT_KEY=None,
        secretKeyDist=None,
        rescaleTech=None,
        h=64,
        sigma=32
    ):
        self.levelBudget = levelBudget
        self.logSlots = logSlots
        self.secretKeyDist = secretKeyDist
        self.rescaleTech = rescaleTech
        self.BsContext = None
        self.logp = logp
        self.slots = 1 << logSlots #todo: need move slots to cipher
        self.qVec = None
        self.left_rot_key_map = {} #{index: [ax， bx]}
        self.key_map = None
        self.correctionFactor = 0

        self.logN = logN
        self.logqi = logqi
        self.L = int(L)
        self.K = int(K)
        self.dnum = math.ceil(L / K)
        self.h = h
        self.sigma = sigma
        self.N = int(1 << logN)
        self.M = self.N << 1
        self.logNh = logN - 1
        self.Nh = self.N >> 1
        self.p = 1 << logqi

        self.moduliQ = [0] * L
        self.qRoots = [0] * L
        self.qRootsInv = [0] * L
        self.qRootPows = [[] for _ in range(L)]
        self.qRootScalePows = [[] for _ in range(L)]
        self.qRootScalePowsOverq = [[] for _ in range(L)]
        self.qRootScalePowsInv = [[] for _ in range(L)]
        self.qRootPowsInv = [[] for _ in range(L)]
        self.auto_index = {}
        bnd = 1
        cnt = 1
        if moduliQ is None and rootsQ is None:
            while True:
                prime = (1 << logq0) + bnd * self.M + 1
                if self.is_prime(prime):
                    self.moduliQ[0] = prime
                    break
                bnd += 1
            # self.qRoots[i] = self.findMthRootOfUnity(self.M, self.moduliQ[i])
            self.qRoots[0] = self.root_of_unity(
                order=self.M, modulus=self.moduliQ[0]
            )
            # print("moduliQ[0]", self.moduliQ[0])
            bnd = 1
            while cnt < L:
                prime1 = (1 << logqi) + bnd * self.M + 1
                if self.is_prime(prime1):
                    self.moduliQ[cnt] = prime1
                    cnt += 1
                prime2 = (1 << logqi) - bnd * self.M + 1
                if self.is_prime(prime2):
                    self.moduliQ[cnt] = prime2
                    # self.qRoots[i] = self.findMthRootOfUnity(self.M, self.moduliQ[i])
                    self.qRoots[cnt] = self.root_of_unity(
                        order=self.M, modulus=self.moduliQ[cnt - 1]
                    )
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
            self.qRootsInv[i] = self.invMod(self.qRoots[i], int(self.moduliQ[i]))
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
                tmp = int(power) << 64
                self.qRootScalePowsOverq[i][jprime] = int(tmp // int(self.moduliQ[i]))
                self.qRootScalePows[i][jprime] = int(
                    self.mulMod(
                        int(self.qRootPows[i][jprime]),
                        int(1 << 32),
                        int(self.moduliQ[i]),
                    )
                )
                self.qRootScalePows[i][jprime] = int(
                    self.mulMod(
                        int(self.qRootScalePows[i][jprime]),
                        int(1 << 32),
                        int(self.moduliQ[i]),
                    )
                )
                self.qRootPowsInv[i][jprime] = int(powerInv)
                self.qRootScalePowsInv[i][jprime] = int(
                    self.mulMod(
                        int(self.qRootPowsInv[i][jprime]),
                        int(1 << 32),
                        int(self.moduliQ[i]),
                    )
                )
                self.qRootScalePowsInv[i][jprime] = int(
                    self.mulMod(
                        int(self.qRootScalePowsInv[i][jprime]),
                        int(1 << 32),
                        int(self.moduliQ[i]),
                    )
                )
                if j < self.N - 1:
                    power = self.mulMod(
                        int(power), int(self.qRoots[i]), int(self.moduliQ[i])
                    )
                    powerInv = self.mulMod(
                        powerInv, int(self.qRootsInv[i]), int(self.moduliQ[i])
                    )
        q_mu = []  # for barret mul mod
        for mod in self.moduliQ:
            x = 2**128 // int(mod)
            low = x & ((1 << 64) - 1)  # 取低64位
            high = x >> 64  # 取高64位
            q_mu.append([low, high])
        self.q_mu = np.array(q_mu, dtype=np.uint64)
        self.q_mu_cuda = torch.from_numpy(np.array(q_mu, dtype=np.uint64)).cuda()
        self.moduliQ_cuda = torch.from_numpy(np.array(self.moduliQ, dtype=np.uint64)).cuda()

        self.moduliP = [0] * self.K
        self.pInvVec = [0] * self.K
        self.pRoots = [0] * self.K
        self.pRootsInv = [0] * self.K
        self.pRootPows = [[] for _ in range(self.K)]
        self.pRootPowsInv = [[] for _ in range(self.K)]
        self.pRootScalePows = [[] for _ in range(self.K)]
        self.pRootScalePowsOverp = [[] for _ in range(self.K)]
        self.pRootScalePowsInv = [[] for _ in range(self.K)]

        if moduliP is None and rootsP is None:
            cnt = 0
            while cnt < self.K:
                prime1 = (1 << logp) + bnd * self.M + 1
                if self.is_prime(prime1):
                    self.moduliP[cnt] = prime1
                    self.pRoots[cnt] = self.root_of_unity(
                        order=self.M, modulus=self.moduliP[cnt]
                    )
                    cnt += 1
                if cnt == self.K:
                    break
                prime2 = (1 << logp) - bnd * self.M + 1
                if self.is_prime(prime2):
                    self.moduliP[cnt] = prime2
                    self.pRoots[cnt] = self.root_of_unity(
                        order=self.M, modulus=self.moduliP[cnt]
                    )
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
            self.pRootsInv[i] = self.invMod(self.pRoots[i], int(self.moduliP[i]))
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
                tmp = int(power) << 64
                self.pRootScalePowsOverp[i][jprime] = tmp // int(self.moduliP[i])
                self.pRootScalePows[i][jprime] = self.mulMod(
                    self.pRootPows[i][jprime], int(1 << 32), int(self.moduliP[i])
                )
                self.pRootScalePows[i][jprime] = self.mulMod(
                    self.pRootScalePows[i][jprime], int(1 << 32), int(self.moduliP[i])
                )
                self.pRootPowsInv[i][jprime] = powerInv
                self.pRootScalePowsInv[i][jprime] = self.mulMod(
                    self.pRootPowsInv[i][jprime], int(1 << 32), int(self.moduliP[i])
                )
                self.pRootScalePowsInv[i][jprime] = self.mulMod(
                    self.pRootScalePowsInv[i][jprime],
                    int(1 << 32),
                    int(self.moduliP[i]),
                )
                if j < self.N - 1:
                    power = self.mulMod(
                        power, int(self.pRoots[i]), int(self.moduliP[i])
                    )
                    powerInv = self.mulMod(
                        powerInv, int(self.pRootsInv[i]), int(self.moduliP[i])
                    )

        p_mu = []  # for barret mul mod
        for mod in self.moduliP:
            x = 2**128 // int(mod)
            low = x & ((1 << 64) - 1)  # 取低64位
            high = x >> 64  # 取高64位
            p_mu.append([low, high])
        self.p_mu = np.array(p_mu, dtype=np.uint64)

        moduliPartQ = [0] * self.dnum
        for j in range(self.dnum):
            moduliPartQ[j] = int(1)
            for i in range(K * j, K * (j + 1)):
                if i < L:
                    moduliPartQ[j] *= int(self.moduliQ[i])

        self.PartQlHatInvModq = [
            [[0 for _ in range(K)] for _ in range(K)] for _ in range(self.dnum)
        ]
        for k in range(self.dnum):
            sizePartQk = (L - (k * K)) if (k == self.dnum - 1) else K
            modulusPartQ = moduliPartQ[k]
            for l in range(sizePartQk):
                if l > 0:
                    modulusPartQ = int(
                        int(modulusPartQ) // int(self.moduliQ[k * K + sizePartQk - l])
                    )
                for i in range(sizePartQk - l):
                    moduli = int(self.moduliQ[k * K + i])
                    QHat = modulusPartQ // moduli
                    QHatInvModqi = int(self.invMod(QHat, moduli))
                    self.PartQlHatInvModq[k][sizePartQk - l - 1][i] = QHatInvModqi

        # 初始化 PartQlHatModp
        self.PartQlHatModp = [
            [
                [[0 for _ in range(self.dnum * K)] for _ in range(K)]
                for _ in range(self.dnum)
            ]
            for _ in range(L)
        ]
        for l in range(L):
            beta = math.ceil((l + 1) / K)
            for k in range(beta):
                partQ_size = (
                    (L - (beta - 1) * K) if (beta == self.dnum and k == beta - 1) else K
                )
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
                        self.moduliQ[:start_idx]
                        + self.moduliQ[end_idx : l + 1]
                        + self.moduliP
                    )

                    for j, mod in enumerate(complBasis_vec):
                        QHatModpj = int(partQHat) % int(mod)
                        self.PartQlHatModp[l][k][i][j] = QHatModpj

        # 初始化 PartQlHatModp_pad
        self.PartQlHatModp_pad = [
            [
                [[0 for _ in range(self.dnum * K)] for _ in range(K)]
                for _ in range(self.dnum)
            ]
            for _ in range(L)
        ]
        for l in range(L):
            beta = math.ceil((l + 1) / K)
            ceil_curr_limbs = beta * K
            for k in range(beta):
                partQ_size = (
                    (L - (beta - 1) * K) if (beta == self.dnum and k == beta - 1) else K
                )
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
                        self.moduliQ[:start_idx] + self.moduliQ[end_idx : l + 1]
                    )
                    offset = len(complBasis_vec)
                    for j, mod in enumerate(complBasis_vec):
                        QHatModpj = int(partQHat) % int(mod)
                        self.PartQlHatModp_pad[l][k][i][j] = QHatModpj

                    complBasis_vec = self.moduliQ[l + 1 : ceil_curr_limbs]
                    for j, mod in enumerate(complBasis_vec):
                        self.PartQlHatModp_pad[l][k][i][offset + j] = 0

                    complBasis_vec = self.moduliP
                    offset = ceil_curr_limbs - K
                    for j, mod in enumerate(complBasis_vec):
                        QHatModpj = int(partQHat) % int(mod)
                        self.PartQlHatModp_pad[l][k][i][offset + j] = QHatModpj

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
            self.pHatInvModp[k] = int(
                self.invMod(int(self.pHatModp[k]), self.moduliP[k])
            )

        # 初始化 pHatModq
        self.pHatModq = [[0] * L for _ in range(K)]
        for k in range(K):
            for i in range(L):
                self.pHatModq[k][i] = int(1)
                for s in list(range(k)) + list(range(k + 1, K)):
                    temp = int(self.moduliP[s]) % int(self.moduliQ[i])
                    self.pHatModq[k][i] = self.mulMod(
                        int(self.pHatModq[k][i]), temp, int(self.moduliQ[i])
                    )

        self.PModq = [0] * L  # 初始化 PModq

        # 计算 PModq
        for i in range(L):
            self.PModq[i] = int(1)
            for k in range(K):
                temp = self.moduliP[k] % self.moduliQ[i]
                self.PModq[i] = self.mulMod(
                    int(self.PModq[i]), int(temp), int(self.moduliQ[i])
                )

        self.PInvModq = [0] * L  # 初始化 PInvModq
        # 计算 PInvModq
        for i in range(L):
            self.PInvModq[i] = self.invMod(int(self.PModq[i]), int(self.moduliQ[i]))

        self.qInvModq = [[0 for _ in range(L)] for _ in range(L)]
        for i in range(L):
            for j in list(range(i)) + list(range(i + 1, L)):
                self.qInvModq[i][j] = self.invMod(
                    int(self.moduliQ[i]), int(self.moduliQ[j])
                )

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
                    QlInvModql = self.mulMod(
                        int(QlInvModql), int(temp), int(self.moduliQ[l])
                    )

                modulusQ = int(1)
                for j in range(l):
                    modulusQ *= int(self.moduliQ[j])

                result = int((int(QlInvModql) * modulusQ) // int(self.moduliQ[l]))
                result %= int(self.moduliQ[i])

                self.QlQlInvModqlDivqlModq[k][i] = np.uint64(result)

        self.mult_swk = [None, None]
        if MULT_SWK is None:
            warnings.warn(
                "\n------------------------\n"
                "MULT_SWK needs to be set"
                "\n------------------------\n",
                UserWarning,
            )
            # todo: set data in numpy array
        else:
            self.mult_swk[0] = MULT_SWK[0]
            self.mult_swk[1] = MULT_SWK[1]

        self.moduliQ = np.array(self.moduliQ, dtype=np.uint64)
        self.moduliP = np.array(self.moduliP, dtype=np.uint64)
        self.qRoots = np.array(self.qRoots, dtype=np.uint64)
        self.pRoots = np.array(self.pRoots, dtype=np.uint64)

        self.pInvVec = np.array(self.pInvVec, dtype=np.uint64)
        self.qRootScalePows = np.array(self.qRootScalePows, dtype=np.uint64)
        self.pRootScalePows = np.array(self.pRootScalePows, dtype=np.uint64)
        self.qRootScalePowsInv = np.array(self.qRootScalePowsInv, dtype=np.uint64)
        self.pRootScalePowsInv = np.array(self.pRootScalePowsInv, dtype=np.uint64)
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
        self.QlQlInvModqlDivqlModq = np.array(
            self.QlQlInvModqlDivqlModq, dtype=np.uint64
        )

        #todo: scalingFactorsReal and scalingFactorsRealBig should be move to cuda?
        # note that they are vector of doubles in openfhe. now is set to float
        # todo: check if self.dmoduliQ needs to be moved to cuda
        DEFAULT_EXTRA_MOD_SIZE = 20
        extraBits = DEFAULT_EXTRA_MOD_SIZE if self.rescaleTech == "FLEXIBLEAUTOEXT" else 0
        # Pre-compute scaling factors for each level (used in FLEXIBLE* scaling techniques)
        if self.rescaleTech == "FLEXIBLEAUTO" or self.rescaleTech == "FLEXIBLEAUTOEXT":
            self.scalingFactorsReal = [0.0] * self.L
            if self.L == 1 and extraBits == 0:
                # mult depth = 0 and FLEXIBLEAUTO
                # when multiplicative depth = 0, we use the scaling mod size instead of modulus size
                # Plaintext modulus is used in EncodingParamsImpl to store the exponent p of the scaling factor
                self.scalingFactorsReal[0] = 2 ** self.logqi
            elif self.L == 2 and extraBits > 0:
                # mult depth = 0 and FLEXIBLEAUTOEXT
                # when multiplicative depth = 0, we use the scaling mod size instead of modulus size
                # Plaintext modulus is used in EncodingParamsImpl to store the exponent p of the scaling factor
                self.scalingFactorsReal[0] = float(self.moduliQ[self.L - 1])
                self.scalingFactorsReal[1] = 2 ** self.logqi
            else:
                self.scalingFactorsReal[0] = float(self.moduliQ[self.L - 1])
                if extraBits > 0:
                    self.scalingFactorsReal[1] = float(self.moduliQ[self.L - 2])
                lastPresetFactor = (
                    self.scalingFactorsReal[0]
                    if extraBits == 0
                    else self.scalingFactorsReal[1]
                )
                # number of levels with pre-calculated factors
                numPresetFactors = 1 if extraBits == 0 else 2

                for k in range(numPresetFactors, self.L):
                    prevSF = self.scalingFactorsReal[k - 1]
                    self.scalingFactorsReal[k] = (
                        prevSF * prevSF / float(self.moduliQ[self.L - k])
                    )
                    ratio = self.scalingFactorsReal[k] / lastPresetFactor
                    if ratio <= 0.5 or ratio >= 2.0:
                        print(
                            "FLEXIBLEAUTO cannot support this number of levels in this parameter setting. Please use FIXEDMANUAL or FIXEDAUTO instead."
                        )

            self.scalingFactorsRealBig = [0.0] * (self.L - 1)
            if len(self.scalingFactorsRealBig) > 0:
                if extraBits == 0:
                    self.scalingFactorsRealBig[0] = (
                        self.scalingFactorsReal[0] * self.scalingFactorsReal[0]
                    )
                else:
                    self.scalingFactorsRealBig[0] = (
                        self.scalingFactorsReal[0] * self.scalingFactorsReal[1]
                    )
                for k in range(1, self.L - 1):
                    self.scalingFactorsRealBig[k] = (
                        self.scalingFactorsReal[k] * self.scalingFactorsReal[k]
                    )
            # Moduli as real
            self.dmoduliQ = [0.0] * self.L
            for i in range(self.L):
                self.dmoduliQ[i] = float(self.moduliQ[i])
        else:
            self.approxSF = 2 ** self.logqi

        # for cuda context
        if torch.cuda.is_available():
            self.max_num_moduli = self.L + self.K
            self.chain_length = self.L
            self.num_special_moduli = self.K
            self.primes = np.hstack((moduliQ, moduliP))

            self.power_of_roots = None
            self.power_of_roots_shoup = None
            self.inverse_power_of_roots_div_two = None
            self.inverse_scaled_power_of_roots_div_two = None
            self.power_of_roots_vec = []
            self.power_of_roots_shoup_vec = []
            self.inv_power_of_roots_vec = []
            self.inv_power_of_roots_shoup_vec = []
            self.barret_k = []
            self.barret_ratio = []

            # for modup
            self.num_moduli_after_modup = self.max_num_moduli
            self.hat_inverse_vec_modup = None
            self.hat_inverse_vec_shoup_modup = None
            self.prod_q_i_mod_q_j_modup = None

            # for moddown
            self.num_moduli_after_moddown = self.chain_length
            self.hat_inverse_vec_moddown = []
            self.hat_inverse_vec_shoup_moddown = []
            self.prod_q_i_mod_q_j_moddown = []
            self.prod_inv_moddown = []
            self.prod_inv_shoup_moddown = []

            # for drop_last_element_and_scale
            self.qlql_inv_mod_ql_div_ql_mod_q = None
            self.qlql_inv_mod_ql_div_ql_mod_q_shoup = None
            self.q_inv_mod_q = None
            self.q_inv_mod_q_shoup = None

            self.swk_bx_cuda = torch.tensor(
                self.mult_swk[0].reshape(-1),dtype=torch.uint64, device="cuda")
            self.swk_ax_cuda = torch.tensor(
                self.mult_swk[1].reshape(-1),dtype=torch.uint64, device="cuda")
            
            # for output & workspace
            self.beta = (int)(self.L / self.K)
            self.inner_workspace = torch.tensor(
                [0] * (4 * self.num_moduli_after_modup * self.N * self.beta),
                dtype=torch.uint64,
                device="cuda",
            )
            self.inner_out = torch.tensor(
                [0] * (2 * self.num_moduli_after_modup * self.N),
                dtype=torch.uint64,
                device="cuda",
            )
            self.moddown_out_ax = torch.tensor(
                [0] * (self.num_moduli_after_moddown * self.N),
                dtype=torch.uint64,
                device="cuda",
            )
            self.moddown_out_bx = torch.tensor(
                [0] * (self.num_moduli_after_moddown * self.N),
                dtype=torch.uint64,
                device="cuda",
            )
            self.modup_out = torch.tensor(
                [0] * (self.num_moduli_after_modup * self.N * self.beta),
                dtype=torch.uint64,
                device="cuda",
            )
            self.rescale_out = torch.tensor(
                [0] * ((self.L - 1) * self.N),
                dtype=torch.uint64,
                device="cuda",
            )
            self.automorphism_transform_out = torch.tensor(
                [0] * (self.num_moduli_after_modup * self.N * self.beta),
                dtype=torch.uint64,
                device="cuda",
            )
            self.switch_modulus_out = torch.tensor(
                [0] * (self.num_moduli_after_modup * self.N * self.beta),
                dtype=torch.uint64,
                device="cuda",
            )

            power_of_roots = self.qRootPows + self.pRootPows
            inverse_power_of_roots = self.qRootPowsInv + self.pRootPowsInv
            # cal basic param
            for i, prime in enumerate(self.primes):
                barret = math.floor(math.log2(prime)) + 63
                self.barret_k.append(barret)

                temp = 1 << (barret - 64)
                temp <<= 64
                self.barret_ratio.append(int(temp) // int(prime))
                power_of_roots_shoup = self.shoup_each(power_of_roots[i], prime)
                inv_power_of_roots_div_two = self.div_two(
                    inverse_power_of_roots[i], prime
                )
                inv_power_of_roots_shoup = self.shoup_each(
                    inv_power_of_roots_div_two, prime
                )

                self.power_of_roots_vec.extend(power_of_roots[i])
                self.power_of_roots_shoup_vec.extend(power_of_roots_shoup)
                self.inv_power_of_roots_vec.extend(inv_power_of_roots_div_two)
                self.inv_power_of_roots_shoup_vec.extend(inv_power_of_roots_shoup)

            self.barret_k = torch.tensor(
                self.barret_k, dtype=torch.uint64, device="cuda"
            )
            self.barret_ratio = torch.tensor(
                self.barret_ratio, dtype=torch.uint64, device="cuda"
            )

            self.power_of_roots = torch.tensor(
                self.power_of_roots_vec, dtype=torch.uint64, device="cuda"
            )
            self.power_of_roots_shoup = torch.tensor(
                self.power_of_roots_shoup_vec, dtype=torch.uint64, device="cuda"
            )
            self.inverse_power_of_roots_div_two = torch.tensor(
                self.inv_power_of_roots_vec, dtype=torch.uint64, device="cuda"
            )
            self.inverse_scaled_power_of_roots_div_two = torch.tensor(
                self.inv_power_of_roots_shoup_vec, dtype=torch.uint64, device="cuda"
            )

            # cal modup param
            prod_q_i_mod_q_j_modup = []
            for l in range(self.L):
                prod_qi_mod_qj = []
                for dnum_idx in range(self.dnum):
                    prod_q_i_mod_q_j = self.PartQlHatModp[l][dnum_idx]
                    prod_q_i_mod_q_j = prod_q_i_mod_q_j.swapaxes(1, 0).flatten()
                    prod_qi_mod_qj.append(prod_q_i_mod_q_j)
                prod_q_i_mod_q_j_modup.append(prod_qi_mod_qj)
            self.prod_q_i_mod_q_j_modup = torch.tensor(
                np.array(prod_q_i_mod_q_j_modup, dtype=np.uint64),
                dtype=torch.uint64,
                device="cuda",
            )

            hat_inverse_vec_modup = []
            hat_inverse_vec_shoup_modup = []
            for dnum_idx in range(self.dnum):
                for k in range(self.K):
                    hat_inv_shoup = []
                    hat_inverse_vec = self.PartQlHatInvModq[dnum_idx][k]
                    hat_inverse_vec_modup.append(hat_inverse_vec)
                    for k_idx in range(self.K):
                        prime_idx = dnum_idx * self.K + k_idx
                        prime = self.primes[prime_idx]
                        shoup = self.shoup(int(hat_inverse_vec[k_idx]), prime)
                        hat_inv_shoup.append(shoup)
                    hat_inverse_vec_shoup_modup.append(hat_inv_shoup)
            self.hat_inverse_vec_modup = torch.tensor(
                np.array(hat_inverse_vec_modup, dtype=np.uint64),
                dtype=torch.uint64,
                device="cuda",
            )
            self.hat_inverse_vec_shoup_modup = torch.tensor(
                np.array(hat_inverse_vec_shoup_modup, dtype=np.uint64),
                dtype=torch.uint64,
                device="cuda",
            )

            # cal moddown param
            start_length = self.num_special_moduli
            end_length = self.chain_length
            start_begin = self.primes[end_length:]
            start_end = start_begin[start_length:]

            hat_inv_moddown = self.pHatInvModp
            hat_inv_shoup_moddown = []
            hat_inverse_vec_moddown = []
            hat_inverse_vec_shoup_moddown = []
            for k in range(self.K):
                prime = self.primes[self.L + k]
                shoup = self.shoup(int(hat_inv_moddown[k]), prime)
                hat_inv_shoup_moddown.append(shoup)
            hat_inverse_vec_moddown.append(hat_inv_moddown)
            self.hat_inverse_vec_moddown = torch.tensor(
                np.array(hat_inverse_vec_moddown, dtype=np.uint64),
                dtype=torch.uint64,
                device="cuda",
            )
            hat_inverse_vec_shoup_moddown.append(hat_inv_shoup_moddown)
            self.hat_inverse_vec_shoup_moddown = torch.tensor(
                np.array(hat_inverse_vec_shoup_moddown, dtype=np.uint64),
                dtype=torch.uint64,
                device="cuda",
            )

            prod_q_i_mod_q_j_moddown = []
            end_primes = self.set_difference(self.primes, start_begin)
            prod_q_i_mod_q_j_moddown.append(self.pHatModq.swapaxes(1, 0).flatten())
            self.prod_q_i_mod_q_j_moddown = torch.tensor(
                np.array(prod_q_i_mod_q_j_moddown, dtype=np.uint64),
                dtype=torch.uint64,
                device="cuda",
            )

            prod_inv = self.PInvModq
            prod_shoup = []

            for i, end_prime in enumerate(end_primes):
                inv = prod_inv[i]
                prod_shoup.append(self.shoup(int(inv), end_prime))

            prod_inv_moddown = []
            prod_inv_moddown.append(prod_inv)
            self.prod_inv_moddown = torch.tensor(
                np.array(prod_inv_moddown, dtype=np.uint64),
                dtype=torch.uint64,
                device="cuda",
            )

            prod_inv_shoup_moddown = []
            prod_inv_shoup_moddown.append(prod_shoup)
            self.prod_inv_shoup_moddown = torch.tensor(
                np.array(prod_shoup, dtype=np.uint64), dtype=torch.uint64, device="cuda"
            )

            # cal rescale param
            QlQlInvModqlDivqlModq = self.QlQlInvModqlDivqlModq.reshape(-1)
            qlql_inv_mod_ql_div_ql_mod_q_vec = []
            qlql_inv_mod_ql_div_ql_mod_q_shoup_vec = []
            for i in range(self.L - 1):
                for j in range(self.L - 1):
                    QlQlInvModqlDivqlModq_i = QlQlInvModqlDivqlModq[
                        i * (self.L - 1) + j
                    ]
                    prime = self.primes[j]
                    shoup = self.shoup(int(QlQlInvModqlDivqlModq_i), prime)
                    qlql_inv_mod_ql_div_ql_mod_q_vec.append(QlQlInvModqlDivqlModq_i)
                    qlql_inv_mod_ql_div_ql_mod_q_shoup_vec.append(shoup)
            self.qlql_inv_mod_ql_div_ql_mod_q = torch.tensor(
                np.array(qlql_inv_mod_ql_div_ql_mod_q_vec, dtype=np.uint64),
                dtype=torch.uint64,
                device="cuda",
            )
            self.qlql_inv_mod_ql_div_ql_mod_q_shoup = torch.tensor(
                np.array(qlql_inv_mod_ql_div_ql_mod_q_shoup_vec, dtype=np.uint64),
                dtype=torch.uint64,
                device="cuda",
            )

            qInvModq = self.qInvModq.reshape(-1)
            qInvModq_vec = []
            qInvModq_shoup_vec = []
            for i in range(self.L):
                for j in range(self.L):
                    qInvModq_i = qInvModq[i * self.L + j]
                    prime = self.primes[j]
                    shoup = self.shoup(int(qInvModq_i), prime)
                    qInvModq_vec.append(qInvModq_i)
                    qInvModq_shoup_vec.append(shoup)
            self.q_inv_mod_q = torch.tensor(
                np.array(qInvModq_vec, dtype=np.uint64),
                dtype=torch.uint64,
                device="cuda",
            )
            self.q_inv_mod_q_shoup = torch.tensor(
                np.array(qInvModq_shoup_vec, dtype=np.uint64),
                dtype=torch.uint64,
                device="cuda",
            )
            self.PModq_cuda = torch.tensor(self.PModq, dtype=torch.uint64, device="cuda")

            self.primes = torch.tensor(self.primes, dtype=torch.uint64, device="cuda")


        swk_bx = MULT_SWK[0].reshape(self.dnum, L + K, self.N)
        swk_ax = MULT_SWK[1].reshape(self.dnum, L + K, self.N)

        #todo: move to bscontext in the future
        self.m_U0hatTPreFFT_mx = BOOT_KEY['C2S']
        self.m_U0PreFFT_mx = BOOT_KEY['S2C']
        self.m_U0hatTPreFFT_dim = BOOT_KEY['C2S_dim']
        self.m_U0PreFFT_dim = BOOT_KEY['S2C_dim']
        self.m_U0hatTPreFFT_limbs = BOOT_KEY['C2S_limbs']
        self.m_U0PreFFT_limbs = BOOT_KEY['S2C_limbs']
        self.m_U0hatTPreFFT_scaling_factor = BOOT_KEY['U0hatTPreFFTScalingFactor']
        self.m_U0PreFFT_scaling_factor = BOOT_KEY['U0PreFFTScalingFactor']

        key_map_ax_fixed = torch.tensor(swk_ax, dtype=torch.uint64, device="cuda")
        key_map_bx_fixed = torch.tensor(swk_bx, dtype=torch.uint64, device="cuda")
        self.key_map = [key_map_bx_fixed, key_map_ax_fixed]

        for i, bx, ax in ROT_SWK:
            self.left_rot_key_map[str(i)] = [torch.tensor(bx, dtype=torch.uint64, device="cuda").reshape(self.dnum, -1, self.N)
                                                    ,
                                                    torch.tensor(ax, dtype=torch.uint64, device="cuda").reshape(self.dnum, -1, self.N)]

   #  Method to retrieve the scaling factor of level l.
   #  For FIXEDMANUAL scaling technique method always returns 2^p, where p corresponds to plaintext modulus
   #  @param l For FLEXIBLEAUTO scaling technique the level whose scaling factor we want to learn.
   #  Levels start from 0 (no scaling done - all towers) and go up to K-1, where K is the number of towers supported.
   #  @return the scaling factor.
    def GetScalingFactorReal(self, cur_limbs= None): #todo: introduce level or transfer limbs to level inside
        if cur_limbs is None:
            cur_limbs = self.L
        lvl = self.L - cur_limbs # openfhe use `level` to do the index
        if self.rescaleTech == "FLEXIBLEAUTO" or self.rescaleTech == "FLEXIBLEAUTOEXT":
            if lvl >= len(self.scalingFactorsReal):
                # openfhetodo: Return an error here.
                return self.approxSF
            return self.scalingFactorsReal[lvl]
        return self.approxSF

    def GetScalingFactorRealBig(self, cur_limbs = None):
        if cur_limbs is None:
            cur_limbs = self.L
        l = self.L - cur_limbs
        if self.rescaleTech == "FLEXIBLEAUTO" or self.rescaleTech == "FLEXIBLEAUTOEXT":
            if l >= len(self.scalingFactorsRealBig):
                # openfhetodo: Return an error here.
                return self.approxSF
            return self.scalingFactorsRealBig[l]
        return self.approxSF

    # Method to retrieve the modulus to be dropped of level l.
    # For FIXEDMANUAL rescaling technique method always returns 2^p, where p corresponds to plaintext modulus
    # @param l index of modulus to be dropped for FLEXIBLEAUTO scaling technique
    # @return the precomputed table
    def GetModReduceFactor(self, cur_limbs = None):
        if cur_limbs is None:
            cur_limbs = 0
        # l = self.L - cur_limbs #todo: check the meaning of input in openfhe
        l = cur_limbs
        if self.rescaleTech == "FLEXIBLEAUTO" or self.rescaleTech == "FLEXIBLEAUTOEXT":
            return self.dmoduliQ[l]
        return self.approxSF

    def shoup(self, in_value, prime):
        temp = in_value << 64
        return int(int(temp) // int(prime))

    def shoup_each(self, values, prime):
        return [self.shoup(value, prime) for value in values]

    def div_two(self, in_list, prime):
        two_inv = self.invMod(2, prime)
        out_list = [self.mulMod(int(x), int(two_inv), int(prime)) for x in in_list]
        return out_list

    def set_difference(self, begin, end):
        remove_set = set(end)
        return [item for item in begin if item not in remove_set]

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
        UINT64_MAX = 0xFFFFFFFFFFFFFFFF
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
            reversed_bits |= n & 1
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
    def is_prime(self, p):
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

    def mod_exp(self, val, exp, modulus):
        return pow(int(val), int(exp), int(modulus))

    def root_of_unity(self, order, modulus):
        """Finds a root of unity in the given modulus.
        Finds a root of unity with the given order in the given prime modulus.
        Args:
            order (int): Order n of the root of unity (an nth root of unity).
            modulus (int): Modulus to find the root of unity in. Note: MUST BE
                PRIME
        Returns:
            A root of unity with the given order in the given modulus.
        """
        if ((modulus - 1) % order) != 0:
            raise ValueError('Must have order q | m - 1, where m is the modulus. \
                The values m = ' + str(modulus) + ' and q = ' + str(order) + ' do not satisfy this.')

        generator = sympy.ntheory.primitive_root(modulus)
        if generator is None:
            raise ValueError('No primitive root of unity mod m = ' + str(modulus))

        result = self.mod_exp(generator, (modulus - 1)//order, modulus)

        if result == 1:
            return self.root_of_unity(order, modulus)

        return result
    def method(self):  # function to initialize variables
        pass

    def Serialize(self):
        self.q_mu_cuda = self.q_mu_cuda.cpu().numpy()
        self.moduliQ_cuda = self.moduliQ_cuda.cpu().numpy()
        self.primes = self.primes.cpu().numpy()
        self.power_of_roots = self.power_of_roots.cpu().numpy()
        self.power_of_roots_shoup = self.power_of_roots_shoup.cpu().numpy()
        self.inverse_power_of_roots_div_two = self.inverse_power_of_roots_div_two.cpu().numpy()
        self.inverse_scaled_power_of_roots_div_two = self.inverse_scaled_power_of_roots_div_two.cpu().numpy()
        self.barret_k = self.barret_k.cpu().numpy()
        self.barret_ratio = self.barret_ratio.cpu().numpy()
        self.hat_inverse_vec_modup = self.hat_inverse_vec_modup.cpu().numpy()
        self.hat_inverse_vec_shoup_modup = self.hat_inverse_vec_shoup_modup.cpu().numpy()
        self.prod_q_i_mod_q_j_modup = self.prod_q_i_mod_q_j_modup.cpu().numpy()
        self.hat_inverse_vec_moddown = self.hat_inverse_vec_moddown.cpu().numpy()
        self.hat_inverse_vec_shoup_moddown = self.hat_inverse_vec_shoup_moddown.cpu().numpy()
        self.prod_q_i_mod_q_j_moddown = self.prod_q_i_mod_q_j_moddown.cpu().numpy()
        self.prod_inv_moddown = self.prod_inv_moddown.cpu().numpy()
        self.prod_inv_shoup_moddown = self.prod_inv_shoup_moddown.cpu().numpy()
        self.qlql_inv_mod_ql_div_ql_mod_q = self.qlql_inv_mod_ql_div_ql_mod_q.cpu().numpy()
        self.qlql_inv_mod_ql_div_ql_mod_q_shoup = self.qlql_inv_mod_ql_div_ql_mod_q_shoup.cpu().numpy()
        self.q_inv_mod_q = self.q_inv_mod_q.cpu().numpy()
        self.q_inv_mod_q_shoup = self.q_inv_mod_q_shoup.cpu().numpy()
        self.swk_bx_cuda = self.swk_bx_cuda.cpu().numpy()
        self.swk_ax_cuda = self.swk_ax_cuda.cpu().numpy()
        self.inner_workspace = self.inner_workspace.cpu().numpy()
        self.inner_out = self.inner_out.cpu().numpy()
        self.moddown_out_ax = self.moddown_out_ax.cpu().numpy()
        self.moddown_out_bx = self.moddown_out_bx.cpu().numpy()
        self.modup_out = self.modup_out.cpu().numpy()
        self.rescale_out = self.rescale_out.cpu().numpy()
        self.automorphism_transform_out = self.automorphism_transform_out.cpu().numpy()
        self.switch_modulus_out = self.switch_modulus_out.cpu().numpy()
        self.PModq_cuda = self.PModq_cuda.cpu().numpy()

        self.key_map = [v.cpu().numpy() for v in self.key_map]
        
        for key, value in self.left_rot_key_map.items():
            self.left_rot_key_map[key] = [v.cpu().numpy() for v in value]

        return pickle.dumps(self)
    

    def Deserialize(ctx_bytes):
        cryptoContext = pickle.loads(ctx_bytes)
        cryptoContext.q_mu_cuda = torch.tensor(cryptoContext.q_mu_cuda, dtype = torch.uint64, device = "cuda")
        cryptoContext.moduliQ_cuda = torch.tensor(cryptoContext.moduliQ_cuda, dtype = torch.uint64, device = "cuda")
        cryptoContext.primes = torch.tensor(cryptoContext.primes, dtype = torch.uint64, device = "cuda")
        cryptoContext.power_of_roots = torch.tensor(cryptoContext.power_of_roots, dtype = torch.uint64, device = "cuda")
        cryptoContext.power_of_roots_shoup = torch.tensor(cryptoContext.power_of_roots_shoup, dtype = torch.uint64, device = "cuda")
        cryptoContext.inverse_power_of_roots_div_two = torch.tensor(cryptoContext.inverse_power_of_roots_div_two, dtype = torch.uint64, device = "cuda")
        cryptoContext.inverse_scaled_power_of_roots_div_two = torch.tensor(cryptoContext.inverse_scaled_power_of_roots_div_two, dtype = torch.uint64, device = "cuda")
        cryptoContext.barret_k = torch.tensor(cryptoContext.barret_k, dtype = torch.uint64, device = "cuda")
        cryptoContext.barret_ratio = torch.tensor(cryptoContext.barret_ratio, dtype = torch.uint64, device = "cuda")
        cryptoContext.hat_inverse_vec_modup = torch.tensor(cryptoContext.hat_inverse_vec_modup, dtype = torch.uint64, device = "cuda")
        cryptoContext.hat_inverse_vec_shoup_modup = torch.tensor(cryptoContext.hat_inverse_vec_shoup_modup, dtype = torch.uint64, device = "cuda")
        cryptoContext.prod_q_i_mod_q_j_modup = torch.tensor(cryptoContext.prod_q_i_mod_q_j_modup, dtype = torch.uint64, device = "cuda")
        cryptoContext.hat_inverse_vec_moddown = torch.tensor(cryptoContext.hat_inverse_vec_moddown, dtype = torch.uint64, device = "cuda")
        cryptoContext.hat_inverse_vec_shoup_moddown = torch.tensor(cryptoContext.hat_inverse_vec_shoup_moddown, dtype = torch.uint64, device = "cuda")
        cryptoContext.prod_q_i_mod_q_j_moddown = torch.tensor(cryptoContext.prod_q_i_mod_q_j_moddown, dtype = torch.uint64, device = "cuda")
        cryptoContext.prod_inv_moddown = torch.tensor(cryptoContext.prod_inv_moddown, dtype = torch.uint64, device = "cuda")
        cryptoContext.prod_inv_shoup_moddown = torch.tensor(cryptoContext.prod_inv_shoup_moddown, dtype = torch.uint64, device = "cuda")
        cryptoContext.qlql_inv_mod_ql_div_ql_mod_q = torch.tensor(cryptoContext.qlql_inv_mod_ql_div_ql_mod_q, dtype = torch.uint64, device = "cuda")
        cryptoContext.qlql_inv_mod_ql_div_ql_mod_q_shoup = torch.tensor(cryptoContext.qlql_inv_mod_ql_div_ql_mod_q_shoup, dtype = torch.uint64, device = "cuda")
        cryptoContext.q_inv_mod_q = torch.tensor(cryptoContext.q_inv_mod_q, dtype = torch.uint64, device = "cuda")
        cryptoContext.q_inv_mod_q_shoup = torch.tensor(cryptoContext.q_inv_mod_q_shoup, dtype = torch.uint64, device = "cuda")
        cryptoContext.swk_bx_cuda = torch.tensor(cryptoContext.swk_bx_cuda, dtype = torch.uint64, device = "cuda")
        cryptoContext.swk_ax_cuda = torch.tensor(cryptoContext.swk_ax_cuda, dtype = torch.uint64, device = "cuda")
        cryptoContext.inner_workspace = torch.tensor(cryptoContext.inner_workspace, dtype = torch.uint64, device = "cuda")
        cryptoContext.inner_out = torch.tensor(cryptoContext.inner_out, dtype = torch.uint64, device = "cuda")
        cryptoContext.moddown_out_ax = torch.tensor(cryptoContext.moddown_out_ax, dtype = torch.uint64, device = "cuda")
        cryptoContext.moddown_out_bx = torch.tensor(cryptoContext.moddown_out_bx, dtype = torch.uint64, device = "cuda")
        cryptoContext.modup_out = torch.tensor(cryptoContext.modup_out, dtype = torch.uint64, device = "cuda")
        cryptoContext.rescale_out = torch.tensor(cryptoContext.rescale_out, dtype = torch.uint64, device = "cuda")
        cryptoContext.automorphism_transform_out = torch.tensor(cryptoContext.automorphism_transform_out, dtype = torch.uint64, device = "cuda")
        cryptoContext.switch_modulus_out = torch.tensor(cryptoContext.switch_modulus_out, dtype = torch.uint64, device = "cuda")
        cryptoContext.PModq_cuda = torch.tensor(cryptoContext.PModq_cuda, dtype = torch.uint64, device = "cuda")

        cryptoContext.key_map = [torch.tensor(v, dtype = torch.uint64, device = "cuda") for v in cryptoContext.key_map]

        for key, value in cryptoContext.left_rot_key_map.items():
            cryptoContext.left_rot_key_map[key] = [torch.tensor(v, dtype = torch.uint64, device = "cuda") for v in value]

        return cryptoContext
