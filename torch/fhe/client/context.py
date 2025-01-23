import numpy as np
import math
import random
import time
import warnings
import pickle
import sympy
from .bs_context import *

class __FOR_SAVE_ONLY_Context:
    def __init__(
        self,
        logN,
        logSlots_list,
        firstMod,  # todo: rename to firstMod
        dcrtBits,  # todo: rename to dcrtBits
        specialMod,  # todo: rename to specialMod
        L,
        K,
        levelBudget_list,
        moduliQ=None,
        moduliP=None,
        rootsQ=None,
        rootsP=None,
        MULT_SWK=None,
        rot_swk_map=None,
        boot_key_map=None,
        secretKeyDist=None,
        rescaleTech=None,
        dim1=None,
        h=64,
        sigma=32,
    ):
        # levelBudget_list = levelBudget_list
        self.logSlots_list = logSlots_list
        self.secretKeyDist = secretKeyDist
        self.rescaleTech = rescaleTech
        # self.BsContext = None
        self.BsContext_map = {}
        self.specialMod = specialMod
        # self.slots = 1 << logSlots #todo: need move slots to cipher
        self.qVec = None
        # self.left_rot_key_map = {}
        self.slots_left_rot_key_map = {}
        self.key_map = None
        self.correctionFactor = 0

        self.logN = logN
        self.dcrtBits = dcrtBits
        self.L = int(L)
        self.K = int(K)
        self.dnum = math.ceil(L / K)
        self.h = h
        self.sigma = sigma
        self.N = int(1 << logN)
        self.M = self.N << 1
        self.logNh = logN - 1
        self.Nh = self.N >> 1
        self.p = 1 << dcrtBits #todo: to be removed?

        self.moduliQ = [0] * L
        qRoots = [0] * L
        qRootsInv = [0] * L
        qRootPows = [[] for _ in range(L)]
        # self.qRootScalePows = [[] for _ in range(L)]
        # self.qRootScalePowsOverq = [[] for _ in range(L)]
        # self.qRootScalePowsInv = [[] for _ in range(L)]
        qRootPowsInv = [[] for _ in range(L)]
        # self.auto_index = {} #todo: to suppor negative input?
        self.slots_precompute_auto_map = {}
        bnd = 1
        cnt = 1
        if moduliQ is None and rootsQ is None:
            while True:
                prime = (1 << firstMod) + bnd * self.M + 1
                if self.is_prime(prime):
                    self.moduliQ[0] = prime
                    break
                bnd += 1
            qRoots[0] = self.root_of_unity(order=self.M, modulus=self.moduliQ[0])
            bnd = 1
            while cnt < L:
                prime1 = (1 << dcrtBits) + bnd * self.M + 1
                if self.is_prime(prime1):
                    self.moduliQ[cnt] = prime1
                    cnt += 1
                prime2 = (1 << dcrtBits) - bnd * self.M + 1
                if self.is_prime(prime2):
                    self.moduliQ[cnt] = prime2
                    qRoots[cnt] = self.root_of_unity(
                        order=self.M, modulus=self.moduliQ[cnt - 1]
                    )
                    cnt += 1
                bnd += 1

            if dcrtBits - logN - 1 - math.ceil(math.log2(bnd)) < 10:
                print("ERROR: too small number of precision")
                print("TRY to use larger dcrtBits or smaller depth")
        else:
            if moduliQ is None:
                print("moduliQ needs to be set!")
                return
            elif rootsQ is None:
                print("rootsQ needs to be set!")
                return
            for i in range(L):
                self.moduliQ[i] = moduliQ[i]
                qRoots[i] = rootsQ[i]

        time0 = time.time()
        for i in range(L):
            qRootsInv[i] = self.invMod(qRoots[i], int(self.moduliQ[i]))
            qRootPows[i] = [0] * self.N
            qRootPowsInv[i] = [0] * self.N
            power = int(1)
            powerInv = int(1)
            for j in range(self.N):
                jprime = self.bitReverse(j) >> (32 - self.logN)
                qRootPows[i][jprime] = int(power)
                qRootPowsInv[i][jprime] = int(powerInv)
                if j < self.N - 1:
                    power = self.mulMod(
                        int(power), int(qRoots[i]), int(self.moduliQ[i])
                    )
                    powerInv = self.mulMod(
                        powerInv, int(qRootsInv[i]), int(self.moduliQ[i])
                    )
        q_mu = []  # for barret mul mod
        
        time1 = time.time()
        print("Inner time1: ", time1 - time0)

        for mod in self.moduliQ:
            x = 2**128 // int(mod)
            low = x & ((1 << 64) - 1)
            high = x >> 64
            q_mu.append([low, high])
        self.q_mu = np.array(q_mu, dtype=np.uint64)
        self.q_mu_cuda = np.array(q_mu, dtype=np.uint64)
        self.moduliQ_cuda = np.array(self.moduliQ, dtype=np.uint64)

        self.moduliP = [0] * self.K
        # self.pInvVec = [0] * self.K
        pRoots = [0] * self.K
        pRootsInv = [0] * self.K
        pRootPows = [[] for _ in range(self.K)]
        pRootPowsInv = [[] for _ in range(self.K)]
        # pRootScalePows = [[] for _ in range(self.K)]
        # self.pRootScalePowsOverp = [[] for _ in range(self.K)]
        # self.pRootScalePowsInv = [[] for _ in range(self.K)]

        if moduliP is None and rootsP is None:
            cnt = 0
            while cnt < self.K:
                prime1 = (1 << specialMod) + bnd * self.M + 1
                if self.is_prime(prime1):
                    self.moduliP[cnt] = prime1
                    pRoots[cnt] = self.root_of_unity(
                        order=self.M, modulus=self.moduliP[cnt]
                    )
                    cnt += 1
                if cnt == self.K:
                    break
                prime2 = (1 << specialMod) - bnd * self.M + 1
                if self.is_prime(prime2):
                    self.moduliP[cnt] = prime2
                    pRoots[cnt] = self.root_of_unity(
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
                pRoots[i] = rootsP[i]

        for i in range(K):
            pRootsInv[i] = self.invMod(pRoots[i], int(self.moduliP[i]))
            # self.pInvVec[i] = self.inv(self.moduliP[i])
            pRootPows[i] = [0] * self.N
            pRootPowsInv[i] = [0] * self.N
            # self.pRootScalePows[i] = [0] * self.N
            # self.pRootScalePowsOverp[i] = [0] * self.N
            # self.pRootScalePowsInv[i] = [0] * self.N
            power = int(1)
            powerInv = int(1)
            for j in range(self.N):
                jprime = self.bitReverse(j) >> (32 - self.logN)
                pRootPows[i][jprime] = int(power)
                tmp = int(power) << 64
                # self.pRootScalePowsOverp[i][jprime] = tmp // int(self.moduliP[i])
                # self.pRootScalePows[i][jprime] = self.mulMod(
                #     pRootPows[i][jprime], int(1 << 32), int(self.moduliP[i])
                # )
                # self.pRootScalePows[i][jprime] = self.mulMod(
                #     self.pRootScalePows[i][jprime], int(1 << 32), int(self.moduliP[i])
                # )
                pRootPowsInv[i][jprime] = powerInv
                # self.pRootScalePowsInv[i][jprime] = self.mulMod(
                #     pRootPowsInv[i][jprime], int(1 << 32), int(self.moduliP[i])
                # )
                # self.pRootScalePowsInv[i][jprime] = self.mulMod(
                #     self.pRootScalePowsInv[i][jprime],
                #     int(1 << 32),
                #     int(self.moduliP[i]),
                # )
                if j < self.N - 1:
                    power = self.mulMod(
                        power, int(pRoots[i]), int(self.moduliP[i])
                    )
                    powerInv = self.mulMod(
                        powerInv, int(pRootsInv[i]), int(self.moduliP[i])
                    )

        time2 = time.time()
        print("Inner time2: ", time2 - time1)

        p_mu = []  # for barret mul mod
        for mod in self.moduliP:
            x = 2**128 // int(mod)
            low = x & ((1 << 64) - 1)
            high = x >> 64
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

        time3 = time.time()
        print("Inner time3: ", time3 - time2)
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

        time4 = time.time()
        print("Inner time4: ", time4 - time3)
        self.pHatModp = [0] * K
        self.pHatInvModp = [0] * K
        # 计算 pHatModp
        for k in range(K):
            self.pHatModp[k] = int(1)
            for j in list(range(k)) + list(range(k + 1, K)):
                temp = int(self.moduliP[j] % self.moduliP[k])
                self.pHatModp[k] = (self.pHatModp[k] * temp) % int(self.moduliP[k])

        for k in range(K):
            self.pHatInvModp[k] = int(
                self.invMod(int(self.pHatModp[k]), self.moduliP[k])
            )

        self.pHatModq = [[0] * L for _ in range(K)]
        for k in range(K):
            for i in range(L):
                self.pHatModq[k][i] = int(1)
                for s in list(range(k)) + list(range(k + 1, K)):
                    temp = int(self.moduliP[s]) % int(self.moduliQ[i])
                    self.pHatModq[k][i] = self.mulMod(
                        int(self.pHatModq[k][i]), temp, int(self.moduliQ[i])
                    )

        self.PModq = [0] * L

        # 计算 PModq
        for i in range(L):
            self.PModq[i] = int(1)
            for k in range(K):
                temp = self.moduliP[k] % self.moduliQ[i]
                self.PModq[i] = self.mulMod(
                    int(self.PModq[i]), int(temp), int(self.moduliQ[i])
                )

        self.PInvModq = [0] * L
        # 计算 PInvModq
        for i in range(L):
            self.PInvModq[i] = self.invMod(int(self.PModq[i]), int(self.moduliQ[i]))

        time5 = time.time()
        print("Inner time5: ", time5 - time4)
        qInvModq = [[0 for _ in range(L)] for _ in range(L)]
        for i in range(L):
            for j in list(range(i)) + list(range(i + 1, L)):
                qInvModq[i][j] = self.invMod(
                    int(self.moduliQ[i]), int(self.moduliQ[j])
                )

        # rescale param
        # sizeQ in openFHE equals to L here.
        self.QlQlInvModqlDivqlModq = [[0] * (L - 1) for _ in range(L - 1)]
        for k in range(L - 1):
            l = L - (k + 1)
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

        time6 = time.time()
        print("Inner time6: ", time6 - time5)

        self.moduliQ = np.array(self.moduliQ, dtype=np.uint64)
        self.moduliP = np.array(self.moduliP, dtype=np.uint64)
        qRoots = np.array(qRoots, dtype=np.uint64)
        pRoots = np.array(pRoots, dtype=np.uint64)

        # # self.pInvVec = np.array(self.pInvVec, dtype=np.uint64)
        # # self.qRootScalePows = np.array(self.qRootScalePows, dtype=np.uint64)
        # self.pRootScalePows = np.array(self.pRootScalePows, dtype=np.uint64)
        # self.qRootScalePowsInv = np.array(self.qRootScalePowsInv, dtype=np.uint64)
        # self.pRootScalePowsInv = np.array(self.pRootScalePowsInv, dtype=np.uint64)
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
        qInvModq = np.array(qInvModq, dtype=np.uint64)
        self.QlQlInvModqlDivqlModq = np.array(
            self.QlQlInvModqlDivqlModq, dtype=np.uint64
        )

        # todo: scalingFactorsReal and scalingFactorsRealBig should be move to cuda?
        # note that they are vector of doubles in openfhe. now is set to float
        # todo: check if self.dmoduliQ needs to be moved to cuda
        DEFAULT_EXTRA_MOD_SIZE = 20
        extraBits = (
            DEFAULT_EXTRA_MOD_SIZE if self.rescaleTech == "FLEXIBLEAUTOEXT" else 0
        )
        # Pre-compute scaling factors for each level (used in FLEXIBLE* scaling techniques)
        if self.rescaleTech == "FLEXIBLEAUTO" or self.rescaleTech == "FLEXIBLEAUTOEXT":
            self.scalingFactorsReal = [0.0] * self.L
            if self.L == 1 and extraBits == 0:
                # mult depth = 0 and FLEXIBLEAUTO
                # when multiplicative depth = 0, we use the scaling mod size instead of modulus size
                # Plaintext modulus is used in EncodingParamsImpl to store the exponent p of the scaling factor
                self.scalingFactorsReal[0] = 2**self.dcrtBits
            elif self.L == 2 and extraBits > 0:
                # mult depth = 0 and FLEXIBLEAUTOEXT
                # when multiplicative depth = 0, we use the scaling mod size instead of modulus size
                # Plaintext modulus is used in EncodingParamsImpl to store the exponent p of the scaling factor
                self.scalingFactorsReal[0] = float(self.moduliQ[self.L - 1])
                self.scalingFactorsReal[1] = 2**self.dcrtBits
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
            self.approxSF = 2**self.dcrtBits

        time7 = time.time()
        print("Inner time7: ", time7 - time6)
        # for cuda context
        if True:
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
            inv_power_of_roots_vec = []
            inv_power_of_roots_shoup_vec = []
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

            self.swk_bx_cuda = np.array(self.mult_swk[0].reshape(-1), dtype=np.uint64)
            self.swk_ax_cuda = np.array(self.mult_swk[1].reshape(-1), dtype=np.uint64)

            # for output & workspace
            self.beta = (int)(self.L / self.K)
            self.inner_workspace = np.array(
                [0] * (4 * self.num_moduli_after_modup * self.N * self.beta),
                dtype=np.uint64,
            )
            self.inner_out = np.array(
                [0] * (2 * self.num_moduli_after_modup * self.N),
                dtype=np.uint64,
            )
            self.moddown_out_ax = np.array(
                [0] * (self.num_moduli_after_moddown * self.N),
                dtype=np.uint64,
            )
            self.moddown_out_bx = np.array(
                [0] * (self.num_moduli_after_moddown * self.N),
                dtype=np.uint64,
            )
            self.modup_out = np.array(
                [0] * (self.num_moduli_after_modup * self.N * self.beta),
                dtype=np.uint64,
            )
            self.rescale_out = np.array(
                [0] * ((self.L - 1) * self.N),
                dtype=np.uint64,
            )
            self.automorphism_transform_out = np.array(
                [0] * (self.num_moduli_after_modup * self.N * self.beta),
                dtype=np.uint64,
            )
            self.switch_modulus_out = np.array(
                [0] * (self.num_moduli_after_modup * self.N * self.beta),
                dtype=np.uint64,
            )

            power_of_roots = qRootPows + pRootPows
            inverse_power_of_roots = qRootPowsInv + pRootPowsInv
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
                inv_power_of_roots_vec.extend(inv_power_of_roots_div_two)
                inv_power_of_roots_shoup_vec.extend(inv_power_of_roots_shoup)

            self.barret_k = np.array(self.barret_k, dtype=np.uint64)
            self.barret_ratio = np.array(self.barret_ratio, dtype=np.uint64)

            self.power_of_roots = np.array(self.power_of_roots_vec, dtype=np.uint64)
            self.power_of_roots_shoup = np.array(
                self.power_of_roots_shoup_vec, dtype=np.uint64
            )
            self.inverse_power_of_roots_div_two = np.array(
                inv_power_of_roots_vec, dtype=np.uint64
            )
            self.inverse_scaled_power_of_roots_div_two = np.array(
                inv_power_of_roots_shoup_vec, dtype=np.uint64
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
            self.prod_q_i_mod_q_j_modup = np.array(
                np.array(prod_q_i_mod_q_j_modup, dtype=np.uint64),
                dtype=np.uint64,
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
            self.hat_inverse_vec_modup = np.array(
                np.array(hat_inverse_vec_modup, dtype=np.uint64),
                dtype=np.uint64,
            )
            self.hat_inverse_vec_shoup_modup = np.array(
                np.array(hat_inverse_vec_shoup_modup, dtype=np.uint64),
                dtype=np.uint64,
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
            self.hat_inverse_vec_moddown = np.array(
                np.array(hat_inverse_vec_moddown, dtype=np.uint64),
                dtype=np.uint64,
            )
            hat_inverse_vec_shoup_moddown.append(hat_inv_shoup_moddown)
            self.hat_inverse_vec_shoup_moddown = np.array(
                np.array(hat_inverse_vec_shoup_moddown, dtype=np.uint64),
                dtype=np.uint64,
            )

            prod_q_i_mod_q_j_moddown = []
            end_primes = self.set_difference(self.primes, start_begin)
            prod_q_i_mod_q_j_moddown.append(self.pHatModq.swapaxes(1, 0).flatten())
            self.prod_q_i_mod_q_j_moddown = np.array(
                np.array(prod_q_i_mod_q_j_moddown, dtype=np.uint64),
                dtype=np.uint64,
            )

            prod_inv = self.PInvModq
            prod_shoup = []

            for i, end_prime in enumerate(end_primes):
                inv = prod_inv[i]
                prod_shoup.append(self.shoup(int(inv), end_prime))

            prod_inv_moddown = []
            prod_inv_moddown.append(prod_inv)
            self.prod_inv_moddown = np.array(
                np.array(prod_inv_moddown, dtype=np.uint64),
                dtype=np.uint64,
            )

            prod_inv_shoup_moddown = []
            prod_inv_shoup_moddown.append(prod_shoup)
            self.prod_inv_shoup_moddown = np.array(
                np.array(prod_shoup, dtype=np.uint64), dtype=np.uint64
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
            self.qlql_inv_mod_ql_div_ql_mod_q = np.array(
                np.array(qlql_inv_mod_ql_div_ql_mod_q_vec, dtype=np.uint64),
                dtype=np.uint64,
            )
            self.qlql_inv_mod_ql_div_ql_mod_q_shoup = np.array(
                np.array(qlql_inv_mod_ql_div_ql_mod_q_shoup_vec, dtype=np.uint64),
                dtype=np.uint64,
            )

            qInvModq = qInvModq.reshape(-1)
            qInvModq_vec = []
            qInvModq_shoup_vec = []
            for i in range(self.L):
                for j in range(self.L):
                    qInvModq_i = qInvModq[i * self.L + j]
                    prime = self.primes[j]
                    shoup = self.shoup(int(qInvModq_i), prime)
                    qInvModq_vec.append(qInvModq_i)
                    qInvModq_shoup_vec.append(shoup)
            self.q_inv_mod_q = np.array(
                np.array(qInvModq_vec, dtype=np.uint64),
                dtype=np.uint64,
            )
            self.q_inv_mod_q_shoup = np.array(
                np.array(qInvModq_shoup_vec, dtype=np.uint64),
                dtype=np.uint64,
            )
            self.PModq_cuda = np.array(self.PModq, dtype=np.uint64)

            self.primes = np.array(self.primes, dtype=np.uint64)

        time8 = time.time()
        print("Inner time8: ", time8 - time7)

        swk_bx = MULT_SWK[0].reshape(self.dnum, L + K, self.N)
        swk_ax = MULT_SWK[1].reshape(self.dnum, L + K, self.N)
        key_map_ax_fixed = np.array(swk_ax, dtype=np.uint64)
        key_map_bx_fixed = np.array(swk_bx, dtype=np.uint64)
        self.key_map = [key_map_bx_fixed, key_map_ax_fixed]

        for log_slots, ROT_SWK in rot_swk_map.items():
            left_rot_key_map = {}
            precompute_auto_map = {}
            for i, bx, ax in ROT_SWK:
                left_rot_key_map[str(i)] = [
                    np.array(bx, dtype=np.uint64).reshape(self.dnum, -1, self.N),
                    np.array(ax, dtype=np.uint64).reshape(self.dnum, -1, self.N),
                ]
            for key, _ in left_rot_key_map.items():
                precompute_auto_map[int(key)] = self.compute_auto_map(
                    int(key), self.N
                )
            self.slots_left_rot_key_map[log_slots] = left_rot_key_map
            self.slots_precompute_auto_map[log_slots] = precompute_auto_map

        # init bs_context
        for logSlots, levelBudget in zip(self.logSlots_list, levelBudget_list):
            self.BsContext_map[str(logSlots)] = BsContext(
                self.N,
                self.K,
                self.moduliQ,
                self.moduliP,
                self.q_mu,
                self.p_mu,
                levelBudget,
                dim1,
                (1 << logSlots),
                0,
                self.rescaleTech,
                self.secretKeyDist,
                boot_key_map[str(logSlots)]
            )

        time9 = time.time()
        print("Inner time9: ", time9 - time8)
        # compute auto index map
        # slots = 1 << logSlots
        # self.auto_index[slots] = self.find_auto_index(slots, self.N << 1)
        # for step in range(int(math.log2(self.N // (2 * slots)))):
        #     self.auto_index[(1 << step) * slots] = self.find_auto_index(
        #         (1 << step) * slots, self.N << 1)
        # for i in self.BsContext.C2S_rot_in + self.BsContext.C2S_rot_out + self.BsContext.S2C_rot_in + self.BsContext.S2C_rot_out:
        #     for j in i:
        #         if j not in self.auto_index:
        #             self.auto_index[j] = self.find_auto_index(j, self.N << 1)

    def compute_auto_map(self, k, N):
        def reverse_bits(num, num_bits):
            """Reverses the bits of a number."""
            rev = 0
            for i in range(num_bits):
                rev = (rev << 1) | (num & 1)
                num >>= 1
            return rev

        """computes the automorphism map"""
        n = N
        m = n << 1  # cyclOrder
        logm = round(np.log2(m))
        logn = round(np.log2(n))
        res = np.zeros(n, dtype=np.int32)
        for j in range(n):
            j_tmp = (j << 1) + 1
            idx = ((j_tmp * k) - (((j_tmp * k) >> logm) << logm)) >> 1
            j_rev = reverse_bits(j, logn)
            idx_rev = reverse_bits(idx, logn)
            res[j_rev] = idx_rev

        return np.array(res)

    def find_auto_index(self, i):
        def inv_mod(
            a, m
        ):  # note: check all the output value before merge with func: invMod!! These two values may differ by m!!
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

        m = self.N << 1

        if i == 0:
            return 1

        # Conjugation automorphism
        # if i == m - 1:
        #     return i
        if i == -1:
            return m - 1

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

    #  Method to retrieve the scaling factor of level l.
    #  For FIXEDMANUAL scaling technique method always returns 2^p, where p corresponds to plaintext modulus
    #  @param l For FLEXIBLEAUTO scaling technique the level whose scaling factor we want to learn.
    #  Levels start from 0 (no scaling done - all towers) and go up to K-1, where K is the number of towers supported.
    #  @return the scaling factor.
    def GetScalingFactorReal(
        self, cur_limbs=None
    ):  # todo: introduce level or transfer limbs to level inside
        if cur_limbs is None:
            cur_limbs = self.L
        lvl = self.L - cur_limbs  # openfhe use `level` to do the index
        if self.rescaleTech == "FLEXIBLEAUTO" or self.rescaleTech == "FLEXIBLEAUTOEXT":
            if lvl >= len(self.scalingFactorsReal):
                # openfhetodo: Return an error here.
                return self.approxSF
            return self.scalingFactorsReal[lvl]
        return self.approxSF

    def GetScalingFactorRealBig(self, cur_limbs=None):
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
    def GetModReduceFactor(self, cur_limbs=None):
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
            raise ValueError(
                "Must have order q | m - 1, where m is the modulus. \
                The values m = "
                + str(modulus)
                + " and q = "
                + str(order)
                + " do not satisfy this."
            )

        generator = sympy.ntheory.primitive_root(modulus)
        if generator is None:
            raise ValueError("No primitive root of unity mod m = " + str(modulus))

        result = self.mod_exp(generator, (modulus - 1) // order, modulus)

        if result == 1:
            return self.root_of_unity(order, modulus)

        return result

    def method(self):  # function to initialize variables
        pass

    def Serialize(self):
        return pickle.dumps(self)

    def Deserialize(ctx_bytes):
        cryptoContext = pickle.loads(ctx_bytes)
        # cryptoContext.q_mu_cuda = np.array(cryptoContext.q_mu_cuda, dtype = np.uint64, device = "cuda")
        # cryptoContext.moduliQ_cuda = np.array(cryptoContext.moduliQ_cuda, dtype = np.uint64, device = "cuda")
        # cryptoContext.primes = np.array(cryptoContext.primes, dtype = np.uint64, device = "cuda")
        # cryptoContext.power_of_roots = np.array(cryptoContext.power_of_roots, dtype = np.uint64, device = "cuda")
        # cryptoContext.power_of_roots_shoup = np.array(cryptoContext.power_of_roots_shoup, dtype = np.uint64, device = "cuda")
        # cryptoContext.inverse_power_of_roots_div_two = np.array(cryptoContext.inverse_power_of_roots_div_two, dtype = np.uint64, device = "cuda")
        # cryptoContext.inverse_scaled_power_of_roots_div_two = np.array(cryptoContext.inverse_scaled_power_of_roots_div_two, dtype = np.uint64, device = "cuda")
        # cryptoContext.barret_k = np.array(cryptoContext.barret_k, dtype = np.uint64, device = "cuda")
        # cryptoContext.barret_ratio = np.array(cryptoContext.barret_ratio, dtype = np.uint64, device = "cuda")
        # cryptoContext.hat_inverse_vec_modup = np.array(cryptoContext.hat_inverse_vec_modup, dtype = np.uint64, device = "cuda")
        # cryptoContext.hat_inverse_vec_shoup_modup = np.array(cryptoContext.hat_inverse_vec_shoup_modup, dtype = np.uint64, device = "cuda")
        # cryptoContext.prod_q_i_mod_q_j_modup = np.array(cryptoContext.prod_q_i_mod_q_j_modup, dtype = np.uint64, device = "cuda")
        # cryptoContext.hat_inverse_vec_moddown = np.array(cryptoContext.hat_inverse_vec_moddown, dtype = np.uint64, device = "cuda")
        # cryptoContext.hat_inverse_vec_shoup_moddown = np.array(cryptoContext.hat_inverse_vec_shoup_moddown, dtype = np.uint64, device = "cuda")
        # cryptoContext.prod_q_i_mod_q_j_moddown = np.array(cryptoContext.prod_q_i_mod_q_j_moddown, dtype = np.uint64, device = "cuda")
        # cryptoContext.prod_inv_moddown = np.array(cryptoContext.prod_inv_moddown, dtype = np.uint64, device = "cuda")
        # cryptoContext.prod_inv_shoup_moddown = np.array(cryptoContext.prod_inv_shoup_moddown, dtype = np.uint64, device = "cuda")
        # cryptoContext.qlql_inv_mod_ql_div_ql_mod_q = np.array(cryptoContext.qlql_inv_mod_ql_div_ql_mod_q, dtype = np.uint64, device = "cuda")
        # cryptoContext.qlql_inv_mod_ql_div_ql_mod_q_shoup = np.array(cryptoContext.qlql_inv_mod_ql_div_ql_mod_q_shoup, dtype = np.uint64, device = "cuda")
        # cryptoContext.q_inv_mod_q = np.array(cryptoContext.q_inv_mod_q, dtype = np.uint64, device = "cuda")
        # cryptoContext.q_inv_mod_q_shoup = np.array(cryptoContext.q_inv_mod_q_shoup, dtype = np.uint64, device = "cuda")
        # cryptoContext.swk_bx_cuda = np.array(cryptoContext.swk_bx_cuda, dtype = np.uint64, device = "cuda")
        # cryptoContext.swk_ax_cuda = np.array(cryptoContext.swk_ax_cuda, dtype = np.uint64, device = "cuda")
        # cryptoContext.inner_workspace = np.array(cryptoContext.inner_workspace, dtype = np.uint64, device = "cuda")
        # cryptoContext.inner_out = np.array(cryptoContext.inner_out, dtype = np.uint64, device = "cuda")
        # cryptoContext.moddown_out_ax = np.array(cryptoContext.moddown_out_ax, dtype = np.uint64, device = "cuda")
        # cryptoContext.moddown_out_bx = np.array(cryptoContext.moddown_out_bx, dtype = np.uint64, device = "cuda")
        # cryptoContext.modup_out = np.array(cryptoContext.modup_out, dtype = np.uint64, device = "cuda")
        # cryptoContext.rescale_out = np.array(cryptoContext.rescale_out, dtype = np.uint64, device = "cuda")
        # cryptoContext.automorphism_transform_out = np.array(cryptoContext.automorphism_transform_out, dtype = np.uint64, device = "cuda")
        # cryptoContext.switch_modulus_out = np.array(cryptoContext.switch_modulus_out, dtype = np.uint64, device = "cuda")
        # cryptoContext.PModq_cuda = np.array(cryptoContext.PModq_cuda, dtype = np.uint64, device = "cuda")

        # cryptoContext.key_map = [np.array(v, dtype = np.uint64, device = "cuda") for v in cryptoContext.key_map]

        # for key, value in cryptoContext.left_rot_key_map.items():
        #     cryptoContext.left_rot_key_map[key] = [np.array(v, dtype = np.uint64, device = "cuda") for v in value]
        # for key, value in cryptoContext.precompute_auto_map.items():
        #     cryptoContext.precompute_auto_map[key] = np.array(value, dtype = torch.int32, device = "cuda")

        # for key, value in cryptoContext.BsContext.QplusP_map.items():
        #     cryptoContext.BsContext.QplusP_map[key] = np.array(value, dtype = np.uint64, device = "cuda")
        # for key, value in cryptoContext.BsContext.QmuplusPmu_map.items():
        #     cryptoContext.BsContext.QmuplusPmu_map[key] = np.array(value, dtype = np.uint64, device = "cuda")

        return cryptoContext
