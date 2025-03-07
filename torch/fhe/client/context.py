import random
import warnings
import pickle
import sympy
import cmath
from .bs_context import *


class __FOR_SAVE_ONLY_Context:
    def __init__(
        self,
        logN,
        logBsSlots_list,
        specialMod,
        levelBudget_list,
        moduliQ_scalar=None,
        moduliP_scalar=None,
        rootsQ=None,
        rootsP=None,
        MULT_SWK=None,
        rot_swk_map=None,
        autoIdx2rotIdx_map=None,
        boot_cnst_map=None,
        dim1=None,
        h=64,
        sigma=32,
    ):
        L = len(moduliQ_scalar)
        K = len(moduliP_scalar)
        self.logBsSlots_list = logBsSlots_list
        self.BsContext_map = {}
        self.specialMod = specialMod
        self.qVec = None

        self.logN = logN
        self.L = int(L)
        self.K = int(K)
        self.h = h
        self.sigma = sigma
        self.N = int(1 << logN)
        self.M = self.N << 1
        self.logNh = logN - 1
        self.Nh = self.N >> 1
        self.p = 1 << 59

        self.moduliQ_scalar = [0] * L
        qRoots = [0] * L
        qRootsInv = [0] * L
        qRootPows = [[] for _ in range(L)]
        qRootPowsInv = [[] for _ in range(L)]
        self.mult_key_map = None
        self.slots_left_rot_key_map = {}
        self.slots_precompute_auto_map = {}
        bnd = 1
        cnt = 1
        if moduliQ_scalar is None and rootsQ is None:
            while True:
                prime = (1 << 60) + bnd * self.M + 1
                if self.is_prime(prime):
                    self.moduliQ_scalar[0] = prime
                    break
                bnd += 1
            qRoots[0] = self.root_of_unity(order=self.M, modulus=self.moduliQ_scalar[0])
            bnd = 1
            while cnt < L:
                prime1 = (1 << 59) + bnd * self.M + 1
                if self.is_prime(prime1):
                    self.moduliQ_scalar[cnt] = prime1
                    cnt += 1
                prime2 = (1 << 59) - bnd * self.M + 1
                if self.is_prime(prime2):
                    self.moduliQ_scalar[cnt] = prime2
                    qRoots[cnt] = self.root_of_unity(
                        order=self.M, modulus=self.moduliQ_scalar[cnt - 1]
                    )
                    cnt += 1
                bnd += 1
            dcrbits = 59
            if dcrbits - logN - 1 - math.ceil(math.log2(bnd)) < 10:
                print("ERROR: too small number of precision")
                print("TRY to use larger dcrtBits or smaller depth")
        else:
            if moduliQ_scalar is None:
                print("moduliQ_scalar needs to be set!")
                return
            elif rootsQ is None:
                print("rootsQ needs to be set!")
                return
            for i in range(L):
                self.moduliQ_scalar[i] = moduliQ_scalar[i]
                qRoots[i] = rootsQ[i]

        for i in range(L):
            qRootsInv[i] = self.invMod(qRoots[i], int(self.moduliQ_scalar[i]))
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
                        int(power), int(qRoots[i]), int(self.moduliQ_scalar[i])
                    )
                    powerInv = self.mulMod(
                        powerInv, int(qRootsInv[i]), int(self.moduliQ_scalar[i])
                    )
        q_mu = []
        for mod in self.moduliQ_scalar:
            x = 2**128 // int(mod)
            low = x & ((1 << 64) - 1)
            high = x >> 64
            q_mu.append([low, high])
        self.q_mu = np.array(q_mu, dtype=np.uint64)
        self.moduliQ = np.array(self.moduliQ_scalar, dtype=np.uint64)

        self.moduliP_scalar = [0] * self.K
        pRoots = [0] * self.K
        pRootsInv = [0] * self.K
        pRootPows = [[] for _ in range(self.K)]
        pRootPowsInv = [[] for _ in range(self.K)]

        if moduliP_scalar is None and rootsP is None:
            cnt = 0
            while cnt < self.K:
                prime1 = (1 << specialMod) + bnd * self.M + 1
                if self.is_prime(prime1):
                    self.moduliP_scalar[cnt] = prime1
                    pRoots[cnt] = self.root_of_unity(
                        order=self.M, modulus=self.moduliP_scalar[cnt]
                    )
                    cnt += 1
                if cnt == self.K:
                    break
                prime2 = (1 << specialMod) - bnd * self.M + 1
                if self.is_prime(prime2):
                    self.moduliP_scalar[cnt] = prime2
                    pRoots[cnt] = self.root_of_unity(
                        order=self.M, modulus=self.moduliP_scalar[cnt]
                    )
                    cnt += 1
                bnd += 1

        else:
            if moduliP_scalar is None:
                print("moduliP_scalar needs to be set")
                return
            elif rootsP is None:
                print("rootsP needs to be set")
                return
            for i in range(K):
                self.moduliP_scalar[i] = moduliP_scalar[i]
                pRoots[i] = rootsP[i]

        for i in range(K):
            pRootsInv[i] = self.invMod(pRoots[i], int(self.moduliP_scalar[i]))
            pRootPows[i] = [0] * self.N
            pRootPowsInv[i] = [0] * self.N
            power = int(1)
            powerInv = int(1)
            for j in range(self.N):
                jprime = self.bitReverse(j) >> (32 - self.logN)
                pRootPows[i][jprime] = int(power)
                tmp = int(power) << 64
                pRootPowsInv[i][jprime] = powerInv
                if j < self.N - 1:
                    power = self.mulMod(
                        power, int(pRoots[i]), int(self.moduliP_scalar[i])
                    )
                    powerInv = self.mulMod(
                        powerInv, int(pRootsInv[i]), int(self.moduliP_scalar[i])
                    )

        p_mu = []
        for mod in self.moduliP_scalar:
            x = 2**128 // int(mod)
            low = x & ((1 << 64) - 1)
            high = x >> 64
            p_mu.append([low, high])
        self.p_mu = np.array(p_mu, dtype=np.uint64)

        moduliPartQ = [0] * 1
        moduliPartQ[0] = int(1)
        for i in range(0, self.L):
            if i < L:
                moduliPartQ[0] *= int(self.moduliQ_scalar[i])

        self.PartQlHatInvModq = [[[0 for _ in range(self.L)] for _ in range(self.L)]]
        sizePartQk = L
        modulusPartQ = moduliPartQ[0]
        for l in range(sizePartQk):
            if l > 0:
                modulusPartQ = int(
                    int(modulusPartQ) // int(self.moduliQ_scalar[sizePartQk - l])
                )
            for i in range(sizePartQk - l):
                moduli = int(self.moduliQ_scalar[i])
                QHat = modulusPartQ // moduli
                QHatInvModqi = int(self.invMod(QHat, moduli))
                self.PartQlHatInvModq[0][sizePartQk - l - 1][i] = QHatInvModqi

        self.PartQlHatModp = [
            [[[0 for _ in range(L + K)] for _ in range(self.L)]] for _ in range(L)
        ]

        for l in range(L):
            partQ_size = L
            modulusPartQ = int(moduliPartQ[0])

            digitSize = l + 1
            for idx in range(digitSize, partQ_size):
                modulusPartQ //= int(self.moduliQ_scalar[idx])

            for i in range(digitSize):
                partQHat = modulusPartQ // int(self.moduliQ_scalar[i])

                start_idx = 0
                end_idx = start_idx + digitSize
                complBasis_vec = (
                    self.moduliQ_scalar[:start_idx]
                    + self.moduliQ_scalar[end_idx : l + 1]
                    + self.moduliP_scalar
                )

                for j, mod in enumerate(complBasis_vec):
                    QHatModpj = int(partQHat) % int(mod)
                    self.PartQlHatModp[l][0][i][j] = QHatModpj

        self.pHatModp = [0] * K
        self.pHatInvModp = [0] * K
        for k in range(K):
            self.pHatModp[k] = int(1)
            for j in list(range(k)) + list(range(k + 1, K)):
                temp = int(self.moduliP_scalar[j] % self.moduliP_scalar[k])
                self.pHatModp[k] = (self.pHatModp[k] * temp) % int(
                    self.moduliP_scalar[k]
                )

        for k in range(K):
            self.pHatInvModp[k] = int(
                self.invMod(int(self.pHatModp[k]), self.moduliP_scalar[k])
            )

        self.pHatModq = [[0] * L for _ in range(K)]
        for k in range(K):
            for i in range(L):
                self.pHatModq[k][i] = int(1)
                for s in list(range(k)) + list(range(k + 1, K)):
                    temp = int(self.moduliP_scalar[s]) % int(self.moduliQ_scalar[i])
                    self.pHatModq[k][i] = self.mulMod(
                        int(self.pHatModq[k][i]), temp, int(self.moduliQ_scalar[i])
                    )

        self.PModq = [0] * L

        for i in range(L):
            self.PModq[i] = int(1)
            for k in range(K):
                temp = self.moduliP_scalar[k] % self.moduliQ_scalar[i]
                self.PModq[i] = self.mulMod(
                    int(self.PModq[i]), int(temp), int(self.moduliQ_scalar[i])
                )

        self.PInvModq = [0] * L

        for i in range(L):
            self.PInvModq[i] = self.invMod(
                int(self.PModq[i]), int(self.moduliQ_scalar[i])
            )

        qInvModq = [[0 for _ in range(L)] for _ in range(L)]
        for i in range(L):
            for j in list(range(i)) + list(range(i + 1, L)):
                qInvModq[i][j] = self.invMod(
                    int(self.moduliQ_scalar[i]), int(self.moduliQ_scalar[j])
                )

        self.QlQlInvModqlDivqlModq = [[0] * (L - 1) for _ in range(L - 1)]
        for k in range(L - 1):
            l = L - (k + 1)
            for i in range(l):
                QlInvModql = int(1)
                for j in range(l):
                    temp = self.invMod(self.moduliQ_scalar[j], self.moduliQ_scalar[l])
                    QlInvModql = self.mulMod(
                        int(QlInvModql), int(temp), int(self.moduliQ_scalar[l])
                    )

                modulusQ = int(1)
                for j in range(l):
                    modulusQ *= int(self.moduliQ_scalar[j])

                result = int(
                    (int(QlInvModql) * modulusQ) // int(self.moduliQ_scalar[l])
                )
                result %= int(self.moduliQ_scalar[i])

                self.QlQlInvModqlDivqlModq[k][i] = np.uint64(result)

        self.mult_swk = [None, None]
        self.mult_swk[0] = MULT_SWK[0]
        self.mult_swk[1] = MULT_SWK[1]

        self.moduliQ_scalar = np.array(self.moduliQ_scalar, dtype=np.uint64)
        self.moduliP_scalar = np.array(self.moduliP_scalar, dtype=np.uint64)

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

        self.approxSF = 2**59

        if True:
            self.max_num_moduli = self.L + self.K
            self.chain_length = self.L
            self.num_special_moduli = self.K
            self.primes = np.hstack((moduliQ_scalar, moduliP_scalar))

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
            self.num_moduli_after_modup = self.max_num_moduli
            self.hat_inverse_vec_modup = None
            self.hat_inverse_vec_shoup_modup = None
            self.prod_q_i_mod_q_j_modup = None
            self.num_moduli_after_moddown = self.chain_length
            self.hat_inverse_vec_moddown = []
            self.hat_inverse_vec_shoup_moddown = []
            self.prod_q_i_mod_q_j_moddown = []
            self.prod_inv_moddown = []
            self.prod_inv_shoup_moddown = []
            self.qlql_inv_mod_ql_div_ql_mod_q = None
            self.qlql_inv_mod_ql_div_ql_mod_q_shoup = None
            self.q_inv_mod_q = None
            self.q_inv_mod_q_shoup = None

            self.swk_bx = np.array(self.mult_swk[0].reshape(-1), dtype=np.uint64)
            self.swk_ax = np.array(self.mult_swk[1].reshape(-1), dtype=np.uint64)

            self.inner_workspace = np.array(
                [0] * (4 * self.num_moduli_after_modup * self.N * 1),
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
                [0] * (self.num_moduli_after_modup * self.N * 1),
                dtype=np.uint64,
            )
            self.rescale_out = np.array(
                [0] * ((self.L - 1) * self.N),
                dtype=np.uint64,
            )
            self.automorphism_transform_out = np.array(
                [0] * (self.num_moduli_after_modup * self.N * 1),
                dtype=np.uint64,
            )
            self.mod_raise_out = np.array(
                [0] * (self.L * self.N),
                dtype=np.uint64,
            )

            power_of_roots = qRootPows + pRootPows
            inverse_power_of_roots = qRootPowsInv + pRootPowsInv

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

            prod_q_i_mod_q_j_modup = []
            for l in range(self.L):
                prod_qi_mod_qj = []
                prod_q_i_mod_q_j = self.PartQlHatModp[l][0]
                prod_q_i_mod_q_j = prod_q_i_mod_q_j.swapaxes(1, 0).flatten()
                prod_qi_mod_qj.append(prod_q_i_mod_q_j)
                prod_q_i_mod_q_j_modup.append(prod_qi_mod_qj)
            self.prod_q_i_mod_q_j_modup = np.array(
                np.array(prod_q_i_mod_q_j_modup, dtype=np.uint64),
                dtype=np.uint64,
            )

            hat_inverse_vec_modup = []
            hat_inverse_vec_shoup_modup = []
            for k in range(self.L):
                hat_inv_shoup = []
                hat_inverse_vec = self.PartQlHatInvModq[0][k]
                hat_inverse_vec_modup.append(hat_inverse_vec)
                for k_idx in range(self.L):
                    prime_idx = k_idx
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

            end_length = self.chain_length
            start_begin = self.primes[end_length:]

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

            self.primes = np.array(self.primes, dtype=np.uint64)

        swk_bx = MULT_SWK[0].reshape(1, L + K, self.N)
        swk_ax = MULT_SWK[1].reshape(1, L + K, self.N)
        key_map_ax_fixed = np.array(swk_ax, dtype=np.uint64)
        key_map_bx_fixed = np.array(swk_bx, dtype=np.uint64)
        self.mult_key_map = [key_map_bx_fixed, key_map_ax_fixed]

        for key, ROT_SWK in rot_swk_map.items():
            left_rot_key_map = {}
            precompute_auto_map = {}
            for autoIdx, bx, ax in ROT_SWK:
                rotIdx = autoIdx2rotIdx_map[autoIdx]
                if int(rotIdx) < 0:
                    rotIdx = self.N // 2 + rotIdx
                left_rot_key_map[int(rotIdx)] = [
                    np.array(bx, dtype=np.uint64).reshape(1, -1, self.N),
                    np.array(ax, dtype=np.uint64).reshape(1, -1, self.N),
                ]
                precompute_auto_map[int(rotIdx)] = self.compute_auto_map(
                    int(autoIdx), self.N
                )
            self.slots_left_rot_key_map[key] = left_rot_key_map
            self.slots_precompute_auto_map[key] = precompute_auto_map

        if logBsSlots_list[0] != 0 and levelBudget_list != [[0, 0]]:
            for logBsSlots, levelBudget in zip(self.logBsSlots_list, levelBudget_list):
                self.BsContext_map[str(logBsSlots)] = BsContext(
                    self.N,
                    self.moduliQ_scalar,
                    self.moduliP_scalar,
                    self.q_mu,
                    self.p_mu,
                    0,
                    boot_cnst_map[str(logBsSlots)],
                )
        else:
            assert logBsSlots_list[0] == 0 and levelBudget_list == [[0, 0]]

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
        m = n << 1
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

    def method(self):
        pass

    def Serialize(self):
        return pickle.dumps(self)

    def Deserialize(ctx_bytes):
        cryptoContext = pickle.loads(ctx_bytes)

        return cryptoContext
