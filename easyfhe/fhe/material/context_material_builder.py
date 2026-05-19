import cmath
import math
import pickle
from functools import lru_cache

import numpy as np

from ..ops.encoding import encode_stage1
from .rotation import bit_reverse_indices, compute_auto_map
from ..runtime.scale_policy import split_rescale_tech


@lru_cache(maxsize=None)
def _crt_root_tables(moduli, roots, ring_dim):
    ring_dim = int(ring_dim)
    bit_reversed = bit_reverse_indices(ring_dim)
    root_pows = []
    root_pows_inv = []

    for modulus, root in zip(moduli, roots):
        modulus = int(modulus)
        root = int(root)
        root_inv = pow(root, -1, modulus)
        powers = np.empty(ring_dim, dtype=np.uint64)
        powers_inv = np.empty(ring_dim, dtype=np.uint64)
        power = 1
        power_inv = 1
        for idx in range(ring_dim):
            powers[idx] = power
            powers_inv[idx] = power_inv
            if idx < ring_dim - 1:
                power = (power * root) % modulus
                power_inv = (power_inv * root_inv) % modulus

        table = np.empty(ring_dim, dtype=np.uint64)
        table_inv = np.empty(ring_dim, dtype=np.uint64)
        table[bit_reversed] = powers
        table_inv[bit_reversed] = powers_inv
        table.setflags(write=False)
        table_inv.setflags(write=False)
        root_pows.append(table)
        root_pows_inv.append(table_inv)

    return tuple(root_pows), tuple(root_pows_inv)


def _shoup_value(value, prime):
    return (int(value) << 64) // int(prime)


@lru_cache(maxsize=None)
def _ntt_tables_cached(primes, roots, ring_dim):
    root_pows, root_pows_inv = _crt_root_tables(primes, roots, int(ring_dim))
    barret_k = []
    barret_ratio = []
    power_of_roots = []
    power_of_roots_shoup = []
    inverse_power_of_roots_div_two = []
    inverse_scaled_power_of_roots_div_two = []

    for prime, roots_for_prime, roots_inv_for_prime in zip(primes, root_pows, root_pows_inv):
        prime = int(prime)
        barret = math.floor(math.log2(prime)) + 63
        barret_k.append(barret)

        temp = 1 << (barret - 64)
        temp <<= 64
        barret_ratio.append(int(temp) // prime)

        two_inv = pow(2, -1, prime)
        inv_div_two = [(int(x) * two_inv) % prime for x in roots_inv_for_prime]

        power_of_roots.extend(int(x) for x in roots_for_prime)
        power_of_roots_shoup.extend(_shoup_value(x, prime) for x in roots_for_prime)
        inverse_power_of_roots_div_two.extend(inv_div_two)
        inverse_scaled_power_of_roots_div_two.extend(_shoup_value(x, prime) for x in inv_div_two)

    return (
        np.asarray(barret_k, dtype=np.uint64),
        np.asarray(barret_ratio, dtype=np.uint64),
        np.asarray(power_of_roots, dtype=np.uint64),
        np.asarray(power_of_roots_shoup, dtype=np.uint64),
        np.asarray(inverse_power_of_roots_div_two, dtype=np.uint64),
        np.asarray(inverse_scaled_power_of_roots_div_two, dtype=np.uint64),
    )


class ContextMaterialBuilder:
    def __init__(
        self,
        logN,
        logBsSlots_list,
        dcrtBits,
        specialMod,
        dnum,
        levelBudget_list,
        depth,
        moduliQ_scalar=None,
        moduliP_scalar=None,
        rootsQ=None,
        rootsP=None,
        MULT_SWK=None,
        rot_swk_map=None,
        autoIdx2rotIdx_map = None,
        secretKeyDist=None,
        rescaleTech=None,
        dim1=None,
        options=None,
        h=64,
        sigma=32,
    ):
        if moduliQ_scalar is None or rootsQ is None:
            raise ValueError("ContextMaterialBuilder requires moduliQ_scalar and rootsQ from the native sampler")
        if moduliP_scalar is None or rootsP is None:
            raise ValueError("ContextMaterialBuilder requires moduliP_scalar and rootsP from the native sampler")

        L = len(moduliQ_scalar)
        K = len(moduliP_scalar)
        alpha = int((L+dnum-1)//dnum)
        self.logBsSlots_list = logBsSlots_list
        self.secretKeyDist = secretKeyDist
        self.rescaleTech = rescaleTech
        self.scale_mode, self.rescale_policy = split_rescale_tech(rescaleTech)
        self.specialMod = specialMod

        self.logN = logN
        self.dcrtBits = dcrtBits
        self.L = int(L)
        self.K = int(K)
        self.dnum = dnum
        self.alpha = alpha
        self.h = h
        self.sigma = sigma
        self.N = int(1 << logN)
        self.M = self.N << 1
        self.logNh = logN - 1
        self.Nh = self.N >> 1

        self.total_left_rot_key_map = {}
        self.total_precompute_auto_map = {}
        self.encode_values = {}

        self.moduliQ_scalar, qRoots = self._init_crt_towers(
            moduliQ_scalar,
            rootsQ,
        )

        self.rootsQ = np.array(qRoots, dtype=np.uint64)
        self.q_mu = self._barrett_mu(self.moduliQ_scalar)
        # self.q_mu_cuda = np.array(q_mu, dtype=np.uint64)
        self.moduliQ = np.array(self.moduliQ_scalar, dtype=np.uint64)

        self.moduliP_scalar, pRoots = self._init_crt_towers(
            moduliP_scalar,
            rootsP,
        )
        p_mu = self._barrett_mu(self.moduliP_scalar)

        pHatInvModp, pHatModq, PInvModq, qInvModq = self._init_basis_switch_tables()

        self._init_scaling_factors()
        self._init_encode_params()


        self._init_runtime_workspace(moduliQ_scalar, moduliP_scalar)
        self._init_ntt_tables(qRoots, pRoots)
        self._init_modup_tables()
        self._init_moddown_tables(pHatInvModp, pHatModq, PInvModq)
        self._init_rescale_tables(qInvModq)
        self.primes = np.array(self.primes, dtype=np.uint64)


        self._install_eval_mult_key(MULT_SWK)
        self._install_rotation_keys(rot_swk_map or {}, autoIdx2rotIdx_map or {}, options)
        self._build_level_modulus_maps(p_mu)
        self._build_slot_conversion_masks(logN)

    def _init_basis_switch_tables(self):
        moduli_part_q = [0] * self.dnum
        for digit in range(self.dnum):
            moduli_part_q[digit] = int(1)
            for i in range(self.alpha * digit, self.alpha * (digit + 1)):
                if i < self.L:
                    moduli_part_q[digit] *= int(self.moduliQ_scalar[i])

        self.PartQlHatInvModq = [
            [[0 for _ in range(self.alpha)] for _ in range(self.alpha)]
            for _ in range(self.dnum)
        ]
        for digit in range(self.dnum):
            size_part_q = (self.L - (digit * self.alpha)) if (digit == self.dnum - 1) else self.alpha
            modulus_part_q = moduli_part_q[digit]
            for level in range(size_part_q):
                if level > 0:
                    modulus_part_q = int(
                        int(modulus_part_q) // int(self.moduliQ_scalar[digit * self.alpha + size_part_q - level])
                    )
                for i in range(size_part_q - level):
                    modulus = int(self.moduliQ_scalar[digit * self.alpha + i])
                    q_hat = modulus_part_q // modulus
                    self.PartQlHatInvModq[digit][size_part_q - level - 1][i] = int(
                        self.invMod(q_hat, modulus)
                    )

        self.PartQlHatModp = [
            [
                [[0 for _ in range(self.L + self.K)] for _ in range(self.alpha)]
                for _ in range(self.dnum)
            ]
            for _ in range(self.L)
        ]
        for level in range(self.L):
            beta = math.ceil((level + 1) / self.alpha)
            for digit in range(beta):
                part_q_size = (
                    (self.L - (beta - 1) * self.alpha)
                    if (beta == self.dnum and digit == beta - 1)
                    else self.alpha
                )
                digit_size = part_q_size
                modulus_part_q = int(moduli_part_q[digit])

                if digit == beta - 1:
                    digit_size = level + 1 - digit * self.alpha
                    for idx in range(digit_size, part_q_size):
                        modulus_part_q //= int(self.moduliQ_scalar[self.alpha * digit + idx])

                for i in range(digit_size):
                    part_q_hat = modulus_part_q // int(self.moduliQ_scalar[self.alpha * digit + i])
                    start_idx = digit * self.alpha
                    end_idx = start_idx + digit_size
                    complement_basis = (
                        self.moduliQ_scalar[:start_idx]
                        + self.moduliQ_scalar[end_idx: level + 1]
                        + self.moduliP_scalar
                    )

                    for j, modulus in enumerate(complement_basis):
                        self.PartQlHatModp[level][digit][i][j] = int(part_q_hat) % int(modulus)

        p_hat_modp = [0] * self.K
        p_hat_inv_modp = [0] * self.K
        for k in range(self.K):
            p_hat_modp[k] = int(1)
            for j in list(range(k)) + list(range(k + 1, self.K)):
                temp = int(self.moduliP_scalar[j] % self.moduliP_scalar[k])
                p_hat_modp[k] = (p_hat_modp[k] * temp) % int(self.moduliP_scalar[k])
        for k in range(self.K):
            p_hat_inv_modp[k] = int(self.invMod(int(p_hat_modp[k]), self.moduliP_scalar[k]))

        p_hat_modq = [[0] * self.L for _ in range(self.K)]
        for k in range(self.K):
            for i in range(self.L):
                p_hat_modq[k][i] = int(1)
                for s in list(range(k)) + list(range(k + 1, self.K)):
                    temp = int(self.moduliP_scalar[s]) % int(self.moduliQ_scalar[i])
                    p_hat_modq[k][i] = self.mulMod(
                        int(p_hat_modq[k][i]),
                        temp,
                        int(self.moduliQ_scalar[i]),
                    )

        self.PModq = [0] * self.L
        for i in range(self.L):
            self.PModq[i] = int(1)
            for k in range(self.K):
                temp = self.moduliP_scalar[k] % self.moduliQ_scalar[i]
                self.PModq[i] = self.mulMod(
                    int(self.PModq[i]),
                    int(temp),
                    int(self.moduliQ_scalar[i]),
                )

        p_inv_modq = [0] * self.L
        for i in range(self.L):
            p_inv_modq[i] = self.invMod(int(self.PModq[i]), int(self.moduliQ_scalar[i]))

        q_inv_modq = [[0 for _ in range(self.L)] for _ in range(self.L)]
        for i in range(self.L):
            for j in list(range(i)) + list(range(i + 1, self.L)):
                q_inv_modq[i][j] = self.invMod(
                    int(self.moduliQ_scalar[i]),
                    int(self.moduliQ_scalar[j]),
                )

        self.QlQlInvModqlDivqlModq = [[0] * (self.L - 1) for _ in range(self.L - 1)]
        for k in range(self.L - 1):
            level = self.L - (k + 1)
            for i in range(level):
                ql_inv_mod_ql = int(1)
                for j in range(level):
                    temp = self.invMod(self.moduliQ_scalar[j], self.moduliQ_scalar[level])
                    ql_inv_mod_ql = self.mulMod(
                        int(ql_inv_mod_ql),
                        int(temp),
                        int(self.moduliQ_scalar[level]),
                    )

                modulus_q = int(1)
                for j in range(level):
                    modulus_q *= int(self.moduliQ_scalar[j])

                result = int((int(ql_inv_mod_ql) * modulus_q) // int(self.moduliQ_scalar[level]))
                result %= int(self.moduliQ_scalar[i])
                self.QlQlInvModqlDivqlModq[k][i] = np.uint64(result)

        self.moduliQ_scalar = np.array(self.moduliQ_scalar, dtype=np.uint64)
        self.moduliP_scalar = np.array(self.moduliP_scalar, dtype=np.uint64)
        self.max_int_diffs = np.array(
            [
                (9223372036854775295 - p) % p
                for p in np.concatenate((self.moduliQ_scalar, self.moduliP_scalar)).tolist()
            ],
            dtype=np.uint64,
        )

        self.PartQlHatInvModq = np.array(self.PartQlHatInvModq, dtype=np.uint64)
        self.PartQlHatModp = np.array(self.PartQlHatModp, dtype=np.uint64)
        self.PModq = np.array(self.PModq, dtype=np.uint64)
        self.QlQlInvModqlDivqlModq = np.array(self.QlQlInvModqlDivqlModq, dtype=np.uint64)
        return (
            np.array(p_hat_inv_modp, dtype=np.uint64),
            np.array(p_hat_modq, dtype=np.uint64),
            np.array(p_inv_modq, dtype=np.uint64),
            np.array(q_inv_modq, dtype=np.uint64),
        )

    def _init_runtime_workspace(self, moduli_q, moduli_p):
        self.max_num_moduli = self.L + self.K
        self.chain_length = self.L
        self.num_special_moduli = self.K
        self.primes = np.hstack((moduli_q, moduli_p))

        self.power_of_roots = None
        self.power_of_roots_shoup = None
        self.inverse_power_of_roots_div_two = None
        self.inverse_scaled_power_of_roots_div_two = None
        self.power_of_roots_vec = []
        self.power_of_roots_shoup_vec = []
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

        self.beta = int((self.L + self.alpha - 1) / self.alpha)
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
            [0] * (self.num_moduli_after_modup * self.N),
            dtype=np.uint64,
        )
        self.mod_raise_out = np.array(
            [0] * (self.L * self.N),
            dtype=np.uint64,
        )

    def _init_ntt_tables(self, q_roots, p_roots):
        (
            self.barret_k,
            self.barret_ratio,
            self.power_of_roots,
            self.power_of_roots_shoup,
            self.inverse_power_of_roots_div_two,
            self.inverse_scaled_power_of_roots_div_two,
        ) = (
            np.array(value, dtype=np.uint64, copy=True)
            for value in _ntt_tables_cached(
                tuple(int(prime) for prime in self.primes),
                tuple(int(root) for root in list(q_roots) + list(p_roots)),
                self.N,
            )
        )

    def _init_modup_tables(self):
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
            for k in range(self.alpha):
                hat_inv_shoup = []
                hat_inverse_vec = self.PartQlHatInvModq[dnum_idx][k]
                hat_inverse_vec_modup.append(hat_inverse_vec)
                for k_idx in range(self.alpha):
                    prime_idx = dnum_idx * self.alpha + k_idx
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

    def _init_moddown_tables(self, p_hat_inv_modp, p_hat_modq, p_inv_modq):
        p_basis = self.primes[self.chain_length:]
        q_basis = self.set_difference(self.primes, p_basis)

        hat_inv_shoup_moddown = []
        for k in range(self.K):
            prime = self.primes[self.L + k]
            shoup = self.shoup(int(p_hat_inv_modp[k]), prime)
            hat_inv_shoup_moddown.append(shoup)
        self.hat_inverse_vec_moddown = np.array(
            np.array([p_hat_inv_modp], dtype=np.uint64),
            dtype=np.uint64,
        )
        self.hat_inverse_vec_shoup_moddown = np.array(
            np.array([hat_inv_shoup_moddown], dtype=np.uint64),
            dtype=np.uint64,
        )

        self.prod_q_i_mod_q_j_moddown = np.array(
            np.array([p_hat_modq.swapaxes(1, 0).flatten()], dtype=np.uint64),
            dtype=np.uint64,
        )

        prod_shoup = []
        for i, q_prime in enumerate(q_basis):
            prod_shoup.append(self.shoup(int(p_inv_modq[i]), q_prime))
        self.prod_inv_moddown = np.array(
            np.array([p_inv_modq], dtype=np.uint64),
            dtype=np.uint64,
        )
        self.prod_inv_shoup_moddown = np.array(
            np.array(prod_shoup, dtype=np.uint64),
            dtype=np.uint64,
        )

    def _init_rescale_tables(self, q_inv_modq):
        qlql_inv_mod_ql_div_ql_mod_q = self.QlQlInvModqlDivqlModq.reshape(-1)
        qlql_inv_vec = []
        qlql_inv_shoup_vec = []
        for i in range(self.L - 1):
            for j in range(self.L - 1):
                value = qlql_inv_mod_ql_div_ql_mod_q[i * (self.L - 1) + j]
                prime = self.primes[j]
                qlql_inv_vec.append(value)
                qlql_inv_shoup_vec.append(self.shoup(int(value), prime))
        self.qlql_inv_mod_ql_div_ql_mod_q = np.array(
            np.array(qlql_inv_vec, dtype=np.uint64),
            dtype=np.uint64,
        )
        self.qlql_inv_mod_ql_div_ql_mod_q_shoup = np.array(
            np.array(qlql_inv_shoup_vec, dtype=np.uint64),
            dtype=np.uint64,
        )

        q_inv_modq = q_inv_modq.reshape(-1)
        q_inv_vec = []
        q_inv_shoup_vec = []
        for i in range(self.L):
            for j in range(self.L):
                value = q_inv_modq[i * self.L + j]
                prime = self.primes[j]
                q_inv_vec.append(value)
                q_inv_shoup_vec.append(self.shoup(int(value), prime))
        self.q_inv_mod_q = np.array(
            np.array(q_inv_vec, dtype=np.uint64),
            dtype=np.uint64,
        )
        self.q_inv_mod_q_shoup = np.array(
            np.array(q_inv_shoup_vec, dtype=np.uint64),
            dtype=np.uint64,
        )

    def _init_crt_towers(self, moduli, roots):
        moduli = [int(modulus) for modulus in moduli]
        roots = [int(root) for root in roots]
        return moduli, roots

    def _barrett_mu(self, moduli):
        mu = []
        for modulus in moduli:
            value = 2**128 // int(modulus)
            low = value & ((1 << 64) - 1)
            high = value >> 64
            mu.append([low, high])
        return np.array(mu, dtype=np.uint64)

    def _init_scaling_factors(self):
        # These are stored as float here although the source values may be double precision.
        self.approxSF = 2**self.dcrtBits

        if self.scale_mode == "flexible":
            self.scalingFactorsReal = [0.0] * self.L
            if self.L == 1:
                self.scalingFactorsReal[0] = self.approxSF
            else:
                self.scalingFactorsReal[0] = float(self.moduliQ_scalar[self.L - 1])
                last_preset_factor = self.scalingFactorsReal[0]
                for k in range(1, self.L):
                    prev_sf = self.scalingFactorsReal[k - 1]
                    self.scalingFactorsReal[k] = prev_sf * prev_sf / float(self.moduliQ_scalar[self.L - k])
                    ratio = self.scalingFactorsReal[k] / last_preset_factor
                    if ratio <= 0.5 or ratio >= 2.0:
                        print(
                            "FLEXIBLEAUTO cannot support this number of levels in this parameter setting. Please use FIXEDMANUAL or FIXEDAUTO instead."
                        )

            self.scalingFactorsRealBig = [0.0] * (self.L - 1)
            if len(self.scalingFactorsRealBig) > 0:
                self.scalingFactorsRealBig[0] = self.scalingFactorsReal[0] * self.scalingFactorsReal[0]
                for k in range(1, self.L - 1):
                    self.scalingFactorsRealBig[k] = self.scalingFactorsReal[k] * self.scalingFactorsReal[k]
        else:
            self.scalingFactorsReal = []
            self.scalingFactorsRealBig = []

    def _init_encode_params(self):
        five_pows = 1
        self.encode_params_rotGroup = []
        for _ in range(self.Nh):
            self.encode_params_rotGroup.append(five_pows)
            five_pows = (five_pows * 5) % self.M

        self.encode_params_ksiPows = []
        for j in range(self.M):
            angle = 2.0 * math.pi * j / self.M
            self.encode_params_ksiPows.append(cmath.exp(1j * angle))
        self.encode_params_ksiPows.append(self.encode_params_ksiPows[0])
        self.encode_params_ksiPows = np.array(self.encode_params_ksiPows, dtype=np.complex128)
        self.encode_params_rotGroup = np.array(self.encode_params_rotGroup, dtype=np.uint32)
        self.encode_bitrev_indices = {
            log_slot: self._bitrev_indices(log_slot)
            for log_slot in range(2, self.logN)
        }

    def _bitrev_indices(self, k):
        n = 1 << int(k)
        rev = np.arange(n, dtype=np.uint32)
        bits = np.arange(k, dtype=np.uint32)
        rev = ((rev[:, None] >> bits) & 1).dot(1 << (k - 1 - bits))
        return rev

    def _install_eval_mult_key(self, mult_swk):
        self.mult_swk_bx = np.array(
            mult_swk[0].reshape(self.dnum, self.L + self.K, self.N),
            dtype=np.uint64,
        )
        self.mult_swk_ax = np.array(
            mult_swk[1].reshape(self.dnum, self.L + self.K, self.N),
            dtype=np.uint64,
        )

    def _install_rotation_keys(self, rot_swk_map, auto_idx_to_rot_idx_map, options):
        max_rns_limbs_by_rot_evk = getattr(options, "rotation_key_limb_limits", None) or {}
        for rot_swk in rot_swk_map.values():
            for auto_idx, bx, ax in rot_swk:
                rot_idx = auto_idx_to_rot_idx_map[auto_idx]
                if rot_idx < 0:
                    rot_idx = self.N // 2 + rot_idx
                limb = max_rns_limbs_by_rot_evk.get(int(rot_idx))
                if limb is None:
                    limb = self.L
                beta = (limb + self.alpha - 1) // self.alpha
                reshaped_bx = np.array(bx, dtype=np.uint64).reshape(self.dnum, -1, self.N)
                reshaped_ax = np.array(ax, dtype=np.uint64).reshape(self.dnum, -1, self.N)

                self.total_left_rot_key_map[int(rot_idx)] = [
                    np.concatenate(
                        [
                            reshaped_bx[:beta, :limb, :],
                            reshaped_bx[:beta, self.L:self.L + self.K, :],
                        ],
                        axis=1,
                    ),
                    np.concatenate(
                        [
                            reshaped_ax[:beta, :limb, :],
                            reshaped_ax[:beta, self.L:self.L + self.K, :],
                        ],
                        axis=1,
                    ),
                ]
                self.total_precompute_auto_map[int(rot_idx)] = self.compute_auto_map(int(auto_idx), self.N)

    def _build_level_modulus_maps(self, p_mu):
        self.QplusP_map = {}
        self.QmuplusPmu_map = {}
        self.QbarretKplusPbarretK_map = {}
        self.QbarretRatioplusPbarretRatio_map = {}
        self.QmaxdiffplusPmaxdiff_map = {}
        for cur_limbs in range(1, self.L + 1):
            self.QplusP_map[cur_limbs] = np.array(
                np.concatenate((self.moduliQ_scalar[:cur_limbs], self.moduliP_scalar[:self.K])),
                dtype=np.uint64,
            )
            self.QmuplusPmu_map[cur_limbs] = np.array(
                np.concatenate((self.q_mu[:cur_limbs], p_mu[:self.K])),
                dtype=np.uint64,
            )
            self.QbarretKplusPbarretK_map[cur_limbs] = np.array(
                np.concatenate((self.barret_k[:cur_limbs], self.barret_k[-self.K:])),
                dtype=np.uint64,
            )
            self.QbarretRatioplusPbarretRatio_map[cur_limbs] = np.array(
                np.concatenate((self.barret_ratio[:cur_limbs], self.barret_ratio[-self.K:])),
                dtype=np.uint64,
            )
            self.QmaxdiffplusPmaxdiff_map[cur_limbs] = np.array(
                np.concatenate((self.max_int_diffs[:cur_limbs], self.max_int_diffs[-self.K:])),
                dtype=np.uint64,
            )

    def _build_slot_conversion_masks(self, logN):
        for i in range(9, int(logN)):
            for j in range(8, i):
                mask = np.asarray(
                    [1] * (1 << j) + [0] * ((1 << i) - (1 << j)),
                    dtype=np.float64,
                )
                mask = encode_stage1(mask, 1 << i, self.N)
                self.encode_values["slot_conversion_mask_{}to{}".format(1 << i, 1 << j)] = mask

    def compute_auto_map(self, k, N):
        return compute_auto_map(k, N)

    def shoup(self, in_value, prime):
        temp = int(in_value) << 64
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

    def method(self):  # function to initialize variables
        pass

    def Serialize(self):
        return pickle.dumps(self)

    def Deserialize(ctx_bytes):
        cryptoContext = pickle.loads(ctx_bytes)

        return cryptoContext
