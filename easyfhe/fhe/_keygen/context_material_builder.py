import cmath
import math
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import easyfhe as torch


@lru_cache(maxsize=None)
def _bit_reverse_indices(ring_dim):
    ring_dim = int(ring_dim)
    if ring_dim <= 0 or ring_dim & (ring_dim - 1):
        raise ValueError(f"ring_dim must be a positive power of two, got {ring_dim}")

    logn = ring_dim.bit_length() - 1
    values = np.arange(ring_dim, dtype=np.uint32)
    values = ((values & np.uint32(0x55555555)) << np.uint32(1)) | ((values >> np.uint32(1)) & np.uint32(0x55555555))
    values = ((values & np.uint32(0x33333333)) << np.uint32(2)) | ((values >> np.uint32(2)) & np.uint32(0x33333333))
    values = ((values & np.uint32(0x0F0F0F0F)) << np.uint32(4)) | ((values >> np.uint32(4)) & np.uint32(0x0F0F0F0F))
    values = ((values & np.uint32(0x00FF00FF)) << np.uint32(8)) | ((values >> np.uint32(8)) & np.uint32(0x00FF00FF))
    values = (values << np.uint32(16)) | (values >> np.uint32(16))
    result = (values >> np.uint32(32 - logn)).astype(np.int32)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=None)
def _crt_root_tables(moduli, roots, ring_dim):
    ring_dim = int(ring_dim)
    bit_reversed = _bit_reverse_indices(ring_dim)
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


def _inv_mod(value, modulus):
    return pow(int(value) % int(modulus), -1, int(modulus))


def _mul_mod(a, b, modulus):
    return int(((int(a) % int(modulus)) * (int(b) % int(modulus))) % int(modulus))


def _as_uint64_vector(name, value):
    array = np.asarray(value, dtype=np.uint64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D uint64 vector, got shape {array.shape}")
    return np.ascontiguousarray(array)


def _as_eval_mult_key(value, *, dnum, limbs, ring_dim):
    array = np.asarray(value, dtype=np.uint64)
    expected_numel = 2 * int(dnum) * int(limbs) * int(ring_dim)
    if array.size != expected_numel:
        raise ValueError(
            "eval_mult_key must contain "
            f"{expected_numel} uint64 values for shape "
            f"(2, {int(dnum)}, {int(limbs)}, {int(ring_dim)}), got {array.shape}"
        )
    return np.ascontiguousarray(array.reshape(2, int(dnum), int(limbs), int(ring_dim)))


def _as_tensor(value, *, dtype):
    if torch.is_tensor(value):
        return value.to(dtype=dtype) if value.dtype != dtype else value
    return torch.as_tensor(value, dtype=dtype)


def _slot_bit_reverse_indices(log_slots):
    return _bit_reverse_indices(1 << int(log_slots)).astype(np.uint32, copy=True)


def _normalize_scale_mode(value):
    value = "fixed" if value is None else str(value).lower()
    if value != "fixed":
        raise ValueError(f"scale_mode must be 'fixed', got {value!r}")
    return value


def _normalize_rescale_policy(value):
    value = "manual" if value is None else str(value).lower()
    if value not in {"manual", "auto"}:
        raise ValueError(f"rescale_policy must be 'manual' or 'auto', got {value!r}")
    return value


def _to_runtime_material(material):
    primes = _as_tensor(material.primes, dtype=torch.uint64)

    encode_bitrev_indices = dict(material.encode_bitrev_indices)
    for key, value in encode_bitrev_indices.items():
        encode_bitrev_indices[key] = _as_tensor(value, dtype=torch.uint32)

    uint64_map_names = (
        "QplusP_map",
        "QmuplusPmu_map",
        "QbarretKplusPbarretK_map",
        "QbarretRatioplusPbarretRatio_map",
        "QmaxdiffplusPmaxdiff_map",
    )
    uint64_maps = {
        name: {
            key: _as_tensor(value, dtype=torch.uint64)
            for key, value in getattr(material, name).items()
        }
        for name in uint64_map_names
    }

    left_rot_key_map = {
        int(rot_idx): [
            _as_tensor(key_pair[0], dtype=torch.uint64),
            _as_tensor(key_pair[1], dtype=torch.uint64),
        ]
        for rot_idx, key_pair in material.total_left_rot_key_map.items()
    }
    precompute_auto_map = {
        int(rot_idx): _as_tensor(auto_map, dtype=torch.int32)
        for rot_idx, auto_map in material.total_precompute_auto_map.items()
    }
    inverse_precompute_auto_map = {
        int(rot_idx): _as_tensor(auto_map, dtype=torch.int32)
        for rot_idx, auto_map in material.total_inverse_precompute_auto_map.items()
    }

    return SimpleNamespace(
        L=material.L,
        dnum=material.dnum,
        alpha=material.alpha,
        K=material.K,
        M=material.M,
        N=material.N,
        Nh=material.Nh,
        approxSF=material.approxSF,
        h=material.h,
        levelBudget=material.levelBudget,
        logN=material.logN,
        logNh=material.logNh,
        logBsSlots_list=material.logBsSlots_list,
        auxModSize=material.auxModSize,
        scale_mode=material.scale_mode,
        rescale_policy=material.rescale_policy,
        dcrtBits=material.dcrtBits,
        max_num_moduli=material.max_num_moduli,
        secretKeyDist=material.secretKeyDist,
        sigma=material.sigma,
        inBS=material.inBS,
        primes=primes,
        barret_k=_as_tensor(material.barret_k, dtype=torch.uint64),
        barret_ratio=_as_tensor(material.barret_ratio, dtype=torch.uint64),
        q_mu=_as_tensor(material.q_mu, dtype=torch.uint64),
        moduliP_scalar=material.moduliP_scalar,
        moduliQ_scalar=material.moduliQ_scalar,
        moduliQ=_as_tensor(material.moduliQ, dtype=torch.uint64),
        scalingFactorsReal=material.scalingFactorsReal,
        scalingFactorsRealBig=material.scalingFactorsRealBig,
        PModq=_as_tensor(material.PModq, dtype=torch.uint64),
        max_int_diffs=_as_tensor(material.max_int_diffs, dtype=torch.uint64),
        QmuplusPmu_map=uint64_maps["QmuplusPmu_map"],
        QplusP_map=uint64_maps["QplusP_map"],
        QmaxdiffplusPmaxdiff_map=uint64_maps["QmaxdiffplusPmaxdiff_map"],
        QbarretKplusPbarretK_map=uint64_maps["QbarretKplusPbarretK_map"],
        QbarretRatioplusPbarretRatio_map=uint64_maps["QbarretRatioplusPbarretRatio_map"],
        automorphism_transform_out=_as_tensor(material.automorphism_transform_out, dtype=torch.uint64),
        inner_out=_as_tensor(material.inner_out, dtype=torch.uint64),
        moddown_out_ax=_as_tensor(material.moddown_out_ax, dtype=torch.uint64),
        moddown_out_bx=_as_tensor(material.moddown_out_bx, dtype=torch.uint64),
        modup_out=_as_tensor(material.modup_out, dtype=torch.uint64),
        rescale_out=_as_tensor(material.rescale_out, dtype=torch.uint64),
        mod_raise_out=_as_tensor(material.mod_raise_out, dtype=torch.uint64),
        hat_inverse_vec_moddown=_as_tensor(material.hat_inverse_vec_moddown, dtype=torch.uint64),
        hat_inverse_vec_shoup_moddown=_as_tensor(material.hat_inverse_vec_shoup_moddown, dtype=torch.uint64),
        prod_inv_moddown=_as_tensor(material.prod_inv_moddown, dtype=torch.uint64),
        prod_inv_shoup_moddown=_as_tensor(material.prod_inv_shoup_moddown, dtype=torch.uint64),
        prod_q_i_mod_q_j_moddown=_as_tensor(material.prod_q_i_mod_q_j_moddown, dtype=torch.uint64),
        hat_inverse_vec_modup=_as_tensor(material.hat_inverse_vec_modup, dtype=torch.uint64),
        hat_inverse_vec_shoup_modup=_as_tensor(material.hat_inverse_vec_shoup_modup, dtype=torch.uint64),
        prod_q_i_mod_q_j_modup=_as_tensor(material.prod_q_i_mod_q_j_modup, dtype=torch.uint64),
        inner_workspace=_as_tensor(material.inner_workspace, dtype=torch.uint64),
        mult_swk_ax=_as_tensor(material.mult_swk_ax, dtype=torch.uint64),
        mult_swk_bx=_as_tensor(material.mult_swk_bx, dtype=torch.uint64),
        inverse_power_of_roots_div_two=_as_tensor(material.inverse_power_of_roots_div_two, dtype=torch.uint64),
        inverse_scaled_power_of_roots_div_two=_as_tensor(material.inverse_scaled_power_of_roots_div_two, dtype=torch.uint64),
        power_of_roots=_as_tensor(material.power_of_roots, dtype=torch.uint64),
        power_of_roots_shoup=_as_tensor(material.power_of_roots_shoup, dtype=torch.uint64),
        left_rot_key_map=left_rot_key_map,
        precompute_auto_map=precompute_auto_map,
        inverse_precompute_auto_map=inverse_precompute_auto_map,
        q_inv_mod_q=_as_tensor(material.q_inv_mod_q, dtype=torch.uint64),
        q_inv_mod_q_shoup=_as_tensor(material.q_inv_mod_q_shoup, dtype=torch.uint64),
        qlql_inv_mod_ql_div_ql_mod_q=_as_tensor(material.qlql_inv_mod_ql_div_ql_mod_q, dtype=torch.uint64),
        qlql_inv_mod_ql_div_ql_mod_q_shoup=_as_tensor(material.qlql_inv_mod_ql_div_ql_mod_q_shoup, dtype=torch.uint64),
        encode_params_ksiPows=_as_tensor(material.encode_params_ksiPows, dtype=torch.complex128),
        encode_params_rotGroup=_as_tensor(material.encode_params_rotGroup, dtype=torch.uint32),
        encode_bitrev_indices=encode_bitrev_indices,
    )


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
    @classmethod
    def from_server(cls, server_material, options):
        return cls.from_public_params(
            log_n=server_material.log_n,
            depth=server_material.depth,
            dcrt_bits=server_material.dcrt_bits,
            special_mod=server_material.special_mod,
            dnum=server_material.dnum,
            secret_key_dist=server_material.secret_key_dist,
            scale_mode=server_material.scale_mode,
            rescale_policy=server_material.rescale_policy,
            moduli_q=server_material.moduli_q,
            roots_q=server_material.roots_q,
            moduli_p=server_material.moduli_p,
            roots_p=server_material.roots_p,
            eval_mult_key=server_material.eval_mult_key,
            rotation_keys=server_material.rotation_keys,
            options=options,
        )

    @classmethod
    def from_public_params(
        cls,
        *,
        log_n,
        depth,
        dcrt_bits,
        special_mod,
        dnum,
        secret_key_dist,
        scale_mode,
        rescale_policy,
        moduli_q,
        roots_q,
        moduli_p,
        roots_p,
        eval_mult_key,
        rotation_keys=(),
        options=None,
    ):
        return cls(
            log_n=log_n,
            depth=depth,
            dcrt_bits=dcrt_bits,
            special_mod=special_mod,
            dnum=dnum,
            moduli_q=moduli_q,
            roots_q=roots_q,
            moduli_p=moduli_p,
            roots_p=roots_p,
            eval_mult_key=eval_mult_key,
            rotation_keys=rotation_keys,
            secret_key_dist=secret_key_dist,
            scale_mode=scale_mode,
            rescale_policy=rescale_policy,
            options=options,
        )

    def to_runtime_material(self):
        return _to_runtime_material(self)

    def __init__(
        self,
        *,
        log_n,
        depth,
        dcrt_bits,
        special_mod,
        dnum,
        moduli_q,
        roots_q,
        moduli_p,
        roots_p,
        eval_mult_key,
        rotation_keys=(),
        secret_key_dist=None,
        scale_mode=None,
        rescale_policy=None,
        options=None,
        h=64,
        sigma=32,
    ):
        log_n = int(log_n)
        dnum = int(dnum)
        special_mod = int(special_mod)
        moduli_q = _as_uint64_vector("moduli_q", moduli_q)
        moduli_p = _as_uint64_vector("moduli_p", moduli_p)
        roots_q = _as_uint64_vector("roots_q", roots_q)
        roots_p = _as_uint64_vector("roots_p", roots_p)

        if log_n <= 0:
            raise ValueError(f"log_n must be positive, got {log_n}")
        L = len(moduli_q)
        K = len(moduli_p)
        if L <= 0:
            raise ValueError("moduli_q must not be empty")
        if K <= 0:
            raise ValueError("moduli_p must not be empty")
        if len(roots_q) != L:
            raise ValueError(f"roots_q length must match moduli_q length {L}, got {len(roots_q)}")
        if len(roots_p) != K:
            raise ValueError(f"roots_p length must match moduli_p length {K}, got {len(roots_p)}")
        if dnum <= 0:
            raise ValueError(f"dnum must be positive, got {dnum}")

        N = 1 << log_n
        eval_mult_key = _as_eval_mult_key(
            eval_mult_key,
            dnum=dnum,
            limbs=L + K,
            ring_dim=N,
        )
        alpha = int((L+dnum-1)//dnum)
        self.logBsSlots_list = []
        self.levelBudget = []
        self.auxModSize = special_mod
        self.secretKeyDist = secret_key_dist
        self.scale_mode = _normalize_scale_mode(scale_mode)
        self.rescale_policy = _normalize_rescale_policy(rescale_policy)
        self.specialMod = special_mod
        self.inBS = False

        self.logN = log_n
        self.dcrtBits = int(dcrt_bits)
        self.L = int(L)
        self.K = int(K)
        self.dnum = dnum
        self.alpha = alpha
        self.h = h
        self.sigma = sigma
        self.N = int(N)
        self.M = self.N << 1
        self.logNh = self.logN - 1
        self.Nh = self.N >> 1

        self.total_left_rot_key_map = {}
        self.total_precompute_auto_map = {}
        self.total_inverse_precompute_auto_map = {}

        self.moduliQ_scalar, qRoots = self._init_crt_towers(
            moduli_q,
            roots_q,
        )

        self.rootsQ = np.asarray(qRoots, dtype=np.uint64)
        self.q_mu = self._barrett_mu(self.moduliQ_scalar)
        self.moduliQ = np.asarray(self.moduliQ_scalar, dtype=np.uint64)

        self.moduliP_scalar, pRoots = self._init_crt_towers(
            moduli_p,
            roots_p,
        )
        p_mu = self._barrett_mu(self.moduliP_scalar)

        pHatInvModp, pHatModq, PInvModq, qInvModq = self._init_basis_switch_tables()

        self._init_scaling_factors()
        self._init_encode_params()

        self._init_runtime_workspace(moduli_q, moduli_p)
        self._init_ntt_tables(qRoots, pRoots)
        self._init_modup_tables()
        self._init_moddown_tables(pHatInvModp, pHatModq, PInvModq)
        self._init_rescale_tables(qInvModq)
        self.primes = np.array(self.primes, dtype=np.uint64)

        self._install_eval_mult_key(eval_mult_key)
        self._install_rotation_keys(rotation_keys or ())
        self._build_level_modulus_maps(p_mu)

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
                        _inv_mod(q_hat, modulus)
                    )

        self.PartQlHatModp = [
            [
                [[0 for _ in range(self.L + self.K)] for _ in range(self.alpha)]
                for _ in range(self.dnum)
            ]
            for _ in range(self.L)
        ]
        for level in range(self.L):
            beta = (level + self.alpha) // self.alpha
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
            p_hat_inv_modp[k] = int(_inv_mod(int(p_hat_modp[k]), self.moduliP_scalar[k]))

        p_hat_modq = [[0] * self.L for _ in range(self.K)]
        for k in range(self.K):
            for i in range(self.L):
                p_hat_modq[k][i] = int(1)
                for s in list(range(k)) + list(range(k + 1, self.K)):
                    temp = int(self.moduliP_scalar[s]) % int(self.moduliQ_scalar[i])
                    p_hat_modq[k][i] = _mul_mod(
                        int(p_hat_modq[k][i]),
                        temp,
                        int(self.moduliQ_scalar[i]),
                    )

        self.PModq = [0] * self.L
        for i in range(self.L):
            self.PModq[i] = int(1)
            for k in range(self.K):
                temp = self.moduliP_scalar[k] % self.moduliQ_scalar[i]
                self.PModq[i] = _mul_mod(
                    int(self.PModq[i]),
                    int(temp),
                    int(self.moduliQ_scalar[i]),
                )

        p_inv_modq = [0] * self.L
        for i in range(self.L):
            p_inv_modq[i] = _inv_mod(int(self.PModq[i]), int(self.moduliQ_scalar[i]))

        q_inv_modq = [[0 for _ in range(self.L)] for _ in range(self.L)]
        for i in range(self.L):
            for j in list(range(i)) + list(range(i + 1, self.L)):
                q_inv_modq[i][j] = _inv_mod(
                    int(self.moduliQ_scalar[i]),
                    int(self.moduliQ_scalar[j]),
                )

        self.QlQlInvModqlDivqlModq = [[0] * (self.L - 1) for _ in range(self.L - 1)]
        for k in range(self.L - 1):
            level = self.L - (k + 1)
            for i in range(level):
                ql_inv_mod_ql = int(1)
                for j in range(level):
                    temp = _inv_mod(self.moduliQ_scalar[j], self.moduliQ_scalar[level])
                    ql_inv_mod_ql = _mul_mod(
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
        self.primes = np.hstack((moduli_q, moduli_p))

        self.beta = int((self.L + self.alpha - 1) / self.alpha)
        num_moduli_after_modup = self.max_num_moduli
        num_moduli_after_moddown = self.L
        inner_workspace_numel = (
            16
            * num_moduli_after_modup
            * self.N
            * max(self.beta, 1)
        )
        self.inner_workspace = np.zeros(inner_workspace_numel, dtype=np.uint64)
        self.inner_out = np.zeros(2 * num_moduli_after_modup * self.N, dtype=np.uint64)
        self.moddown_out_ax = np.zeros(num_moduli_after_moddown * self.N, dtype=np.uint64)
        self.moddown_out_bx = np.zeros(num_moduli_after_moddown * self.N, dtype=np.uint64)
        self.modup_out = np.zeros(num_moduli_after_modup * self.N * self.beta, dtype=np.uint64)
        self.rescale_out = np.zeros((self.L - 1) * self.N, dtype=np.uint64)
        self.automorphism_transform_out = np.zeros(num_moduli_after_modup * self.N, dtype=np.uint64)
        self.mod_raise_out = np.zeros(self.L * self.N, dtype=np.uint64)

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
        self.prod_q_i_mod_q_j_modup = np.asarray(prod_q_i_mod_q_j_modup, dtype=np.uint64)

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
                    shoup = _shoup_value(int(hat_inverse_vec[k_idx]), prime)
                    hat_inv_shoup.append(shoup)
                hat_inverse_vec_shoup_modup.append(hat_inv_shoup)
        self.hat_inverse_vec_modup = np.asarray(hat_inverse_vec_modup, dtype=np.uint64)
        self.hat_inverse_vec_shoup_modup = np.asarray(hat_inverse_vec_shoup_modup, dtype=np.uint64)

    def _init_moddown_tables(self, p_hat_inv_modp, p_hat_modq, p_inv_modq):
        q_basis = self.primes[:self.L]

        hat_inv_shoup_moddown = []
        for k in range(self.K):
            prime = self.primes[self.L + k]
            shoup = _shoup_value(int(p_hat_inv_modp[k]), prime)
            hat_inv_shoup_moddown.append(shoup)
        self.hat_inverse_vec_moddown = np.asarray([p_hat_inv_modp], dtype=np.uint64)
        self.hat_inverse_vec_shoup_moddown = np.asarray([hat_inv_shoup_moddown], dtype=np.uint64)

        self.prod_q_i_mod_q_j_moddown = np.asarray(
            [p_hat_modq.swapaxes(1, 0).flatten()],
            dtype=np.uint64,
        )

        prod_shoup = []
        for i, q_prime in enumerate(q_basis):
            prod_shoup.append(_shoup_value(int(p_inv_modq[i]), q_prime))
        self.prod_inv_moddown = np.asarray([p_inv_modq], dtype=np.uint64)
        self.prod_inv_shoup_moddown = np.asarray(prod_shoup, dtype=np.uint64)

    def _init_rescale_tables(self, q_inv_modq):
        qlql_inv_mod_ql_div_ql_mod_q = self.QlQlInvModqlDivqlModq.reshape(-1)
        qlql_inv_vec = []
        qlql_inv_shoup_vec = []
        for i in range(self.L - 1):
            for j in range(self.L - 1):
                value = qlql_inv_mod_ql_div_ql_mod_q[i * (self.L - 1) + j]
                prime = self.primes[j]
                qlql_inv_vec.append(value)
                qlql_inv_shoup_vec.append(_shoup_value(int(value), prime))
        self.qlql_inv_mod_ql_div_ql_mod_q = np.asarray(qlql_inv_vec, dtype=np.uint64)
        self.qlql_inv_mod_ql_div_ql_mod_q_shoup = np.asarray(qlql_inv_shoup_vec, dtype=np.uint64)

        q_inv_modq = q_inv_modq.reshape(-1)
        q_inv_vec = []
        q_inv_shoup_vec = []
        for i in range(self.L):
            for j in range(self.L):
                value = q_inv_modq[i * self.L + j]
                prime = self.primes[j]
                q_inv_vec.append(value)
                q_inv_shoup_vec.append(_shoup_value(int(value), prime))
        self.q_inv_mod_q = np.asarray(q_inv_vec, dtype=np.uint64)
        self.q_inv_mod_q_shoup = np.asarray(q_inv_shoup_vec, dtype=np.uint64)

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
        self.approxSF = 2**self.dcrtBits
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
            log_slot: _slot_bit_reverse_indices(log_slot)
            for log_slot in range(2, self.logN)
        }

    def _install_eval_mult_key(self, mult_swk):
        self.mult_swk_bx = np.ascontiguousarray(mult_swk[0])
        self.mult_swk_ax = np.ascontiguousarray(mult_swk[1])

    def _install_rotation_keys(self, rot_swk_map):
        for rot_idx, bx, ax, auto_map, inverse_auto_map in rot_swk_map:
            bx = np.asarray(bx, dtype=np.uint64)
            ax = np.asarray(ax, dtype=np.uint64)
            auto_map = np.asarray(auto_map, dtype=np.int32)
            inverse_auto_map = np.asarray(inverse_auto_map, dtype=np.int32)
            if bx.shape != ax.shape:
                raise ValueError(f"rotation key {rot_idx} bx/ax shape mismatch: {bx.shape} vs {ax.shape}")
            if bx.ndim != 3 or bx.shape[-1] != self.N:
                raise ValueError(f"rotation key {rot_idx} must be [beta, limbs, N], got {bx.shape}")
            if auto_map.shape != (self.N,):
                raise ValueError(f"rotation key {rot_idx} auto_map must have shape ({self.N},), got {auto_map.shape}")
            if inverse_auto_map.shape != (self.N,):
                raise ValueError(
                    f"rotation key {rot_idx} inverse_auto_map must have shape ({self.N},), got {inverse_auto_map.shape}"
                )
            self.total_left_rot_key_map[int(rot_idx)] = [
                np.ascontiguousarray(bx),
                np.ascontiguousarray(ax),
            ]
            self.total_precompute_auto_map[int(rot_idx)] = np.ascontiguousarray(auto_map)
            self.total_inverse_precompute_auto_map[int(rot_idx)] = np.ascontiguousarray(inverse_auto_map)

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
