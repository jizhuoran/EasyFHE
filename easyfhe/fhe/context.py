from enum import Enum
import easyfhe as torch
from .ciphertext import Plaintext, Cipher, PreEncodeValues
from .config import *
import numpy as np
def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    return f"{message}\n"


class LargeScalingFactorConstants(Enum):
    MAX_BITS_IN_WORD = 61
    MAX_LOG_STEP = 60

def get_item(item_name, content_map):
    if item_name in content_map:
        return content_map[item_name]
    return None

def parse_content_map(gpufhe_content_map, device, config):
    L = get_item("L", gpufhe_content_map)
    dnum = get_item("dnum", gpufhe_content_map)
    alpha = get_item("alpha", gpufhe_content_map)
    K = get_item("K", gpufhe_content_map)
    M = get_item("M", gpufhe_content_map)
    N = get_item("N", gpufhe_content_map)
    Nh = get_item("Nh", gpufhe_content_map)
    approxSF = get_item("approxSF", gpufhe_content_map)
    h = get_item("h", gpufhe_content_map)
    levelBudget = get_item("levelBudget", gpufhe_content_map)
    logN = get_item("logN", gpufhe_content_map)
    logNh = get_item("logNh", gpufhe_content_map)
    logBsSlots_list = get_item("logBsSlots_list", gpufhe_content_map)
    specialMod = get_item("specialMod", gpufhe_content_map)
    rescaleTech = get_item("rescaleTech", gpufhe_content_map)
    dcrtBits = get_item("dcrtBits", gpufhe_content_map)
    max_num_moduli = get_item("max_num_moduli", gpufhe_content_map)
    secretKeyDist = get_item("secretKeyDist", gpufhe_content_map)
    sigma = get_item("sigma", gpufhe_content_map)
    primes = get_item("primes", gpufhe_content_map)
    barret_k = get_item("barret_k", gpufhe_content_map)
    barret_ratio = get_item("barret_ratio", gpufhe_content_map)
    q_mu = get_item("q_mu", gpufhe_content_map)
    moduliP_scalar = get_item("moduliP_scalar", gpufhe_content_map)
    moduliQ_scalar = get_item("moduliQ_scalar", gpufhe_content_map)
    moduliQ = get_item("moduliQ", gpufhe_content_map)
    scalingFactorsReal = get_item("scalingFactorsReal", gpufhe_content_map)
    scalingFactorsRealBig = get_item("scalingFactorsRealBig", gpufhe_content_map)
    PModq = get_item("PModq", gpufhe_content_map)
    QmuplusPmu_map = get_item("QmuplusPmu_map", gpufhe_content_map)
    QplusP_map = get_item("QplusP_map", gpufhe_content_map)
    automorphism_transform_out = get_item("automorphism_transform_out", gpufhe_content_map)
    inner_out = get_item("inner_out", gpufhe_content_map)
    moddown_out_ax = get_item("moddown_out_ax", gpufhe_content_map)
    moddown_out_bx = get_item("moddown_out_bx", gpufhe_content_map)
    modup_out = get_item("modup_out", gpufhe_content_map)
    rescale_out = get_item("rescale_out", gpufhe_content_map)
    mod_raise_out = get_item("mod_raise_out", gpufhe_content_map)
    hat_inverse_vec_moddown = get_item("hat_inverse_vec_moddown", gpufhe_content_map)
    hat_inverse_vec_shoup_moddown = get_item("hat_inverse_vec_shoup_moddown", gpufhe_content_map)
    prod_inv_moddown = get_item("prod_inv_moddown", gpufhe_content_map)
    prod_inv_shoup_moddown = get_item("prod_inv_shoup_moddown", gpufhe_content_map)
    prod_q_i_mod_q_j_moddown = get_item("prod_q_i_mod_q_j_moddown", gpufhe_content_map)
    hat_inverse_vec_modup = get_item("hat_inverse_vec_modup", gpufhe_content_map)
    hat_inverse_vec_shoup_modup = get_item("hat_inverse_vec_shoup_modup", gpufhe_content_map)
    prod_q_i_mod_q_j_modup = get_item("prod_q_i_mod_q_j_modup", gpufhe_content_map)
    inner_workspace = get_item("inner_workspace", gpufhe_content_map)
    mult_swk_ax = get_item("mult_swk_ax", gpufhe_content_map)
    mult_swk_bx = get_item("mult_swk_bx", gpufhe_content_map)
    inverse_power_of_roots_div_two = get_item("inverse_power_of_roots_div_two", gpufhe_content_map)
    inverse_scaled_power_of_roots_div_two = get_item("inverse_scaled_power_of_roots_div_two", gpufhe_content_map)
    power_of_roots = get_item("power_of_roots", gpufhe_content_map)
    power_of_roots_shoup = get_item("power_of_roots_shoup", gpufhe_content_map)
    total_left_rot_key_map = get_item("total_left_rot_key_map", gpufhe_content_map)
    total_precompute_auto_map = get_item("total_precompute_auto_map", gpufhe_content_map)
    q_inv_mod_q = get_item("q_inv_mod_q", gpufhe_content_map)
    q_inv_mod_q_shoup = get_item("q_inv_mod_q_shoup", gpufhe_content_map)
    qlql_inv_mod_ql_div_ql_mod_q = get_item("qlql_inv_mod_ql_div_ql_mod_q", gpufhe_content_map)
    qlql_inv_mod_ql_div_ql_mod_q_shoup = get_item("qlql_inv_mod_ql_div_ql_mod_q_shoup", gpufhe_content_map)
    QmaxdiffplusPmaxdiff_map = get_item("QmaxdiffplusPmaxdiff_map", gpufhe_content_map)
    encode_values = get_item("encode_values", gpufhe_content_map)
    QbarretKplusPbarretK_map = get_item("QbarretKplusPbarretK_map", gpufhe_content_map)
    QbarretRatioplusPbarretRatio_map = get_item("QbarretRatioplusPbarretRatio_map", gpufhe_content_map)

    encode_params_ksiPows = get_item("encode_params_ksiPows", gpufhe_content_map)
    encode_params_rotGroup = get_item("encode_params_rotGroup", gpufhe_content_map)
    encode_bitrev_indices = get_item("encode_bitrev_indices", gpufhe_content_map)

    q_mu = torch.tensor(q_mu, dtype = torch.uint64)
    moduliQ = torch.tensor(moduliQ, dtype = torch.uint64)
    primes = torch.tensor(primes, dtype = torch.uint64)
    power_of_roots = torch.tensor(power_of_roots, dtype = torch.uint64)
    power_of_roots_shoup = torch.tensor(power_of_roots_shoup, dtype = torch.uint64)
    inverse_power_of_roots_div_two = torch.tensor(inverse_power_of_roots_div_two, dtype = torch.uint64)
    inverse_scaled_power_of_roots_div_two = torch.tensor(inverse_scaled_power_of_roots_div_two, dtype = torch.uint64)
    barret_k = torch.tensor(barret_k, dtype = torch.uint64)
    barret_ratio = torch.tensor(barret_ratio, dtype = torch.uint64)
    hat_inverse_vec_modup = torch.tensor(hat_inverse_vec_modup, dtype = torch.uint64)
    hat_inverse_vec_shoup_modup = torch.tensor(hat_inverse_vec_shoup_modup, dtype = torch.uint64)
    prod_q_i_mod_q_j_modup = torch.tensor(prod_q_i_mod_q_j_modup, dtype = torch.uint64)
    hat_inverse_vec_moddown = torch.tensor(hat_inverse_vec_moddown, dtype = torch.uint64)
    hat_inverse_vec_shoup_moddown = torch.tensor(hat_inverse_vec_shoup_moddown, dtype = torch.uint64)
    prod_q_i_mod_q_j_moddown = torch.tensor(prod_q_i_mod_q_j_moddown, dtype = torch.uint64)
    prod_inv_moddown = torch.tensor(prod_inv_moddown, dtype = torch.uint64)
    prod_inv_shoup_moddown = torch.tensor(prod_inv_shoup_moddown, dtype = torch.uint64)
    qlql_inv_mod_ql_div_ql_mod_q = torch.tensor(qlql_inv_mod_ql_div_ql_mod_q, dtype = torch.uint64)
    qlql_inv_mod_ql_div_ql_mod_q_shoup = torch.tensor(qlql_inv_mod_ql_div_ql_mod_q_shoup, dtype = torch.uint64)
    q_inv_mod_q = torch.tensor(q_inv_mod_q, dtype = torch.uint64)
    q_inv_mod_q_shoup = torch.tensor(q_inv_mod_q_shoup, dtype = torch.uint64)
    mult_swk_bx = torch.tensor(mult_swk_bx, dtype = torch.uint64)
    mult_swk_ax = torch.tensor(mult_swk_ax, dtype = torch.uint64)
    inner_workspace = torch.tensor(inner_workspace, dtype = torch.uint64)
    inner_out = torch.tensor(inner_out, dtype = torch.uint64)
    moddown_out_ax = torch.tensor(moddown_out_ax, dtype = torch.uint64)
    moddown_out_bx = torch.tensor(moddown_out_bx, dtype = torch.uint64)
    modup_out = torch.tensor(modup_out, dtype = torch.uint64)
    rescale_out = torch.tensor(rescale_out, dtype = torch.uint64)
    automorphism_transform_out = torch.tensor(automorphism_transform_out, dtype = torch.uint64)
    mod_raise_out = torch.tensor(mod_raise_out, dtype = torch.uint64)
    PModq = torch.tensor(PModq, dtype = torch.uint64)
    max_int_diffs = torch.tensor([(9223372036854775295 - prime) % prime for prime in primes.tolist()], dtype = torch.uint64)

    encode_params_rotGroup = torch.tensor(encode_params_rotGroup, dtype=torch.uint32)
    encode_params_ksiPows = torch.tensor(encode_params_ksiPows, dtype=torch.complex128)
    for key, value in encode_bitrev_indices.items():
        encode_bitrev_indices[key] = torch.tensor(value, dtype=torch.uint32)


    for key, value in QplusP_map.items():
        QplusP_map[key] = torch.tensor(value, dtype = torch.uint64)
    for key, value in QmuplusPmu_map.items():
        QmuplusPmu_map[key] = torch.tensor(value, dtype = torch.uint64)
    for key, value in QbarretKplusPbarretK_map.items():
        QbarretKplusPbarretK_map[key] = torch.tensor(value, dtype = torch.uint64)
    for key, value in QbarretRatioplusPbarretRatio_map.items():
        QbarretRatioplusPbarretRatio_map[key] = torch.tensor(value, dtype = torch.uint64)
    for key, value in QmaxdiffplusPmaxdiff_map.items():
        QmaxdiffplusPmaxdiff_map[key] = torch.tensor(value, dtype = torch.uint64)

    left_rot_key_map = {
        int(rotIdx): [
            torch.tensor(key_pair[0], dtype=torch.uint64),
            torch.tensor(key_pair[1], dtype=torch.uint64),
        ]
        for rotIdx, key_pair in total_left_rot_key_map.items()
    }

    precompute_auto_map = {
        int(rotIdx): torch.tensor(auto_map, dtype=torch.int32)
        for rotIdx, auto_map in total_precompute_auto_map.items()
    }

    for key, value in encode_values.items():
        if isinstance(value, Plaintext):
            encode_values[key].cv = [torch.tensor(value.cv, dtype = torch.uint64)]
            Cipher._id_counter = max(Cipher._id_counter, value.cipher_id)
        elif isinstance(value, PreEncodeValues):
            encode_values[key].encoded_values = torch.tensor(value.encoded_values)


    return Context(
        device, config,
        L,
        dnum,
        alpha,
        K,
        M,
        N,
        Nh,
        approxSF,
        h,
        levelBudget,
        logN,
        logNh,
        logBsSlots_list,
        specialMod,
        rescaleTech,
        dcrtBits,
        max_num_moduli,
        secretKeyDist,
        sigma,
        False,
        False,
        primes,
        barret_k,
        barret_ratio,
        q_mu,
        moduliP_scalar,
        moduliQ_scalar,
        moduliQ,
        scalingFactorsReal,
        scalingFactorsRealBig,
        PModq,
        max_int_diffs,
        QmuplusPmu_map,
        QplusP_map,
        automorphism_transform_out,
        inner_out,
        moddown_out_ax,
        moddown_out_bx,
        modup_out,
        rescale_out,
        mod_raise_out,
        hat_inverse_vec_moddown,
        hat_inverse_vec_shoup_moddown,
        prod_inv_moddown,
        prod_inv_shoup_moddown,
        prod_q_i_mod_q_j_moddown,
        hat_inverse_vec_modup,
        hat_inverse_vec_shoup_modup,
        prod_q_i_mod_q_j_modup,
        inner_workspace,
        mult_swk_ax,
        mult_swk_bx,
        inverse_power_of_roots_div_two,
        inverse_scaled_power_of_roots_div_two,
        power_of_roots,
        power_of_roots_shoup,
        left_rot_key_map,
        precompute_auto_map,
        q_inv_mod_q,
        q_inv_mod_q_shoup,
        qlql_inv_mod_ql_div_ql_mod_q,
        qlql_inv_mod_ql_div_ql_mod_q_shoup,
        QmaxdiffplusPmaxdiff_map,
        encode_params_ksiPows,
        encode_params_rotGroup,
        encode_bitrev_indices,
        encode_values,
        QbarretKplusPbarretK_map,
        QbarretRatioplusPbarretRatio_map,
    )


class Context:
    def __init__(
        self,
        device,
        config,
        L,
        dnum,
        alpha,
        K,
        M,
        N,
        Nh,
        approxSF,
        h,
        levelBudget,
        logN,
        logNh,
        logBsSlots_list,
        auxModSize,
        rescaleTech,
        dcrtBits,
        max_num_moduli,
        secretKeyDist,
        sigma,
        inBS,
        in_check_period,
        primes,
        barret_k,
        barret_ratio,
        q_mu,
        moduliP_scalar,
        moduliQ_scalar,
        moduliQ,
        scalingFactorsReal,
        scalingFactorsRealBig,
        PModq,
        max_int_diffs,
        QmuplusPmu_map,
        QplusP_map,
        automorphism_transform_out,
        inner_out,
        moddown_out_ax,
        moddown_out_bx,
        modup_out,
        rescale_out,
        mod_raise_out,
        hat_inverse_vec_moddown,
        hat_inverse_vec_shoup_moddown,
        prod_inv_moddown,
        prod_inv_shoup_moddown,
        prod_q_i_mod_q_j_moddown,
        hat_inverse_vec_modup,
        hat_inverse_vec_shoup_modup,
        prod_q_i_mod_q_j_modup,
        inner_workspace,
        mult_swk_ax,
        mult_swk_bx,
        inverse_power_of_roots_div_two,
        inverse_scaled_power_of_roots_div_two,
        power_of_roots,
        power_of_roots_shoup,
        left_rot_key_map,
        precompute_auto_map,
        q_inv_mod_q,
        q_inv_mod_q_shoup,
        qlql_inv_mod_ql_div_ql_mod_q,
        qlql_inv_mod_ql_div_ql_mod_q_shoup,
        QmaxdiffplusPmaxdiff_map,
        encode_params_ksiPows,
        encode_params_rotGroup,
        encode_bitrev_indices,
        encode_values,
        QbarretKplusPbarretK_map,
        QbarretRatioplusPbarretRatio_map
    ):

        #  self, gpufhe_content_map, config):

        self.device = device
        self.config = config
        self.inBS = inBS
        self.in_check_period = in_check_period

        # common params
        self.L = L
        self.dnum = dnum
        self.alpha = alpha
        self.K = K
        self.M = M
        self.N = N
        self.Nh = Nh
        self.approxSF = approxSF
        self.h = h
        self.levelBudget = levelBudget
        self.logN = logN
        self.logNh = logNh
        self.logBsSlots_list = logBsSlots_list
        self.auxModSize = auxModSize
        self.rescaleTech = rescaleTech
        self.dcrtBits = dcrtBits
        self.max_num_moduli = max_num_moduli
        self.secretKeyDist = secretKeyDist
        self.sigma = sigma

        # for common op
        self.primes = primes.clone().to(device)
        self.barret_k = barret_k.clone().to(device)
        self.barret_ratio = barret_ratio.clone().to(device)
        self.primes_list = [int(p) for p in primes.tolist()]

        # Precompute switch_modulus_map for mod_raise and drop_last_element_and_scale
        num_primes = len(self.primes_list)
        switch_map = []
        for old_idx in range(num_primes):
            old_mod = self.primes_list[old_idx]
            for new_idx in range(num_primes):
                new_mod = self.primes_list[new_idx]
                if old_mod > new_mod:
                    diff = new_mod - (old_mod % new_mod)
                else:
                    diff = new_mod - old_mod
                switch_map.append(diff)
        self.switch_modulus_map = torch.tensor(switch_map, dtype=torch.uint64, device=device)
        self.q_mu = q_mu.clone().to(device)
        self.moduliP_scalar = moduliP_scalar # for computation on cpu
        self.moduliQ_scalar = moduliQ_scalar # for computation on cpu
        self.moduliQ = moduliQ.clone().to(device)
        self.scalingFactorsReal = scalingFactorsReal
        self.scalingFactorsRealBig = scalingFactorsRealBig

        # for cv_mul
        self.PModq = PModq.clone().to(device)
        self.QmuplusPmu_map = {key: value.clone().to(device) for key, value in QmuplusPmu_map.items()}
        self.QplusP_map =  {key: value.clone().to(device) for key, value in QplusP_map.items()}

        # output space
        self.automorphism_transform_out = automorphism_transform_out.clone().to(device)
        self.inner_out = inner_out.clone().to(device)
        self.moddown_out_ax = moddown_out_ax.clone().to(device)
        self.moddown_out_bx = moddown_out_bx.clone().to(device)
        self.modup_out = modup_out.clone().to(device)
        self.rescale_out = rescale_out.clone().to(device)
        self.mod_raise_out = mod_raise_out.clone().to(device)

        # for moddown
        self.hat_inverse_vec_moddown = hat_inverse_vec_moddown.clone().to(device)
        self.hat_inverse_vec_shoup_moddown = hat_inverse_vec_shoup_moddown.clone().to(device)
        self.prod_inv_moddown = prod_inv_moddown.clone().to(device)
        self.prod_inv_shoup_moddown = prod_inv_shoup_moddown.clone().to(device)
        self.prod_q_i_mod_q_j_moddown = prod_q_i_mod_q_j_moddown.clone().to(device)

        # for modup
        self.hat_inverse_vec_modup = hat_inverse_vec_modup.clone().to(device)
        self.hat_inverse_vec_shoup_modup = hat_inverse_vec_shoup_modup.clone().to(device)
        self.prod_q_i_mod_q_j_modup = prod_q_i_mod_q_j_modup.clone().to(device)

        # for innerproduct
        self.inner_workspace = inner_workspace.clone().to(device)
        self.mult_swk_ax = mult_swk_ax.clone().to(device)
        self.mult_swk_bx = mult_swk_bx.clone().to(device)

        # for ntt&intt
        self.inverse_power_of_roots_div_two = inverse_power_of_roots_div_two.clone().to(device)
        self.inverse_scaled_power_of_roots_div_two = inverse_scaled_power_of_roots_div_two.clone().to(device)
        self.power_of_roots = power_of_roots.clone().to(device)
        self.power_of_roots_shoup = power_of_roots_shoup.clone().to(device)

        # for rotation
        self.left_rot_key_map = {key: [value[0].clone().to("cpu"), value[1].clone().to("cpu")] for key, value in left_rot_key_map.items()}
        self.precompute_auto_map = {key: value.clone().to("cpu") for key, value in precompute_auto_map.items()}

        # for cv_drop
        self.q_inv_mod_q = q_inv_mod_q.clone().to(device)
        self.q_inv_mod_q_shoup = q_inv_mod_q_shoup.clone().to(device)
        self.qlql_inv_mod_ql_div_ql_mod_q = qlql_inv_mod_ql_div_ql_mod_q.clone().to(device)
        self.qlql_inv_mod_ql_div_ql_mod_q_shoup = qlql_inv_mod_ql_div_ql_mod_q_shoup.clone().to(device)

        # for encode
        self.QmaxdiffplusPmaxdiff_map = {key: value.clone().to(device) for key, value in QmaxdiffplusPmaxdiff_map.items()}
        self.QbarretKplusPbarretK_map = {key: value.clone().to(device) for key, value in QbarretKplusPbarretK_map.items()}
        self.QbarretRatioplusPbarretRatio_map = {key: value.clone().to(device) for key, value in QbarretRatioplusPbarretRatio_map.items()}
        self.max_int_diffs = max_int_diffs.clone().to(device)
        self.encode_params_ksiPows = encode_params_ksiPows
        self.encode_params_rotGroup = encode_params_rotGroup
        self.encode_bitrev_indices = encode_bitrev_indices
        self.encode_values = {}
        for key, value in encode_values.items():
            if isinstance(value, Plaintext):
                self.encode_values[key] = value.deep_copy()
                self.encode_values[key].cv = [cv.to(device) for cv in self.encode_values[key].cv]
            elif isinstance(value, PreEncodeValues):
                self.encode_values[key] = value.deep_copy()
                self.encode_values[key].encoded_values = self.encode_values[key].encoded_values.to(device)
            else:
                raise ValueError("Unknown type in encode_values")
            
        if config.AUTO_LOAD_KEYS and device == "cuda":
            for key, value in self.left_rot_key_map.items():
                self.left_rot_key_map[key] = [
                    value[0].cuda(), value[1].cuda()
                ]
            for key, value in self.precompute_auto_map.items():
                self.precompute_auto_map[key] = value.cuda()

    def construct_copy(self, device):
        return Context(
            device,
            self.config,
            self.L,
            self.dnum,
            self.alpha,
            self.K,
            self.M,
            self.N,
            self.Nh,
            self.approxSF,
            self.h,
            self.levelBudget,
            self.logN,
            self.logNh,
            self.logBsSlots_list,
            self.auxModSize,
            self.rescaleTech,
            self.dcrtBits,
            self.max_num_moduli,
            self.secretKeyDist,
            self.sigma,
            self.inBS,
            self.in_check_period,
            self.primes,
            self.barret_k,
            self.barret_ratio,
            self.q_mu,
            self.moduliP_scalar,
            self.moduliQ_scalar,
            self.moduliQ,
            self.scalingFactorsReal,
            self.scalingFactorsRealBig,
            self.PModq,
            self.max_int_diffs,
            self.QmuplusPmu_map,
            self.QplusP_map,
            self.automorphism_transform_out,
            self.inner_out,
            self.moddown_out_ax,
            self.moddown_out_bx,
            self.modup_out,
            self.rescale_out,
            self.mod_raise_out,
            self.hat_inverse_vec_moddown,
            self.hat_inverse_vec_shoup_moddown,
            self.prod_inv_moddown,
            self.prod_inv_shoup_moddown,
            self.prod_q_i_mod_q_j_moddown,
            self.hat_inverse_vec_modup,
            self.hat_inverse_vec_shoup_modup,
            self.prod_q_i_mod_q_j_modup,
            self.inner_workspace,
            self.mult_swk_ax,
            self.mult_swk_bx,
            self.inverse_power_of_roots_div_two,
            self.inverse_scaled_power_of_roots_div_two,
            self.power_of_roots,
            self.power_of_roots_shoup,
            self.left_rot_key_map,
            self.precompute_auto_map,
            self.q_inv_mod_q,
            self.q_inv_mod_q_shoup,
            self.qlql_inv_mod_ql_div_ql_mod_q,
            self.qlql_inv_mod_ql_div_ql_mod_q_shoup,
            self.QmaxdiffplusPmaxdiff_map,
            self.encode_params_ksiPows,
            self.encode_params_rotGroup,
            self.encode_bitrev_indices,
            self.encode_values,
            self.QbarretKplusPbarretK_map,
            self.QbarretRatioplusPbarretRatio_map
        )

    def cuda(self):
        return self.construct_copy("cuda")

    def cpu(self):
        return self.construct_copy("cpu")
    
    def norm_rot_index(self, i):
        if i < 0:
            i = self.N // 2 + i
        return i

    #  Method to retrieve the scaling factor of level l.
    #  For FIXEDMANUAL scaling technique method always returns 2^p, where p corresponds to plaintext modulus
    #  @param l For FLEXIBLEAUTO scaling technique the level whose scaling factor we want to learn.
    #  Levels start from 0 (no scaling done - all towers) and go up to K-1, where K is the number of towers supported.
    #  @return the scaling factor.
    def GetScalingFactorReal(self, cur_limbs= None):
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
        l = cur_limbs
        if self.rescaleTech == "FLEXIBLEAUTO" or self.rescaleTech == "FLEXIBLEAUTOEXT":
            return float(self.moduliQ_scalar[l]) # Moduli as real
        return self.approxSF

    def get_rotation_key(self, rot_index):
        if self.device == "cuda" and not self.left_rot_key_map[rot_index][0].is_cuda:
            return [self.left_rot_key_map[rot_index][0].cuda(), self.left_rot_key_map[rot_index][1].cuda()]
        else:
            return self.left_rot_key_map[rot_index]

    def get_precompute_auto(self, key):
        if self.device == "cuda" and not self.precompute_auto_map[key].is_cuda:
            return self.precompute_auto_map[key].cuda()
        else:
            return self.precompute_auto_map[key]


    def __repr__(self):
        s = []
        s.append(f"{'L:':20} {self.L}")
        s.append(f"{'logBsSlots:':20} {self.logBsSlots_list}")
        s.append(f"{'N:':20} {self.N}")
        s.append(f"{'dnum:':20} {self.dnum}")
        s.append(f"{'dcrtBits:':20} {self.dcrtBits}")
        # s.append(f"{'firstMod:':20} {self.firstMod}")  # todo: to add
        s.append(f"{'K:':20} {self.K}")
        s.append(f"{'levelBudget_list:':20} {self.levelBudget}")
        s.append(f"{'rescaleTech:':20} {self.rescaleTech}")
        s.append(f"{'secretKeyDist:':20} {self.secretKeyDist}")
        s.append(f"{'device:':20} {self.device}")
        return "<Context>\n" + "\n".join(s)

