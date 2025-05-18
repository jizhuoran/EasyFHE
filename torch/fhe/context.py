from enum import Enum
import torch
from .ciphertext import Plaintext, Cipher, PreEncodeValues
from .config import *
import copy
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
    return Context(
        device, config,
        get_item("L", gpufhe_content_map),
        get_item("dnum", gpufhe_content_map),
        get_item("alpha", gpufhe_content_map),
        get_item("K", gpufhe_content_map),
        get_item("M", gpufhe_content_map),
        get_item("N", gpufhe_content_map),
        get_item("Nh", gpufhe_content_map),
        get_item("approxSF", gpufhe_content_map),
        get_item("h", gpufhe_content_map),
        get_item("levelBudget", gpufhe_content_map),
        get_item("logN", gpufhe_content_map),
        get_item("logNh", gpufhe_content_map),
        get_item("logBsSlots_list", gpufhe_content_map),
        get_item("specialMod", gpufhe_content_map),
        get_item("rescaleTech", gpufhe_content_map),
        get_item("dcrtBits", gpufhe_content_map),
        get_item("max_num_moduli", gpufhe_content_map),
        get_item("p", gpufhe_content_map),
        get_item("secretKeyDist", gpufhe_content_map),
        get_item("sigma", gpufhe_content_map),
        False,
        False,
        get_item("primes", gpufhe_content_map),
        get_item("barret_k", gpufhe_content_map),
        get_item("barret_ratio", gpufhe_content_map),
        get_item("dmoduliQ", gpufhe_content_map),
        get_item("pHatInvModp", gpufhe_content_map),
        get_item("pHatModp", gpufhe_content_map),
        get_item("pHatModq", gpufhe_content_map),
        get_item("p_mu", gpufhe_content_map),
        get_item("q_mu", gpufhe_content_map),
        get_item("moduliP_scalar", gpufhe_content_map),
        get_item("moduliQ_scalar", gpufhe_content_map),
        get_item("moduliQ", gpufhe_content_map),
        get_item("scalingFactorsReal", gpufhe_content_map),
        get_item("scalingFactorsRealBig", gpufhe_content_map),
        get_item("PModq", gpufhe_content_map),
        get_item("PInvModq", gpufhe_content_map),
        get_item("QmuplusPmu_map", gpufhe_content_map),
        get_item("QplusP_map", gpufhe_content_map),
        get_item("automorphism_transform_out", gpufhe_content_map),
        get_item("inner_out", gpufhe_content_map),
        get_item("moddown_out_ax", gpufhe_content_map),
        get_item("moddown_out_bx", gpufhe_content_map),
        get_item("modup_out", gpufhe_content_map),
        get_item("rescale_out", gpufhe_content_map),
        get_item("mod_raise_out", gpufhe_content_map),
        get_item("hat_inverse_vec_moddown", gpufhe_content_map),
        get_item("hat_inverse_vec_shoup_moddown", gpufhe_content_map),
        get_item("prod_inv_moddown", gpufhe_content_map),
        get_item("prod_inv_shoup_moddown", gpufhe_content_map),
        get_item("prod_q_i_mod_q_j_moddown", gpufhe_content_map),
        get_item("hat_inverse_vec_modup", gpufhe_content_map),
        get_item("hat_inverse_vec_shoup_modup", gpufhe_content_map),
        get_item("prod_q_i_mod_q_j_modup", gpufhe_content_map),
        get_item("inner_workspace", gpufhe_content_map),
        get_item("mult_swk_ax", gpufhe_content_map),
        get_item("mult_swk_bx", gpufhe_content_map),
        get_item("inverse_power_of_roots_div_two", gpufhe_content_map),
        get_item("inverse_scaled_power_of_roots_div_two", gpufhe_content_map),
        get_item("power_of_roots", gpufhe_content_map),
        get_item("power_of_roots_shoup", gpufhe_content_map),
        get_item("slots_left_rot_key_map", gpufhe_content_map),
        get_item("total_left_rot_key_map", gpufhe_content_map),
        get_item("slots_precompute_auto_map", gpufhe_content_map),
        get_item("qVec", gpufhe_content_map),
        get_item("q_inv_mod_q", gpufhe_content_map),
        get_item("q_inv_mod_q_shoup", gpufhe_content_map),
        get_item("qlql_inv_mod_ql_div_ql_mod_q", gpufhe_content_map),
        get_item("qlql_inv_mod_ql_div_ql_mod_q_shoup", gpufhe_content_map),
        get_item("QmaxdiffplusPmaxdiff_map", gpufhe_content_map),
        get_item("encode_values", gpufhe_content_map),
        get_item("QbarretKplusPbarretK_map", gpufhe_content_map),
        get_item("QbarretRatioplusPbarretRatio_map", gpufhe_content_map)
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
        p,
        secretKeyDist,
        sigma,
        inBS,
        in_check_period,
        primes,
        barret_k,
        barret_ratio,
        dmoduliQ,
        pHatInvModp,
        pHatModp,
        pHatModq,
        p_mu,
        q_mu,
        moduliP_scalar,
        moduliQ_scalar,
        moduliQ,
        scalingFactorsReal,
        scalingFactorsRealBig,
        PModq,
        PInvModq,
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
        slots_left_rot_key_map,
        total_left_rot_key_map,
        slots_precompute_auto_map,
        qVec,
        q_inv_mod_q,
        q_inv_mod_q_shoup,
        qlql_inv_mod_ql_div_ql_mod_q,
        qlql_inv_mod_ql_div_ql_mod_q_shoup,
        QmaxdiffplusPmaxdiff_map,
        encode_values,
        QbarretKplusPbarretK_map,
        QbarretRatioplusPbarretRatio_map,
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
        self.p = p
        self.secretKeyDist = secretKeyDist
        self.sigma = sigma

        # for common op
        self.primes = primes
        self.barret_k = barret_k
        self.barret_ratio = barret_ratio
        self.dmoduliQ = dmoduliQ
        self.pHatInvModp = pHatInvModp
        self.pHatModp = pHatModp
        self.pHatModq = pHatModq
        self.p_mu = p_mu
        self.q_mu = q_mu
        self.moduliP_scalar = moduliP_scalar
        self.moduliQ_scalar = moduliQ_scalar
        self.moduliQ = moduliQ
        self.scalingFactorsReal = scalingFactorsReal
        self.scalingFactorsRealBig = scalingFactorsRealBig

        # for cv_mul
        self.PModq = PModq
        self.PInvModq = PInvModq
        self.QmuplusPmu_map = QmuplusPmu_map
        self.QplusP_map = QplusP_map

        # output space
        self.automorphism_transform_out = automorphism_transform_out
        self.inner_out = inner_out
        self.moddown_out_ax = moddown_out_ax
        self.moddown_out_bx = moddown_out_bx
        self.modup_out = modup_out
        self.rescale_out = rescale_out
        self.mod_raise_out = mod_raise_out

        # for moddown
        self.hat_inverse_vec_moddown = hat_inverse_vec_moddown
        self.hat_inverse_vec_shoup_moddown = hat_inverse_vec_shoup_moddown
        self.prod_inv_moddown = prod_inv_moddown
        self.prod_inv_shoup_moddown = prod_inv_shoup_moddown
        self.prod_q_i_mod_q_j_moddown = prod_q_i_mod_q_j_moddown

        # for modup
        self.hat_inverse_vec_modup = hat_inverse_vec_modup
        self.hat_inverse_vec_shoup_modup = hat_inverse_vec_shoup_modup
        self.prod_q_i_mod_q_j_modup = prod_q_i_mod_q_j_modup

        # for innerproduct
        self.inner_workspace = inner_workspace
        self.mult_swk_ax = mult_swk_ax
        self.mult_swk_bx = mult_swk_bx

        # for ntt&intt
        self.inverse_power_of_roots_div_two = inverse_power_of_roots_div_two
        self.inverse_scaled_power_of_roots_div_two = inverse_scaled_power_of_roots_div_two
        self.power_of_roots = power_of_roots
        self.power_of_roots_shoup = power_of_roots_shoup

        # for rotation
        self.slots_left_rot_key_map = slots_left_rot_key_map
        self.total_left_rot_key_map = total_left_rot_key_map
        self.slots_precompute_auto_map = slots_precompute_auto_map

        # for cv_drop
        self.qVec = qVec
        self.q_inv_mod_q = q_inv_mod_q
        self.q_inv_mod_q_shoup = q_inv_mod_q_shoup
        self.qlql_inv_mod_ql_div_ql_mod_q = qlql_inv_mod_ql_div_ql_mod_q
        self.qlql_inv_mod_ql_div_ql_mod_q_shoup = qlql_inv_mod_ql_div_ql_mod_q_shoup

        # for encode
        self.QmaxdiffplusPmaxdiff_map = QmaxdiffplusPmaxdiff_map
        self.encode_values = encode_values
        self.QbarretKplusPbarretK_map = QbarretKplusPbarretK_map
        self.QbarretRatioplusPbarretRatio_map = QbarretRatioplusPbarretRatio_map

                        
        # convert all params to tensor
        self.q_mu = torch.tensor(self.q_mu, dtype = torch.uint64, device=self.device)
        self.moduliQ = torch.tensor(self.moduliQ, dtype = torch.uint64, device=self.device)
        self.primes = torch.tensor(self.primes, dtype = torch.uint64, device=self.device)
        self.power_of_roots = torch.tensor(self.power_of_roots, dtype = torch.uint64, device=self.device)
        self.power_of_roots_shoup = torch.tensor(self.power_of_roots_shoup, dtype = torch.uint64, device=self.device)
        self.inverse_power_of_roots_div_two = torch.tensor(self.inverse_power_of_roots_div_two, dtype = torch.uint64, device=self.device)
        self.inverse_scaled_power_of_roots_div_two = torch.tensor(self.inverse_scaled_power_of_roots_div_two, dtype = torch.uint64, device=self.device)
        self.barret_k = torch.tensor(self.barret_k, dtype = torch.uint64, device=self.device)
        self.barret_ratio = torch.tensor(self.barret_ratio, dtype = torch.uint64, device=self.device)
        self.hat_inverse_vec_modup = torch.tensor(self.hat_inverse_vec_modup, dtype = torch.uint64, device=self.device)
        self.hat_inverse_vec_shoup_modup = torch.tensor(self.hat_inverse_vec_shoup_modup, dtype = torch.uint64, device=self.device)
        self.prod_q_i_mod_q_j_modup = torch.tensor(self.prod_q_i_mod_q_j_modup, dtype = torch.uint64, device=self.device)
        self.hat_inverse_vec_moddown = torch.tensor(self.hat_inverse_vec_moddown, dtype = torch.uint64, device=self.device)
        self.hat_inverse_vec_shoup_moddown = torch.tensor(self.hat_inverse_vec_shoup_moddown, dtype = torch.uint64, device=self.device)
        self.prod_q_i_mod_q_j_moddown = torch.tensor(self.prod_q_i_mod_q_j_moddown, dtype = torch.uint64, device=self.device)
        self.prod_inv_moddown = torch.tensor(self.prod_inv_moddown, dtype = torch.uint64, device=self.device)
        self.prod_inv_shoup_moddown = torch.tensor(self.prod_inv_shoup_moddown, dtype = torch.uint64, device=self.device)
        self.qlql_inv_mod_ql_div_ql_mod_q = torch.tensor(self.qlql_inv_mod_ql_div_ql_mod_q, dtype = torch.uint64, device=self.device)
        self.qlql_inv_mod_ql_div_ql_mod_q_shoup = torch.tensor(self.qlql_inv_mod_ql_div_ql_mod_q_shoup, dtype = torch.uint64, device=self.device)
        self.q_inv_mod_q = torch.tensor(self.q_inv_mod_q, dtype = torch.uint64, device=self.device)
        self.q_inv_mod_q_shoup = torch.tensor(self.q_inv_mod_q_shoup, dtype = torch.uint64, device=self.device)
        self.mult_swk_bx = torch.tensor(self.mult_swk_bx, dtype = torch.uint64, device=self.device)
        self.mult_swk_ax = torch.tensor(self.mult_swk_ax, dtype = torch.uint64, device=self.device)
        self.inner_workspace = torch.tensor(self.inner_workspace, dtype = torch.uint64, device=self.device)
        self.inner_out = torch.tensor(self.inner_out, dtype = torch.uint64, device=self.device)
        self.moddown_out_ax = torch.tensor(self.moddown_out_ax, dtype = torch.uint64, device=self.device)
        self.moddown_out_bx = torch.tensor(self.moddown_out_bx, dtype = torch.uint64, device=self.device)
        self.modup_out = torch.tensor(self.modup_out, dtype = torch.uint64, device=self.device)
        self.rescale_out = torch.tensor(self.rescale_out, dtype = torch.uint64, device=self.device)
        self.automorphism_transform_out = torch.tensor(self.automorphism_transform_out, dtype = torch.uint64, device=self.device)
        self.mod_raise_out = torch.tensor(self.mod_raise_out, dtype = torch.uint64, device=self.device)
        self.PModq = torch.tensor(self.PModq, dtype = torch.uint64, device=self.device)
        self.max_int_diffs = torch.tensor([(9223372036854775295 - prime) % prime for prime in self.primes.tolist()], dtype = torch.uint64, device=self.device)

        for key, value in self.QplusP_map.items():
            self.QplusP_map[key] = torch.tensor(value, dtype = torch.uint64, device=self.device)
        for key, value in self.QmuplusPmu_map.items():
            self.QmuplusPmu_map[key] = torch.tensor(value, dtype = torch.uint64, device=self.device)
        for key, value in self.QbarretKplusPbarretK_map.items():
            self.QbarretKplusPbarretK_map[key] = torch.tensor(value, dtype = torch.uint64, device=self.device)
        for key, value in self.QbarretRatioplusPbarretRatio_map.items():
            self.QbarretRatioplusPbarretRatio_map[key] = torch.tensor(value, dtype = torch.uint64, device=self.device)
        for key, value in self.QmaxdiffplusPmaxdiff_map.items():
            self.QmaxdiffplusPmaxdiff_map[key] = torch.tensor(value, dtype = torch.uint64, device=self.device)

        self.left_rot_key_map = {}
        self.precompute_auto_map = {}

        for key_name in self.slots_left_rot_key_map:
            for key in self.slots_left_rot_key_map[str(key_name)]:
                self.left_rot_key_map[key] = [
                    torch.tensor(v, dtype=torch.uint64, device=self.device)
                    for v in self.total_left_rot_key_map[key]
                ]
        for key_name in self.slots_precompute_auto_map:
            for key, value in self.slots_precompute_auto_map[str(key_name)].items():
                self.precompute_auto_map[key] = torch.tensor(
                    value, dtype=torch.int32, device=self.device
                )

        for key, value in self.encode_values.items():
            if isinstance(value, Plaintext):
                self.encode_values[key].cv = [torch.tensor(value.cv, dtype = torch.uint64, device=self.device)]
                Cipher._id_counter = max(Cipher._id_counter, value.cipher_id)
            elif isinstance(value, PreEncodeValues):
                self.encode_values[key].encoded_values = torch.tensor(value.encoded_values, device=self.device)

    def cuda(self):
        def recursive_to_cuda(obj):
            if isinstance(obj, dict):  # 处理字典类型
                return {k: recursive_to_cuda(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):  # 处理列表/元组
                return type(obj)(recursive_to_cuda(x) for x in obj)
            elif isinstance(obj, torch.Tensor):  # 处理张量
                return obj.cuda()
            else:  # 其他类型直接返回
                return obj
        new_instance = self.__class__.__new__(self.__class__)
        attrs_to_cuda = ["q_mu", "moduliQ", "primes", "power_of_roots", "power_of_roots_shoup",
                        "inverse_power_of_roots_div_two", "inverse_scaled_power_of_roots_div_two",
                        "barret_k", "barret_ratio", "hat_inverse_vec_modup", "hat_inverse_vec_shoup_modup",
                        "prod_q_i_mod_q_j_modup", "hat_inverse_vec_moddown", "hat_inverse_vec_shoup_moddown",
                        "prod_q_i_mod_q_j_moddown", "prod_inv_moddown", "prod_inv_shoup_moddown",
                        "qlql_inv_mod_ql_div_ql_mod_q", "qlql_inv_mod_ql_div_ql_mod_q_shoup",
                        "q_inv_mod_q", "q_inv_mod_q_shoup", "mult_swk_bx", "mult_swk_ax",
                        "inner_workspace", "inner_out", "moddown_out_ax", "moddown_out_bx",
                        "modup_out", "rescale_out", "automorphism_transform_out", 
                        "mod_raise_out", "PModq", "max_int_diffs","QplusP_map", "QmuplusPmu_map","QbarretKplusPbarretK_map",
                        "QbarretRatioplusPbarretRatio_map", "QmaxdiffplusPmaxdiff_map",
                        "left_rot_key_map", "precompute_auto_map"]
        for attr in self.__dict__:
            value = getattr(self, attr)

            if attr in attrs_to_cuda:
                new_value = recursive_to_cuda(value)
            else:
                new_value = value  # 直接复制非迁移属性

            setattr(new_instance, attr, new_value)

        new_instance.device = "cuda"
        return new_instance
    # move to cpu
    def cpu(self):
        def recursive_to_cpu(obj):
            if isinstance(obj, dict):  # 处理字典
                return {k: recursive_to_cpu(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):  # 处理列表/元组
                return type(obj)(recursive_to_cpu(x) for x in obj)
            elif hasattr(obj, "cpu"):  # 处理张量
                return obj.cpu()
            elif hasattr(obj, "copy"):  # 处理自定义对象
                return obj.copy().cpu()  # 假设自定义对象有 copy() 和 cpu() 方法
            else:
                return obj
        new_instance = self.__class__.__new__(self.__class__)
        attrs_to_cpu = ["q_mu", "moduliQ", "primes", "power_of_roots", "power_of_roots_shoup",
                        "inverse_power_of_roots_div_two", "inverse_scaled_power_of_roots_div_two",
                        "barret_k", "barret_ratio", "hat_inverse_vec_modup", "hat_inverse_vec_shoup_modup",
                        "prod_q_i_mod_q_j_modup", "hat_inverse_vec_moddown", "hat_inverse_vec_shoup_moddown",
                        "prod_q_i_mod_q_j_moddown", "prod_inv_moddown", "prod_inv_shoup_moddown",
                        "qlql_inv_mod_ql_div_ql_mod_q", "qlql_inv_mod_ql_div_ql_mod_q_shoup",
                        "q_inv_mod_q", "q_inv_mod_q_shoup", "mult_swk_bx", "mult_swk_ax",
                        "inner_workspace", "inner_out", "moddown_out_ax", "moddown_out_bx",
                        "modup_out", "rescale_out", "automorphism_transform_out", 
                        "mod_raise_out", "PModq", "max_int_diffs","QplusP_map", "QmuplusPmu_map","QbarretKplusPbarretK_map",
                        "QbarretRatioplusPbarretRatio_map", "QmaxdiffplusPmaxdiff_map",
                        "left_rot_key_map", "precompute_auto_map", "encode_values"]

        new_instance.encode_values = {}
        for key, value in self.encode_values.items():
            if isinstance(value, Plaintext):
                copied = copy.deepcopy(value)
                copied.cv = [recursive_to_cpu(v) for v in value.cv]
                new_instance.encode_values[key] = copied
            elif isinstance(value, PreEncodeValues):
                copied = copy.deepcopy(value)
                copied.encoded_values = recursive_to_cpu(value.encoded_values)
                new_instance.encode_values[key] = copied
            else:
                raise TypeError(f"Unsupported type in encode_values: {type(value)}")
        for attr in self.__dict__:
            value = getattr(self, attr)
            if (attr=="encode_values"):
                continue
            if attr in attrs_to_cpu:
                new_value = recursive_to_cpu(value)
            else:
                new_value = value  # 直接复制非迁移属性

            setattr(new_instance, attr, new_value)

        new_instance.device = "cpu"
        return new_instance
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
            return self.dmoduliQ[l]
        return self.approxSF

    def get_rotation_key(self, rot_index):
        if self.device == "cuda" and not self.left_rot_key_map[rot_index][0].is_cuda:
            return [self.left_rot_key_map[rot_index][0].cuda(), self.left_rot_key_map[rot_index][1].cuda()]
        else:
            return [
                torch.tensor(v, dtype=torch.uint64, device=self.device)
                for v in self.total_left_rot_key_map[rot_index]
            ]

    def get_precompute_auto(self, key):
        if self.device == "cuda" and not self.precompute_auto_map[key].is_cuda:
            return self.precompute_auto_map[key].cuda()
        else:
            return self.precompute_auto_map[key]

    def load_rotation_keys(self, key_name):
        for key in self.slots_left_rot_key_map[str(key_name)]:
            if key not in self.left_rot_key_map:
                self.left_rot_key_map[key] = [
                    torch.tensor(v, dtype=torch.uint64, device=self.device) # fixme:??
                    for v in self.total_left_rot_key_map[key]
                ]
        for key, value in self.slots_precompute_auto_map[str(key_name)].items():
            self.precompute_auto_map[key] = torch.tensor(
                value, dtype=torch.int32, device=self.device
            )
