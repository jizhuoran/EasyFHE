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


class Context:
    def __init__(self, gpufhe_content_map, config):
        #common params
        self.L = get_item("L", gpufhe_content_map)
        self.dnum = get_item("dnum", gpufhe_content_map)
        self.alpha = get_item("alpha", gpufhe_content_map)
        self.K = get_item("K", gpufhe_content_map)
        self.M = get_item("M", gpufhe_content_map)
        self.N = get_item("N", gpufhe_content_map)
        self.Nh = get_item("Nh", gpufhe_content_map)
        self.approxSF = get_item("approxSF", gpufhe_content_map)
        self.h = get_item("h", gpufhe_content_map)
        self.levelBudget = get_item("levelBudget", gpufhe_content_map)
        self.logN = get_item("logN", gpufhe_content_map)
        self.logNh = get_item("logNh", gpufhe_content_map)
        self.logBsSlots_list = get_item("logBsSlots_list", gpufhe_content_map)
        self.auxModSize = get_item("specialMod", gpufhe_content_map)
        self.rescaleTech = get_item("rescaleTech", gpufhe_content_map)
        self.dcrtBits = get_item("dcrtBits", gpufhe_content_map)
        self.max_num_moduli = get_item("max_num_moduli", gpufhe_content_map)
        self.p = get_item("p", gpufhe_content_map)
        self.secretKeyDist = get_item("secretKeyDist", gpufhe_content_map)
        self.sigma = get_item("sigma", gpufhe_content_map)

        #for common op
        self.primes = get_item("primes", gpufhe_content_map)
        self.barret_k = get_item("barret_k", gpufhe_content_map)
        self.barret_ratio = get_item("barret_ratio", gpufhe_content_map)
        self.dmoduliQ = get_item("dmoduliQ", gpufhe_content_map)
        self.pHatInvModp = get_item("pHatInvModp", gpufhe_content_map)
        self.pHatModp = get_item("pHatModp", gpufhe_content_map)
        self.pHatModq = get_item("pHatModq", gpufhe_content_map)
        self.p_mu = get_item("p_mu", gpufhe_content_map)
        self.q_mu = get_item("q_mu", gpufhe_content_map)
        self.moduliP_scalar = get_item("moduliP_scalar", gpufhe_content_map)
        self.moduliQ_scalar = get_item("moduliQ_scalar", gpufhe_content_map)
        self.moduliQ = get_item("moduliQ", gpufhe_content_map)
        self.scalingFactorsReal = get_item("scalingFactorsReal", gpufhe_content_map)
        self.scalingFactorsRealBig = get_item("scalingFactorsRealBig", gpufhe_content_map)

        #for cv_mul
        self.PModq = get_item("PModq", gpufhe_content_map)
        self.PInvModq = get_item("PInvModq", gpufhe_content_map)
        self.QmuplusPmu_map = get_item("QmuplusPmu_map", gpufhe_content_map)
        self.QplusP_map = get_item("QplusP_map", gpufhe_content_map)

        #output space
        self.automorphism_transform_out = get_item("automorphism_transform_out", gpufhe_content_map)
        self.inner_out = get_item("inner_out", gpufhe_content_map)
        self.moddown_out_ax = get_item("moddown_out_ax", gpufhe_content_map)
        self.moddown_out_bx = get_item("moddown_out_bx", gpufhe_content_map)
        self.modup_out = get_item("modup_out", gpufhe_content_map)
        self.rescale_out = get_item("rescale_out", gpufhe_content_map)
        self.mod_raise_out = get_item("mod_raise_out", gpufhe_content_map)

        #for moddown
        self.hat_inverse_vec_moddown = get_item("hat_inverse_vec_moddown", gpufhe_content_map)
        self.hat_inverse_vec_shoup_moddown = get_item("hat_inverse_vec_shoup_moddown", gpufhe_content_map)
        self.prod_inv_moddown = get_item("prod_inv_moddown", gpufhe_content_map)
        self.prod_inv_shoup_moddown = get_item("prod_inv_shoup_moddown", gpufhe_content_map)
        self.prod_q_i_mod_q_j_moddown = get_item("prod_q_i_mod_q_j_moddown", gpufhe_content_map)

        #for modup
        self.hat_inverse_vec_modup = get_item("hat_inverse_vec_modup", gpufhe_content_map)
        self.hat_inverse_vec_shoup_modup = get_item("hat_inverse_vec_shoup_modup", gpufhe_content_map)
        self.prod_q_i_mod_q_j_modup = get_item("prod_q_i_mod_q_j_modup", gpufhe_content_map)

        #for innerproduct
        self.inner_workspace = get_item("inner_workspace", gpufhe_content_map)
        self.mult_swk_ax = get_item("mult_swk_ax", gpufhe_content_map)
        self.mult_swk_bx = get_item("mult_swk_bx", gpufhe_content_map)

        #for ntt&intt
        self.inverse_power_of_roots_div_two = get_item("inverse_power_of_roots_div_two", gpufhe_content_map)
        self.inverse_scaled_power_of_roots_div_two = get_item("inverse_scaled_power_of_roots_div_two", gpufhe_content_map)
        self.power_of_roots = get_item("power_of_roots", gpufhe_content_map)
        self.power_of_roots_shoup = get_item("power_of_roots_shoup", gpufhe_content_map)

        #for rotation
        self.slots_left_rot_key_map = get_item("slots_left_rot_key_map", gpufhe_content_map)
        self.total_left_rot_key_map = get_item("total_left_rot_key_map", gpufhe_content_map)
        self.slots_precompute_auto_map = get_item("slots_precompute_auto_map", gpufhe_content_map)

        #for cv_drop
        self.qVec = get_item("qVec", gpufhe_content_map)
        self.q_inv_mod_q = get_item("q_inv_mod_q", gpufhe_content_map)
        self.q_inv_mod_q_shoup = get_item("q_inv_mod_q_shoup", gpufhe_content_map)
        self.qlql_inv_mod_ql_div_ql_mod_q = get_item("qlql_inv_mod_ql_div_ql_mod_q", gpufhe_content_map)
        self.qlql_inv_mod_ql_div_ql_mod_q_shoup = get_item("qlql_inv_mod_ql_div_ql_mod_q_shoup", gpufhe_content_map)

        #for encode
        self.QmaxdiffplusPmaxdiff_map = get_item("QmaxdiffplusPmaxdiff_map", gpufhe_content_map)
        self.encode_values = get_item("encode_values", gpufhe_content_map)
        self.QbarretKplusPbarretK_map = get_item("QbarretKplusPbarretK_map", gpufhe_content_map)
        self.QbarretRatioplusPbarretRatio_map = get_item("QbarretRatioplusPbarretRatio_map", gpufhe_content_map)

        #convert all params to tensor
        self.q_mu = torch.tensor(self.q_mu, dtype = torch.uint64)
        self.moduliQ = torch.tensor(self.moduliQ, dtype = torch.uint64)
        self.primes = torch.tensor(self.primes, dtype = torch.uint64)
        self.power_of_roots = torch.tensor(self.power_of_roots, dtype = torch.uint64)
        self.power_of_roots_shoup = torch.tensor(self.power_of_roots_shoup, dtype = torch.uint64)
        self.inverse_power_of_roots_div_two = torch.tensor(self.inverse_power_of_roots_div_two, dtype = torch.uint64)
        self.inverse_scaled_power_of_roots_div_two = torch.tensor(self.inverse_scaled_power_of_roots_div_two, dtype = torch.uint64)
        self.barret_k = torch.tensor(self.barret_k, dtype = torch.uint64)
        self.barret_ratio = torch.tensor(self.barret_ratio, dtype = torch.uint64)
        self.hat_inverse_vec_modup = torch.tensor(self.hat_inverse_vec_modup, dtype = torch.uint64)
        self.hat_inverse_vec_shoup_modup = torch.tensor(self.hat_inverse_vec_shoup_modup, dtype = torch.uint64)
        self.prod_q_i_mod_q_j_modup = torch.tensor(self.prod_q_i_mod_q_j_modup, dtype = torch.uint64)
        self.hat_inverse_vec_moddown = torch.tensor(self.hat_inverse_vec_moddown, dtype = torch.uint64)
        self.hat_inverse_vec_shoup_moddown = torch.tensor(self.hat_inverse_vec_shoup_moddown, dtype = torch.uint64)
        self.prod_q_i_mod_q_j_moddown = torch.tensor(self.prod_q_i_mod_q_j_moddown, dtype = torch.uint64)
        self.prod_inv_moddown = torch.tensor(self.prod_inv_moddown, dtype = torch.uint64)
        self.prod_inv_shoup_moddown = torch.tensor(self.prod_inv_shoup_moddown, dtype = torch.uint64)
        self.qlql_inv_mod_ql_div_ql_mod_q = torch.tensor(self.qlql_inv_mod_ql_div_ql_mod_q, dtype = torch.uint64)
        self.qlql_inv_mod_ql_div_ql_mod_q_shoup = torch.tensor(self.qlql_inv_mod_ql_div_ql_mod_q_shoup, dtype = torch.uint64)
        self.q_inv_mod_q = torch.tensor(self.q_inv_mod_q, dtype = torch.uint64)
        self.q_inv_mod_q_shoup = torch.tensor(self.q_inv_mod_q_shoup, dtype = torch.uint64)
        self.mult_swk_bx = torch.tensor(self.mult_swk_bx, dtype = torch.uint64)
        self.mult_swk_ax = torch.tensor(self.mult_swk_ax, dtype = torch.uint64)
        self.inner_workspace = torch.tensor(self.inner_workspace, dtype = torch.uint64)
        self.inner_out = torch.tensor(self.inner_out, dtype = torch.uint64)
        self.moddown_out_ax = torch.tensor(self.moddown_out_ax, dtype = torch.uint64)
        self.moddown_out_bx = torch.tensor(self.moddown_out_bx, dtype = torch.uint64)
        self.modup_out = torch.tensor(self.modup_out, dtype = torch.uint64)
        self.rescale_out = torch.tensor(self.rescale_out, dtype = torch.uint64)
        self.automorphism_transform_out = torch.tensor(self.automorphism_transform_out, dtype = torch.uint64)
        self.mod_raise_out = torch.tensor(self.mod_raise_out, dtype = torch.uint64)
        self.PModq = torch.tensor(self.PModq, dtype = torch.uint64)
        self.max_int_diffs = torch.tensor([(9223372036854775295 - prime) % prime for prime in self.primes.tolist()], dtype = torch.uint64)

        for key, value in self.QplusP_map.items():
            self.QplusP_map[key] = torch.tensor(value, dtype = torch.uint64)
        for key, value in self.QmuplusPmu_map.items():
            self.QmuplusPmu_map[key] = torch.tensor(value, dtype = torch.uint64)
        for key, value in self.QbarretKplusPbarretK_map.items():
            self.QbarretKplusPbarretK_map[key] = torch.tensor(value, dtype = torch.uint64)
        for key, value in self.QbarretRatioplusPbarretRatio_map.items():
            self.QbarretRatioplusPbarretRatio_map[key] = torch.tensor(value, dtype = torch.uint64)
        for key, value in self.QmaxdiffplusPmaxdiff_map.items():
            self.QmaxdiffplusPmaxdiff_map[key] = torch.tensor(value, dtype = torch.uint64)

        self.left_rot_key_map = {}
        self.precompute_auto_map = {}

        for key_name in self.slots_left_rot_key_map:
            for key in self.slots_left_rot_key_map[str(key_name)]:
                self.left_rot_key_map[key] = [
                    torch.tensor(v, dtype=torch.uint64)
                    for v in self.total_left_rot_key_map[key]
                ]
        for key_name in self.slots_precompute_auto_map:
            for key, value in self.slots_precompute_auto_map[str(key_name)].items():
                self.precompute_auto_map[key] = torch.tensor(
                    value, dtype=torch.int32
                )

        for key, value in self.encode_values.items():
            if isinstance(value, Plaintext):
                self.encode_values[key].cv = [torch.tensor(value.cv, dtype = torch.uint64)]
                Cipher._id_counter = max(Cipher._id_counter, value.cipher_id)
            elif isinstance(value, PreEncodeValues):
                self.encode_values[key].encoded_values = torch.tensor(value.encoded_values)

        self.config = config
        self.inBS = False
        self.in_check_period = False
        self.gpufhe_content_map = gpufhe_content_map
        self.device = None

    
    def cuda(self):
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
        for attr in dir(self):
            if attr.startswith("__") :
                continue
            if(attr not in attrs_to_cuda):
                value = getattr(self, attr)
                new_value = value
                setattr(new_instance, attr, new_value)
        new_instance.device = "cuda"
        new_instance.q_mu = self.q_mu.cuda()
        new_instance.moduliQ = self.moduliQ.cuda()
        new_instance.primes = self.primes.cuda()
        new_instance.power_of_roots = self.power_of_roots.cuda()
        new_instance.power_of_roots_shoup = self.power_of_roots_shoup.cuda()
        new_instance.inverse_power_of_roots_div_two = self.inverse_power_of_roots_div_two.cuda()
        new_instance.inverse_scaled_power_of_roots_div_two = self.inverse_scaled_power_of_roots_div_two.cuda()
        new_instance.barret_k = self.barret_k.cuda()
        new_instance.barret_ratio = self.barret_ratio.cuda()
        new_instance.hat_inverse_vec_modup = self.hat_inverse_vec_modup.cuda()
        new_instance.hat_inverse_vec_shoup_modup = self.hat_inverse_vec_shoup_modup.cuda()
        new_instance.prod_q_i_mod_q_j_modup = self.prod_q_i_mod_q_j_modup.cuda()
        new_instance.hat_inverse_vec_moddown = self.hat_inverse_vec_moddown.cuda()
        new_instance.hat_inverse_vec_shoup_moddown = self.hat_inverse_vec_shoup_moddown.cuda()
        new_instance.prod_q_i_mod_q_j_moddown = self.prod_q_i_mod_q_j_moddown.cuda()
        new_instance.prod_inv_moddown = self.prod_inv_moddown.cuda()
        new_instance.prod_inv_shoup_moddown = self.prod_inv_shoup_moddown.cuda()
        new_instance.qlql_inv_mod_ql_div_ql_mod_q = self.qlql_inv_mod_ql_div_ql_mod_q.cuda()
        new_instance.qlql_inv_mod_ql_div_ql_mod_q_shoup = self.qlql_inv_mod_ql_div_ql_mod_q_shoup.cuda()
        new_instance.q_inv_mod_q = self.q_inv_mod_q.cuda()
        new_instance.q_inv_mod_q_shoup = self.q_inv_mod_q_shoup.cuda()
        new_instance.mult_swk_bx = self.mult_swk_bx.cuda()
        new_instance.mult_swk_ax = self.mult_swk_ax.cuda()
        new_instance.inner_workspace = self.inner_workspace.cuda()
        new_instance.inner_out = self.inner_out.cuda()
        new_instance.moddown_out_ax = self.moddown_out_ax.cuda()
        new_instance.moddown_out_bx = self.moddown_out_bx.cuda()
        new_instance.modup_out = self.modup_out.cuda()
        new_instance.rescale_out = self.rescale_out.cuda()
        new_instance.automorphism_transform_out = self.automorphism_transform_out.cuda()
        new_instance.mod_raise_out = self.mod_raise_out.cuda()
        new_instance.PModq = self.PModq.cuda()
        new_instance.max_int_diffs = self.max_int_diffs.cuda()
        new_instance.QplusP_map = {k: v.cuda() for k, v in self.QplusP_map.items()}
        new_instance.QmuplusPmu_map = {k: v.cuda() for k, v in self.QmuplusPmu_map.items()}
        new_instance.QbarretKplusPbarretK_map = {k: v.cuda() for k, v in self.QbarretKplusPbarretK_map.items()}
        new_instance.QbarretRatioplusPbarretRatio_map = {k: v.cuda() for k, v in self.QbarretRatioplusPbarretRatio_map.items()}
        new_instance.QmaxdiffplusPmaxdiff_map = {k: v.cuda() for k, v in self.QmaxdiffplusPmaxdiff_map.items()}
        new_instance.left_rot_key_map = {k: [v_.cuda() for v_ in v] for k, v in self.left_rot_key_map.items()}
        new_instance.precompute_auto_map = {k: v.cuda() for k, v in self.precompute_auto_map.items()}

        return new_instance
    # move to cpu
    def cpu(self):
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
        for attr in dir(self):
            if attr.startswith("__") :
                continue
            if(attr not in attrs_to_cpu):
                value = getattr(self, attr)
                new_value = value
                setattr(new_instance, attr, new_value)
        new_instance.device = "cpu"
        new_instance.q_mu = self.q_mu.cpu()
        new_instance.moduliQ = self.moduliQ.cpu()
        new_instance.primes = self.primes.cpu()
        new_instance.power_of_roots = self.power_of_roots.cpu()
        new_instance.power_of_roots_shoup = self.power_of_roots_shoup.cpu()
        new_instance.inverse_power_of_roots_div_two = self.inverse_power_of_roots_div_two.cpu()
        new_instance.inverse_scaled_power_of_roots_div_two = self.inverse_scaled_power_of_roots_div_two.cpu()
        new_instance.barret_k = self.barret_k.cpu()
        new_instance.barret_ratio = self.barret_ratio.cpu()
        new_instance.hat_inverse_vec_modup = self.hat_inverse_vec_modup.cpu()
        new_instance.hat_inverse_vec_shoup_modup = self.hat_inverse_vec_shoup_modup.cpu()
        new_instance.prod_q_i_mod_q_j_modup = self.prod_q_i_mod_q_j_modup.cpu()
        new_instance.hat_inverse_vec_moddown = self.hat_inverse_vec_moddown.cpu()
        new_instance.hat_inverse_vec_shoup_moddown = self.hat_inverse_vec_shoup_moddown.cpu()
        new_instance.prod_q_i_mod_q_j_moddown = self.prod_q_i_mod_q_j_moddown.cpu()
        new_instance.prod_inv_moddown = self.prod_inv_moddown.cpu()
        new_instance.prod_inv_shoup_moddown = self.prod_inv_shoup_moddown.cpu()
        new_instance.qlql_inv_mod_ql_div_ql_mod_q = self.qlql_inv_mod_ql_div_ql_mod_q.cpu()
        new_instance.qlql_inv_mod_ql_div_ql_mod_q_shoup = self.qlql_inv_mod_ql_div_ql_mod_q_shoup.cpu()
        new_instance.q_inv_mod_q = self.q_inv_mod_q.cpu()
        new_instance.q_inv_mod_q_shoup = self.q_inv_mod_q_shoup.cpu()
        new_instance.mult_swk_bx = self.mult_swk_bx.cpu()
        new_instance.mult_swk_ax = self.mult_swk_ax.cpu()
        new_instance.inner_workspace = self.inner_workspace.cpu()
        new_instance.inner_out = self.inner_out.cpu()
        new_instance.moddown_out_ax = self.moddown_out_ax.cpu()
        new_instance.moddown_out_bx = self.moddown_out_bx.cpu()
        new_instance.modup_out = self.modup_out.cpu()
        new_instance.rescale_out = self.rescale_out.cpu()
        new_instance.automorphism_transform_out = self.automorphism_transform_out.cpu()
        new_instance.mod_raise_out = self.mod_raise_out.cpu()
        new_instance.PModq = self.PModq.cpu()
        new_instance.max_int_diffs = self.max_int_diffs.cpu()
        new_instance.QplusP_map = {k: v.cpu() for k, v in self.QplusP_map.items()}
        new_instance.QmuplusPmu_map = {k: v.cpu() for k, v in self.QmuplusPmu_map.items()}
        new_instance.QbarretKplusPbarretK_map = {k: v.cpu() for k, v in self.QbarretKplusPbarretK_map.items()}
        new_instance.QbarretRatioplusPbarretRatio_map = {k: v.cpu() for k, v in self.QbarretRatioplusPbarretRatio_map.items()}
        new_instance.QmaxdiffplusPmaxdiff_map = {k: v.cpu() for k, v in self.QmaxdiffplusPmaxdiff_map.items()}
        new_instance.left_rot_key_map = {k: [v_.cpu() for v_ in v] for k, v in self.left_rot_key_map.items()}
        new_instance.precompute_auto_map = {k: v.cpu() for k, v in self.precompute_auto_map.items()}

        new_instance.encode_values = {}
        for key, value in self.encode_values.items():
            if isinstance(value, Plaintext):
                new_instance.encode_values[key] = copy.deepcopy(value)
                new_instance.encode_values[key].cv = [value.cv[0].cpu()]
            elif isinstance(value, PreEncodeValues):
                new_instance.encode_values[key] = copy.deepcopy(value)
                new_instance.encode_values[key].encoded_values = value.encoded_values.cpu()
            else:
                raise TypeError("Unsupported type for encode_values value: {}".format(type(value)))

        if self.config.AUTO_LOAD_KEYS:
            for key, value in new_instance.left_rot_key_map.items():
                new_instance.left_rot_key_map[key] = [v.cpu() for v in value]
            for key, value in new_instance.precompute_auto_map.items():
                new_instance.precompute_auto_map[key] = value.cpu()
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