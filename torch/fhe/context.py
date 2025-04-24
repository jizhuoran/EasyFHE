from enum import Enum
from .bs_context import *
from .config import *

def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    return f"{message}\n"


class LargeScalingFactorConstants(Enum):
    MAX_BITS_IN_WORD = 61
    MAX_LOG_STEP = 60


class Context:
    def __init__(self, BsContext_content_map, gpufhe_content_map, config):
        self.L = get_item("L", gpufhe_content_map)
        self.dnum = get_item("dnum", gpufhe_content_map)
        self.alpha = get_item("alpha", gpufhe_content_map)
        self.K = get_item("K", gpufhe_content_map)
        self.M = get_item("M", gpufhe_content_map)
        self.N = get_item("N", gpufhe_content_map)
        self.Nh = get_item("Nh", gpufhe_content_map)
        self.PInvModq = get_item("PInvModq", gpufhe_content_map)
        self.PModq = get_item("PModq", gpufhe_content_map)
        self.PartQlHatInvModq = get_item("PartQlHatInvModq", gpufhe_content_map)
        self.PartQlHatModp = get_item("PartQlHatModp", gpufhe_content_map)
        self.QlQlInvModqlDivqlModq = get_item("QlQlInvModqlDivqlModq", gpufhe_content_map)
        self.approxSF = get_item("approxSF", gpufhe_content_map)
        self.automorphism_transform_out = get_item("automorphism_transform_out", gpufhe_content_map)
        self.barret_k = get_item("barret_k", gpufhe_content_map)
        self.barret_ratio = get_item("barret_ratio", gpufhe_content_map)
        self.beta = get_item("beta", gpufhe_content_map)
        self.chain_length = get_item("chain_length", gpufhe_content_map)
        self.dmoduliQ = get_item("dmoduliQ", gpufhe_content_map)
        self.h = get_item("h", gpufhe_content_map)
        self.hat_inverse_vec_moddown = get_item("hat_inverse_vec_moddown", gpufhe_content_map)
        self.hat_inverse_vec_modup = get_item("hat_inverse_vec_modup", gpufhe_content_map)
        self.hat_inverse_vec_shoup_moddown = get_item("hat_inverse_vec_shoup_moddown", gpufhe_content_map)
        self.hat_inverse_vec_shoup_modup = get_item("hat_inverse_vec_shoup_modup", gpufhe_content_map)
        self.inner_out = get_item("inner_out", gpufhe_content_map)
        self.inner_workspace = get_item("inner_workspace", gpufhe_content_map)
        self.inverse_power_of_roots_div_two = get_item("inverse_power_of_roots_div_two", gpufhe_content_map)
        self.inverse_scaled_power_of_roots_div_two = get_item("inverse_scaled_power_of_roots_div_two", gpufhe_content_map)
        self.levelBudget = get_item("levelBudget", gpufhe_content_map)
        self.logN = get_item("logN", gpufhe_content_map)
        self.logNh = get_item("logNh", gpufhe_content_map)
        self.logBsSlots_list = get_item("logBsSlots_list", gpufhe_content_map)
        self.auxModSize = get_item("specialMod", gpufhe_content_map)
        self.dcrtBits = get_item("dcrtBits", gpufhe_content_map)
        self.max_num_moduli = get_item("max_num_moduli", gpufhe_content_map)
        self.moddown_out_ax = get_item("moddown_out_ax", gpufhe_content_map)
        self.moddown_out_bx = get_item("moddown_out_bx", gpufhe_content_map)
        self.moduliP_scalar = get_item("moduliP_scalar", gpufhe_content_map)
        self.moduliQ_scalar = get_item("moduliQ_scalar", gpufhe_content_map)
        self.moduliQ = get_item("moduliQ", gpufhe_content_map)
        self.modup_out = get_item("modup_out", gpufhe_content_map)
        self.mult_swk = get_item("mult_swk", gpufhe_content_map)
        self.num_moduli_after_moddown = get_item("num_moduli_after_moddown", gpufhe_content_map)
        self.num_moduli_after_modup = get_item("num_moduli_after_modup", gpufhe_content_map)
        self.num_special_moduli = get_item("num_special_moduli", gpufhe_content_map)
        self.p = get_item("p", gpufhe_content_map)
        self.pHatInvModp = get_item("pHatInvModp", gpufhe_content_map)
        self.pHatModp = get_item("pHatModp", gpufhe_content_map)
        self.pHatModq = get_item("pHatModq", gpufhe_content_map)
        self.p_mu = get_item("p_mu", gpufhe_content_map)
        self.power_of_roots = get_item("power_of_roots", gpufhe_content_map)
        self.power_of_roots_shoup = get_item("power_of_roots_shoup", gpufhe_content_map)
        self.power_of_roots_shoup_vec = get_item("power_of_roots_shoup_vec", gpufhe_content_map)
        self.power_of_roots_vec = get_item("power_of_roots_vec", gpufhe_content_map)
        self.mult_key_map = get_item("mult_key_map", gpufhe_content_map)
        self.slots_left_rot_key_map = get_item("slots_left_rot_key_map", gpufhe_content_map)
        self.total_left_rot_key_map = get_item("total_left_rot_key_map", gpufhe_content_map)
        self.slots_precompute_auto_map = get_item("slots_precompute_auto_map", gpufhe_content_map)
        self.primes = get_item("primes", gpufhe_content_map)
        self.prod_inv_moddown = get_item("prod_inv_moddown", gpufhe_content_map)
        self.prod_inv_shoup_moddown = get_item("prod_inv_shoup_moddown", gpufhe_content_map)
        self.prod_q_i_mod_q_j_moddown = get_item("prod_q_i_mod_q_j_moddown", gpufhe_content_map)
        self.prod_q_i_mod_q_j_modup = get_item("prod_q_i_mod_q_j_modup", gpufhe_content_map)
        self.qVec = get_item("qVec", gpufhe_content_map)
        self.q_inv_mod_q = get_item("q_inv_mod_q", gpufhe_content_map)
        self.q_inv_mod_q_shoup = get_item("q_inv_mod_q_shoup", gpufhe_content_map)
        self.q_mu = get_item("q_mu", gpufhe_content_map)
        self.qlql_inv_mod_ql_div_ql_mod_q = get_item("qlql_inv_mod_ql_div_ql_mod_q", gpufhe_content_map)
        self.qlql_inv_mod_ql_div_ql_mod_q_shoup = get_item("qlql_inv_mod_ql_div_ql_mod_q_shoup", gpufhe_content_map)
        self.rescaleTech = get_item("rescaleTech", gpufhe_content_map)
        self.rescale_out = get_item("rescale_out", gpufhe_content_map)
        self.scalingFactorsReal = get_item("scalingFactorsReal", gpufhe_content_map)
        self.scalingFactorsRealBig = get_item("scalingFactorsRealBig", gpufhe_content_map)
        self.secretKeyDist = get_item("secretKeyDist", gpufhe_content_map)
        self.sigma = get_item("sigma", gpufhe_content_map)
        self.mod_raise_out = get_item("mod_raise_out", gpufhe_content_map)
        self.swk_ax = get_item("swk_ax", gpufhe_content_map)
        self.swk_bx = get_item("swk_bx", gpufhe_content_map)
        self.QmuplusPmu_map = get_item("QmuplusPmu_map", gpufhe_content_map)
        self.QplusP_map = get_item("QplusP_map", gpufhe_content_map)
        self.QbarretKplusPbarretK_map = get_item("QbarretKplusPbarretK_map", gpufhe_content_map)
        self.QbarretRatioplusPbarretRatio_map = get_item("QbarretRatioplusPbarretRatio_map", gpufhe_content_map)
        self.QmaxdiffplusPmaxdiff_map = get_item("QmaxdiffplusPmaxdiff_map", gpufhe_content_map)
        self.BsContext_map = {}
        if self.logBsSlots_list[0]!=0: # if logBsSlots_list[0] is 0, then there are no BS ops in this application
            for logBsSlots in self.logBsSlots_list:
                _BsContext = BsContext(BsContext_content_map[str(logBsSlots)])
                self.BsContext_map[str(logBsSlots)] = _BsContext
        self.encode_params_ksiPows = get_item("encode_params_ksiPows", gpufhe_content_map)
        self.encode_params_rotGroup = get_item("encode_params_rotGroup", gpufhe_content_map)
        self.encode_temp = get_item("encode_temp", gpufhe_content_map)
        self.encode_inverse = get_item("encode_inverse", gpufhe_content_map)
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
        self.swk_bx = torch.tensor(self.swk_bx, dtype = torch.uint64)
        self.swk_ax = torch.tensor(self.swk_ax, dtype = torch.uint64)
        self.inner_workspace = torch.tensor(self.inner_workspace, dtype = torch.uint64)
        self.inner_out = torch.tensor(self.inner_out, dtype = torch.uint64)
        self.moddown_out_ax = torch.tensor(self.moddown_out_ax, dtype = torch.uint64)
        self.moddown_out_bx = torch.tensor(self.moddown_out_bx, dtype = torch.uint64)
        self.modup_out = torch.tensor(self.modup_out, dtype = torch.uint64)
        self.rescale_out = torch.tensor(self.rescale_out, dtype = torch.uint64)
        self.automorphism_transform_out = torch.tensor(self.automorphism_transform_out, dtype = torch.uint64)
        self.mod_raise_out = torch.tensor(self.mod_raise_out, dtype = torch.uint64)
        self.PModq = torch.tensor(self.PModq, dtype = torch.uint64)
        self.mult_key_map = [torch.tensor(v, dtype = torch.uint64) for v in self.mult_key_map]
        self.encode_params_ksiPows = torch.tensor(self.encode_params_ksiPows, dtype = torch.double)
        self.encode_params_rotGroup = torch.tensor(self.encode_params_rotGroup, dtype = torch.int64)
        self.encode_temp = torch.tensor(self.encode_temp, dtype = torch.int64)
        self.encode_inverse = torch.tensor(self.encode_inverse, dtype = torch.double)

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

        self.to_cuda()
        self.BsContext = None
        self.left_rot_key_map = {}
        self.precompute_auto_map = {}

        self.config = config
        self.inBS = False
        self.in_check_period = False

        self.device = "cpu" # fixme: ??

    def to_cuda(self):
        self.device = "cuda" #fixme: remove?
        self.q_mu = self.q_mu.cuda()
        self.moduliQ = self.moduliQ.cuda()
        self.primes = self.primes.cuda()
        self.power_of_roots = self.power_of_roots.cuda()
        self.power_of_roots_shoup = self.power_of_roots_shoup.cuda()
        self.inverse_power_of_roots_div_two = self.inverse_power_of_roots_div_two.cuda()
        self.inverse_scaled_power_of_roots_div_two = self.inverse_scaled_power_of_roots_div_two.cuda()
        self.barret_k = self.barret_k.cuda()
        self.barret_ratio = self.barret_ratio.cuda()
        self.hat_inverse_vec_modup = self.hat_inverse_vec_modup.cuda()
        self.hat_inverse_vec_shoup_modup = self.hat_inverse_vec_shoup_modup.cuda()
        self.prod_q_i_mod_q_j_modup = self.prod_q_i_mod_q_j_modup.cuda()
        self.hat_inverse_vec_moddown = self.hat_inverse_vec_moddown.cuda()
        self.hat_inverse_vec_shoup_moddown = self.hat_inverse_vec_shoup_moddown.cuda()
        self.prod_q_i_mod_q_j_moddown = self.prod_q_i_mod_q_j_moddown.cuda()
        self.prod_inv_moddown = self.prod_inv_moddown.cuda()
        self.prod_inv_shoup_moddown = self.prod_inv_shoup_moddown.cuda()
        self.qlql_inv_mod_ql_div_ql_mod_q = self.qlql_inv_mod_ql_div_ql_mod_q.cuda()
        self.qlql_inv_mod_ql_div_ql_mod_q_shoup = self.qlql_inv_mod_ql_div_ql_mod_q_shoup.cuda()
        self.q_inv_mod_q = self.q_inv_mod_q.cuda()
        self.q_inv_mod_q_shoup = self.q_inv_mod_q_shoup.cuda()
        self.swk_bx = self.swk_bx.cuda()
        self.swk_ax = self.swk_ax.cuda()
        self.inner_workspace = self.inner_workspace.cuda()
        self.inner_out = self.inner_out.cuda()
        self.moddown_out_ax = self.moddown_out_ax.cuda()
        self.moddown_out_bx = self.moddown_out_bx.cuda()
        self.modup_out = self.modup_out.cuda()
        self.rescale_out = self.rescale_out.cuda()
        self.automorphism_transform_out = self.automorphism_transform_out.cuda()
        self.mod_raise_out = self.mod_raise_out.cuda()
        self.PModq = self.PModq.cuda()
        self.mult_key_map = [v.cuda() for v in self.mult_key_map]
        self.encode_params_ksiPows = self.encode_params_ksiPows.cuda()
        self.encode_params_rotGroup = self.encode_params_rotGroup.cuda()
        self.encode_temp = self.encode_temp.cuda()
        self.encode_inverse = self.encode_inverse.cuda()
        self.max_int_diffs = self.max_int_diffs.cuda()
        for key, value in self.QplusP_map.items():
            self.QplusP_map[key] = value.cuda()
        for key, value in self.QmuplusPmu_map.items():
            self.QmuplusPmu_map[key] = value.cuda()
        for key, value in self.QbarretKplusPbarretK_map.items():
            self.QbarretKplusPbarretK_map[key] = value.cuda()
        for key, value in self.QbarretRatioplusPbarretRatio_map.items():
            self.QbarretRatioplusPbarretRatio_map[key] = value.cuda()
        for key, value in self.QmaxdiffplusPmaxdiff_map.items():
            self.QmaxdiffplusPmaxdiff_map[key] = value.cuda()


    # move to cpu
    def cpu(self):
        self.device = "cpu"
        self.q_mu = self.q_mu.cpu()
        self.moduliQ = self.moduliQ.cpu()
        self.primes = self.primes.cpu()
        self.power_of_roots = self.power_of_roots.cpu()
        self.power_of_roots_shoup = self.power_of_roots_shoup.cpu()
        self.inverse_power_of_roots_div_two = self.inverse_power_of_roots_div_two.cpu()
        self.inverse_scaled_power_of_roots_div_two =  self.inverse_scaled_power_of_roots_div_two.cpu()
        self.barret_k = self.barret_k.cpu()
        self.barret_ratio = self.barret_ratio.cpu()
        self.hat_inverse_vec_modup = self.hat_inverse_vec_modup.cpu()
        self.hat_inverse_vec_shoup_modup = self.hat_inverse_vec_shoup_modup.cpu()
        self.prod_q_i_mod_q_j_modup = self.prod_q_i_mod_q_j_modup.cpu()
        self.hat_inverse_vec_moddown =self.hat_inverse_vec_moddown.cpu()
        self.hat_inverse_vec_shoup_moddown = self.hat_inverse_vec_shoup_moddown.cpu()
        self.prod_q_i_mod_q_j_moddown = self.prod_q_i_mod_q_j_moddown.cpu()
        self.prod_inv_moddown =  self.prod_inv_moddown.cpu()
        self.prod_inv_shoup_moddown =  self.prod_inv_shoup_moddown.cpu()
        self.qlql_inv_mod_ql_div_ql_mod_q =  self.qlql_inv_mod_ql_div_ql_mod_q.cpu()
        self.qlql_inv_mod_ql_div_ql_mod_q_shoup = self.qlql_inv_mod_ql_div_ql_mod_q_shoup.cpu()
        self.q_inv_mod_q = self.q_inv_mod_q.cpu()
        self.q_inv_mod_q_shoup =self.q_inv_mod_q_shoup.cpu()
        self.swk_bx = self.swk_bx.cpu()
        self.swk_ax = self.swk_ax.cpu()
        self.inner_workspace = self.inner_workspace.cpu()
        self.inner_out =self.inner_out.cpu()
        self.moddown_out_ax = self.moddown_out_ax.cpu()
        self.moddown_out_bx = self.moddown_out_bx.cpu()
        self.modup_out = self.modup_out.cpu()
        self.rescale_out = self.rescale_out.cpu()
        self.automorphism_transform_out = self.automorphism_transform_out.cpu()
        self.mod_raise_out = self.mod_raise_out.cpu()
        self.PModq =self.PModq.cpu()
        self.mult_key_map = [v.cpu() for v in self.mult_key_map]

        # fixme: to be removed
        # self.encode_params_ksiPows_real = self.encode_params_ksiPows_real.cpu()
        # self.encode_params_ksiPows_imag = self.encode_params_ksiPows_imag.cpu()
        self.encode_params_rotGroup = self.encode_params_rotGroup.cpu()
        self.encode_temp = self.encode_temp.cpu()
        # self.encode_out = self.encode_out.cpu()


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
        if rot_index in self.left_rot_key_map:
            return self.left_rot_key_map[rot_index]
        else:
            return [
                torch.tensor(v, dtype=torch.uint64, device="cuda")
                for v in self.total_left_rot_key_map[rot_index]
            ]

    def get_precompute_auto(self, key):
        if key in self.precompute_auto_map:
            return self.precompute_auto_map[key]
        else:
            for k, v in self.slots_precompute_auto_map.items():
                if key in v:
                    return torch.tensor(v[key], dtype=torch.int32, device="cuda")
        assert False and "Key not found in precompute_auto_map"

    def load_rotation_keys(self, key_name):
        if not self.config.AUTO_LOAD_KEYS:
            print("AUTO_LOAD_KEYS is disabled. Do not call this function.")
            return
        assert str(key_name) in self.slots_left_rot_key_map
        assert str(key_name) in self.slots_precompute_auto_map
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

    def load_bootstrapping_context(self, logBsSlots):
        self.BsContext = self.BsContext_map[str(logBsSlots)]
        self.BsContext.to_cuda()
        self.load_rotation_keys(logBsSlots)
