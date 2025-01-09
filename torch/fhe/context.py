import torch
from .bs_context import *

def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    return f"{message}\n"

class Context:
    def __init__(self, BsContext_content_map, gpufhe_content_map):
        self.K = get_item("K", gpufhe_content_map)
        self.L = get_item("L", gpufhe_content_map)
        self.M = get_item("M", gpufhe_content_map)
        self.N = get_item("N", gpufhe_content_map)
        self.Nh = get_item("Nh", gpufhe_content_map)
        self.PInvModq = get_item("PInvModq", gpufhe_content_map)
        self.PModq = get_item("PModq", gpufhe_content_map)
        self.PModq_cuda = get_item("PModq_cuda", gpufhe_content_map)
        self.PartQlHatInvModq = get_item("PartQlHatInvModq", gpufhe_content_map)
        self.PartQlHatModp = get_item("PartQlHatModp", gpufhe_content_map)
        self.PartQlHatModp_pad = get_item("PartQlHatModp_pad", gpufhe_content_map)
        self.QHatInvModq = get_item("QHatInvModq", gpufhe_content_map)
        self.QHatModp = get_item("QHatModp", gpufhe_content_map)
        self.QlQlInvModqlDivqlModq = get_item("QlQlInvModqlDivqlModq", gpufhe_content_map)
        self.approxSF = get_item("approxSF", gpufhe_content_map)
        self.automorphism_transform_out = get_item("automorphism_transform_out", gpufhe_content_map)
        self.barret_k = get_item("barret_k", gpufhe_content_map)
        self.barret_ratio = get_item("barret_ratio", gpufhe_content_map)
        self.beta = get_item("beta", gpufhe_content_map)
        self.chain_length = get_item("chain_length", gpufhe_content_map)
        self.correctionFactor = get_item("correctionFactor", gpufhe_content_map)
        self.dmoduliQ = get_item("dmoduliQ", gpufhe_content_map)
        self.dnum = get_item("dnum", gpufhe_content_map)
        self.h = get_item("h", gpufhe_content_map)
        self.hat_inverse_vec_moddown = get_item("hat_inverse_vec_moddown", gpufhe_content_map)
        self.hat_inverse_vec_modup = get_item("hat_inverse_vec_modup", gpufhe_content_map)
        self.hat_inverse_vec_shoup_moddown = get_item("hat_inverse_vec_shoup_moddown", gpufhe_content_map)
        self.hat_inverse_vec_shoup_modup = get_item("hat_inverse_vec_shoup_modup", gpufhe_content_map)
        self.inner_out = get_item("inner_out", gpufhe_content_map)
        self.inner_workspace = get_item("inner_workspace", gpufhe_content_map)
        self.inv_power_of_roots_shoup_vec = get_item("inv_power_of_roots_shoup_vec", gpufhe_content_map)
        self.inv_power_of_roots_vec = get_item("inv_power_of_roots_vec", gpufhe_content_map)
        self.inverse_power_of_roots_div_two = get_item("inverse_power_of_roots_div_two", gpufhe_content_map)
        self.inverse_scaled_power_of_roots_div_two = get_item("inverse_scaled_power_of_roots_div_two", gpufhe_content_map)
        self.key_map = get_item("key_map", gpufhe_content_map)
        self.left_rot_key_map = get_item("left_rot_key_map", gpufhe_content_map)
        self.levelBudget = get_item("levelBudget", gpufhe_content_map)
        self.logN = get_item("logN", gpufhe_content_map)
        self.logNh = get_item("logNh", gpufhe_content_map)
        self.logSlots = get_item("logSlots", gpufhe_content_map)
        self.auxModSize = get_item("logp", gpufhe_content_map)
        self.dcrtBits = get_item("logqi", gpufhe_content_map)
        #todo: need to add firstMod? correspond to firstMod in openfhe, correspond to q0 in client.py
        self.m_U0PreFFT_dim = get_item("m_U0PreFFT_dim", gpufhe_content_map)
        self.m_U0PreFFT_limbs = get_item("m_U0PreFFT_limbs", gpufhe_content_map)
        self.m_U0PreFFT_mx = get_item("m_U0PreFFT_mx", gpufhe_content_map)
        self.m_U0PreFFT_scaling_factor = get_item("m_U0PreFFT_scaling_factor", gpufhe_content_map)
        self.m_U0hatTPreFFT_dim = get_item("m_U0hatTPreFFT_dim", gpufhe_content_map)
        self.m_U0hatTPreFFT_limbs = get_item("m_U0hatTPreFFT_limbs", gpufhe_content_map)
        self.m_U0hatTPreFFT_mx = get_item("m_U0hatTPreFFT_mx", gpufhe_content_map)
        self.m_U0hatTPreFFT_scaling_factor = get_item("m_U0hatTPreFFT_scaling_factor", gpufhe_content_map)
        self.max_num_moduli = get_item("max_num_moduli", gpufhe_content_map)
        self.moddown_out_ax = get_item("moddown_out_ax", gpufhe_content_map)
        self.moddown_out_bx = get_item("moddown_out_bx", gpufhe_content_map)
        self.moduliP = get_item("moduliP", gpufhe_content_map)
        self.moduliQ = get_item("moduliQ", gpufhe_content_map)
        self.moduliQ_cuda = get_item("moduliQ_cuda", gpufhe_content_map)
        self.modup_out = get_item("modup_out", gpufhe_content_map)
        self.mult_swk = get_item("mult_swk", gpufhe_content_map)
        self.num_moduli_after_moddown = get_item("num_moduli_after_moddown", gpufhe_content_map)
        self.num_moduli_after_modup = get_item("num_moduli_after_modup", gpufhe_content_map)
        self.num_special_moduli = get_item("num_special_moduli", gpufhe_content_map)
        self.p = get_item("p", gpufhe_content_map)
        self.pHatInvModp = get_item("pHatInvModp", gpufhe_content_map)
        self.pHatModp = get_item("pHatModp", gpufhe_content_map)
        self.pHatModq = get_item("pHatModq", gpufhe_content_map)
        self.pInvVec = get_item("pInvVec", gpufhe_content_map)
        self.pRootPows = get_item("pRootPows", gpufhe_content_map)
        self.pRootPowsInv = get_item("pRootPowsInv", gpufhe_content_map)
        self.pRootScalePows = get_item("pRootScalePows", gpufhe_content_map)
        self.pRootScalePowsInv = get_item("pRootScalePowsInv", gpufhe_content_map)
        self.pRootScalePowsOverp = get_item("pRootScalePowsOverp", gpufhe_content_map)
        self.pRoots = get_item("pRoots", gpufhe_content_map)
        self.pRootsInv = get_item("pRootsInv", gpufhe_content_map)
        self.p_mu = get_item("p_mu", gpufhe_content_map)
        self.power_of_roots = get_item("power_of_roots", gpufhe_content_map)
        self.power_of_roots_shoup = get_item("power_of_roots_shoup", gpufhe_content_map)
        self.power_of_roots_shoup_vec = get_item("power_of_roots_shoup_vec", gpufhe_content_map)
        self.power_of_roots_vec = get_item("power_of_roots_vec", gpufhe_content_map)
        self.precompute_auto_map = get_item("precompute_auto_map", gpufhe_content_map)
        self.primes = get_item("primes", gpufhe_content_map)
        self.prod_inv_moddown = get_item("prod_inv_moddown", gpufhe_content_map)
        self.prod_inv_shoup_moddown = get_item("prod_inv_shoup_moddown", gpufhe_content_map)
        self.prod_q_i_mod_q_j_moddown = get_item("prod_q_i_mod_q_j_moddown", gpufhe_content_map)
        self.prod_q_i_mod_q_j_modup = get_item("prod_q_i_mod_q_j_modup", gpufhe_content_map)
        self.qInvModq = get_item("qInvModq", gpufhe_content_map)
        self.qRootPows = get_item("qRootPows", gpufhe_content_map)
        self.qRootPowsInv = get_item("qRootPowsInv", gpufhe_content_map)
        self.qRootScalePows = get_item("qRootScalePows", gpufhe_content_map)
        self.qRootScalePowsInv = get_item("qRootScalePowsInv", gpufhe_content_map)
        self.qRootScalePowsOverq = get_item("qRootScalePowsOverq", gpufhe_content_map)
        self.qRoots = get_item("qRoots", gpufhe_content_map)
        self.qRootsInv = get_item("qRootsInv", gpufhe_content_map)
        self.qVec = get_item("qVec", gpufhe_content_map)
        self.q_inv_mod_q = get_item("q_inv_mod_q", gpufhe_content_map)
        self.q_inv_mod_q_shoup = get_item("q_inv_mod_q_shoup", gpufhe_content_map)
        self.q_mu = get_item("q_mu", gpufhe_content_map)
        self.q_mu_cuda = get_item("q_mu_cuda", gpufhe_content_map)
        self.qlql_inv_mod_ql_div_ql_mod_q = get_item("qlql_inv_mod_ql_div_ql_mod_q", gpufhe_content_map)
        self.qlql_inv_mod_ql_div_ql_mod_q_shoup = get_item("qlql_inv_mod_ql_div_ql_mod_q_shoup", gpufhe_content_map)
        self.rescaleTech = get_item("rescaleTech", gpufhe_content_map)
        self.rescale_out = get_item("rescale_out", gpufhe_content_map)
        self.scalingFactorsReal = get_item("scalingFactorsReal", gpufhe_content_map)
        self.scalingFactorsRealBig = get_item("scalingFactorsRealBig", gpufhe_content_map)
        self.secretKeyDist = get_item("secretKeyDist", gpufhe_content_map)
        self.sigma = get_item("sigma", gpufhe_content_map)
        self.switch_modulus_out = get_item("switch_modulus_out", gpufhe_content_map)
        self.swk_ax_cuda = get_item("swk_ax_cuda", gpufhe_content_map)
        self.swk_bx_cuda = get_item("swk_bx_cuda", gpufhe_content_map)
        
        _BsContext = BsContext(BsContext_content_map)
        self.BsContext = _BsContext
        self.to_cuda()


    def to_cuda(self):
        self.q_mu_cuda = torch.tensor(self.q_mu_cuda, dtype = torch.uint64, device = "cuda")
        self.moduliQ_cuda = torch.tensor(self.moduliQ_cuda, dtype = torch.uint64, device = "cuda")
        self.primes = torch.tensor(self.primes, dtype = torch.uint64, device = "cuda")
        self.power_of_roots = torch.tensor(self.power_of_roots, dtype = torch.uint64, device = "cuda")
        self.power_of_roots_shoup = torch.tensor(self.power_of_roots_shoup, dtype = torch.uint64, device = "cuda")
        self.inverse_power_of_roots_div_two = torch.tensor(self.inverse_power_of_roots_div_two, dtype = torch.uint64, device = "cuda")
        self.inverse_scaled_power_of_roots_div_two = torch.tensor(self.inverse_scaled_power_of_roots_div_two, dtype = torch.uint64, device = "cuda")
        self.barret_k = torch.tensor(self.barret_k, dtype = torch.uint64, device = "cuda")
        self.barret_ratio = torch.tensor(self.barret_ratio, dtype = torch.uint64, device = "cuda")
        self.hat_inverse_vec_modup = torch.tensor(self.hat_inverse_vec_modup, dtype = torch.uint64, device = "cuda")
        self.hat_inverse_vec_shoup_modup = torch.tensor(self.hat_inverse_vec_shoup_modup, dtype = torch.uint64, device = "cuda")
        self.prod_q_i_mod_q_j_modup = torch.tensor(self.prod_q_i_mod_q_j_modup, dtype = torch.uint64, device = "cuda")
        self.hat_inverse_vec_moddown = torch.tensor(self.hat_inverse_vec_moddown, dtype = torch.uint64, device = "cuda")
        self.hat_inverse_vec_shoup_moddown = torch.tensor(self.hat_inverse_vec_shoup_moddown, dtype = torch.uint64, device = "cuda")
        self.prod_q_i_mod_q_j_moddown = torch.tensor(self.prod_q_i_mod_q_j_moddown, dtype = torch.uint64, device = "cuda")
        self.prod_inv_moddown = torch.tensor(self.prod_inv_moddown, dtype = torch.uint64, device = "cuda")
        self.prod_inv_shoup_moddown = torch.tensor(self.prod_inv_shoup_moddown, dtype = torch.uint64, device = "cuda")
        self.qlql_inv_mod_ql_div_ql_mod_q = torch.tensor(self.qlql_inv_mod_ql_div_ql_mod_q, dtype = torch.uint64, device = "cuda")
        self.qlql_inv_mod_ql_div_ql_mod_q_shoup = torch.tensor(self.qlql_inv_mod_ql_div_ql_mod_q_shoup, dtype = torch.uint64, device = "cuda")
        self.q_inv_mod_q = torch.tensor(self.q_inv_mod_q, dtype = torch.uint64, device = "cuda")
        self.q_inv_mod_q_shoup = torch.tensor(self.q_inv_mod_q_shoup, dtype = torch.uint64, device = "cuda")
        self.swk_bx_cuda = torch.tensor(self.swk_bx_cuda, dtype = torch.uint64, device = "cuda")
        self.swk_ax_cuda = torch.tensor(self.swk_ax_cuda, dtype = torch.uint64, device = "cuda")
        self.inner_workspace = torch.tensor(self.inner_workspace, dtype = torch.uint64, device = "cuda")
        self.inner_out = torch.tensor(self.inner_out, dtype = torch.uint64, device = "cuda")
        self.moddown_out_ax = torch.tensor(self.moddown_out_ax, dtype = torch.uint64, device = "cuda")
        self.moddown_out_bx = torch.tensor(self.moddown_out_bx, dtype = torch.uint64, device = "cuda")
        self.modup_out = torch.tensor(self.modup_out, dtype = torch.uint64, device = "cuda")
        self.rescale_out = torch.tensor(self.rescale_out, dtype = torch.uint64, device = "cuda")
        self.automorphism_transform_out = torch.tensor(self.automorphism_transform_out, dtype = torch.uint64, device = "cuda")
        self.switch_modulus_out = torch.tensor(self.switch_modulus_out, dtype = torch.uint64, device = "cuda")
        self.PModq_cuda = torch.tensor(self.PModq_cuda, dtype = torch.uint64, device = "cuda")

        self.key_map = [torch.tensor(v, dtype = torch.uint64, device = "cuda") for v in self.key_map]

        for key, value in self.left_rot_key_map.items():
            self.left_rot_key_map[key] = [torch.tensor(v, dtype = torch.uint64, device = "cuda") for v in value]
        for key, value in self.precompute_auto_map.items():
            self.precompute_auto_map[key] = torch.tensor(value, dtype = torch.int32, device = "cuda")

        for key, value in self.BsContext.QplusP_map.items():
            self.BsContext.QplusP_map[key] = torch.tensor(value, dtype = torch.uint64, device = "cuda")
        for key, value in self.BsContext.QmuplusPmu_map.items():
            self.BsContext.QmuplusPmu_map[key] = torch.tensor(value, dtype = torch.uint64, device = "cuda")

        
        for i in range(len(self.BsContext.m_U0hatTPreFFT)):
            for j in range(len(self.BsContext.m_U0hatTPreFFT[i])):
                self.BsContext.m_U0hatTPreFFT[i][j].mx = torch.tensor(self.BsContext.m_U0hatTPreFFT[i][j].mx, dtype = torch.uint64, device = "cuda")

        for i in range(len(self.BsContext.m_U0PreFFT)):
            for j in range(len(self.BsContext.m_U0PreFFT[i])):
                self.BsContext.m_U0PreFFT[i][j].mx = torch.tensor(self.BsContext.m_U0PreFFT[i][j].mx, dtype = torch.uint64, device = "cuda")            

    def find_auto_index(self, i):
        def inv_mod(a, m): #note: check all the output value before merge with func: invMod!! These two values may differ by m!!
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

        m = (self.N << 1)

        if i == 0:
            return 1

        # Conjugation automorphism
        if i == m - 1:
            return i

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

