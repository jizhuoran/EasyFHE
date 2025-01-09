import torch
from .bs_context import *

def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    return f"{message}\n"

class Context:

    def __init__(self, BsContext_content_map, gpufhe_content_map):
        self.K = gpufhe_content_map["K"]
        self.L = gpufhe_content_map["L"]
        self.M = gpufhe_content_map["M"]
        self.N = gpufhe_content_map["N"]
        self.Nh = gpufhe_content_map["Nh"]
        self.PInvModq = gpufhe_content_map["PInvModq"]
        self.PModq = gpufhe_content_map["PModq"]
        self.PModq_cuda = gpufhe_content_map["PModq_cuda"]
        self.PartQlHatInvModq = gpufhe_content_map["PartQlHatInvModq"]
        self.PartQlHatModp = gpufhe_content_map["PartQlHatModp"]
        self.PartQlHatModp_pad = gpufhe_content_map["PartQlHatModp_pad"]
        self.QHatInvModq = gpufhe_content_map["QHatInvModq"]
        self.QHatModp = gpufhe_content_map["QHatModp"]
        self.QlQlInvModqlDivqlModq = gpufhe_content_map["QlQlInvModqlDivqlModq"]
        self.automorphism_transform_out = gpufhe_content_map["automorphism_transform_out"]
        self.barret_k = gpufhe_content_map["barret_k"]
        self.barret_ratio = gpufhe_content_map["barret_ratio"]
        self.beta = gpufhe_content_map["beta"]
        self.chain_length = gpufhe_content_map["chain_length"]
        self.correctionFactor = gpufhe_content_map["correctionFactor"]
        self.dmoduliQ = gpufhe_content_map["dmoduliQ"]
        self.dnum = gpufhe_content_map["dnum"]
        self.h = gpufhe_content_map["h"]
        self.hat_inverse_vec_moddown = gpufhe_content_map["hat_inverse_vec_moddown"]
        self.hat_inverse_vec_modup = gpufhe_content_map["hat_inverse_vec_modup"]
        self.hat_inverse_vec_shoup_moddown = gpufhe_content_map["hat_inverse_vec_shoup_moddown"]
        self.hat_inverse_vec_shoup_modup = gpufhe_content_map["hat_inverse_vec_shoup_modup"]
        self.inner_out = gpufhe_content_map["inner_out"]
        self.inner_workspace = gpufhe_content_map["inner_workspace"]
        self.inv_power_of_roots_shoup_vec = gpufhe_content_map["inv_power_of_roots_shoup_vec"]
        self.inv_power_of_roots_vec = gpufhe_content_map["inv_power_of_roots_vec"]
        self.inverse_power_of_roots_div_two = gpufhe_content_map["inverse_power_of_roots_div_two"]
        self.inverse_scaled_power_of_roots_div_two = gpufhe_content_map["inverse_scaled_power_of_roots_div_two"]
        self.key_map = gpufhe_content_map["key_map"]
        self.left_rot_key_map = gpufhe_content_map["left_rot_key_map"]
        self.levelBudget = gpufhe_content_map["levelBudget"]
        self.logN = gpufhe_content_map["logN"]
        self.logNh = gpufhe_content_map["logNh"]
        self.logSlots = gpufhe_content_map["logSlots"]
        self.logp = gpufhe_content_map["logp"]
        self.logqi = gpufhe_content_map["logqi"]
        self.m_U0PreFFT_dim = gpufhe_content_map["m_U0PreFFT_dim"]
        self.m_U0PreFFT_limbs = gpufhe_content_map["m_U0PreFFT_limbs"]
        self.m_U0PreFFT_mx = gpufhe_content_map["m_U0PreFFT_mx"]
        self.m_U0PreFFT_scaling_factor = gpufhe_content_map["m_U0PreFFT_scaling_factor"]
        self.m_U0hatTPreFFT_dim = gpufhe_content_map["m_U0hatTPreFFT_dim"]
        self.m_U0hatTPreFFT_limbs = gpufhe_content_map["m_U0hatTPreFFT_limbs"]
        self.m_U0hatTPreFFT_mx = gpufhe_content_map["m_U0hatTPreFFT_mx"]
        self.m_U0hatTPreFFT_scaling_factor = gpufhe_content_map["m_U0hatTPreFFT_scaling_factor"]
        self.max_num_moduli = gpufhe_content_map["max_num_moduli"]
        self.moddown_out_ax = gpufhe_content_map["moddown_out_ax"]
        self.moddown_out_bx = gpufhe_content_map["moddown_out_bx"]
        self.moduliP = gpufhe_content_map["moduliP"]
        self.moduliQ = gpufhe_content_map["moduliQ"]
        self.moduliQ_cuda = gpufhe_content_map["moduliQ_cuda"]
        self.modup_out = gpufhe_content_map["modup_out"]
        self.mult_swk = gpufhe_content_map["mult_swk"]
        self.num_moduli_after_moddown = gpufhe_content_map["num_moduli_after_moddown"]
        self.num_moduli_after_modup = gpufhe_content_map["num_moduli_after_modup"]
        self.num_special_moduli = gpufhe_content_map["num_special_moduli"]
        self.p = gpufhe_content_map["p"]
        self.pHatInvModp = gpufhe_content_map["pHatInvModp"]
        self.pHatModp = gpufhe_content_map["pHatModp"]
        self.pHatModq = gpufhe_content_map["pHatModq"]
        self.pInvVec = gpufhe_content_map["pInvVec"]
        self.pRootPows = gpufhe_content_map["pRootPows"]
        self.pRootPowsInv = gpufhe_content_map["pRootPowsInv"]
        self.pRootScalePows = gpufhe_content_map["pRootScalePows"]
        self.pRootScalePowsInv = gpufhe_content_map["pRootScalePowsInv"]
        self.pRootScalePowsOverp = gpufhe_content_map["pRootScalePowsOverp"]
        self.pRoots = gpufhe_content_map["pRoots"]
        self.pRootsInv = gpufhe_content_map["pRootsInv"]
        self.p_mu = gpufhe_content_map["p_mu"]
        self.power_of_roots = gpufhe_content_map["power_of_roots"]
        self.power_of_roots_shoup = gpufhe_content_map["power_of_roots_shoup"]
        self.power_of_roots_shoup_vec = gpufhe_content_map["power_of_roots_shoup_vec"]
        self.power_of_roots_vec = gpufhe_content_map["power_of_roots_vec"]
        self.precompute_auto_map = gpufhe_content_map["precompute_auto_map"]
        self.primes = gpufhe_content_map["primes"]
        self.prod_inv_moddown = gpufhe_content_map["prod_inv_moddown"]
        self.prod_inv_shoup_moddown = gpufhe_content_map["prod_inv_shoup_moddown"]
        self.prod_q_i_mod_q_j_moddown = gpufhe_content_map["prod_q_i_mod_q_j_moddown"]
        self.prod_q_i_mod_q_j_modup = gpufhe_content_map["prod_q_i_mod_q_j_modup"]
        self.qInvModq = gpufhe_content_map["qInvModq"]
        self.qRootPows = gpufhe_content_map["qRootPows"]
        self.qRootPowsInv = gpufhe_content_map["qRootPowsInv"]
        self.qRootScalePows = gpufhe_content_map["qRootScalePows"]
        self.qRootScalePowsInv = gpufhe_content_map["qRootScalePowsInv"]
        self.qRootScalePowsOverq = gpufhe_content_map["qRootScalePowsOverq"]
        self.qRoots = gpufhe_content_map["qRoots"]
        self.qRootsInv = gpufhe_content_map["qRootsInv"]
        self.qVec = gpufhe_content_map["qVec"]
        self.q_inv_mod_q = gpufhe_content_map["q_inv_mod_q"]
        self.q_inv_mod_q_shoup = gpufhe_content_map["q_inv_mod_q_shoup"]
        self.q_mu = gpufhe_content_map["q_mu"]
        self.q_mu_cuda = gpufhe_content_map["q_mu_cuda"]
        self.qlql_inv_mod_ql_div_ql_mod_q = gpufhe_content_map["qlql_inv_mod_ql_div_ql_mod_q"]
        self.qlql_inv_mod_ql_div_ql_mod_q_shoup = gpufhe_content_map["qlql_inv_mod_ql_div_ql_mod_q_shoup"]
        self.rescaleTech = gpufhe_content_map["rescaleTech"]
        self.rescale_out = gpufhe_content_map["rescale_out"]
        self.scalingFactorsReal = gpufhe_content_map["scalingFactorsReal"]
        self.scalingFactorsRealBig = gpufhe_content_map["scalingFactorsRealBig"]
        self.secretKeyDist = gpufhe_content_map["secretKeyDist"]
        self.sigma = gpufhe_content_map["sigma"]
        self.switch_modulus_out = gpufhe_content_map["switch_modulus_out"]
        self.swk_ax_cuda = gpufhe_content_map["swk_ax_cuda"]
        self.swk_bx_cuda = gpufhe_content_map["swk_bx_cuda"]
        
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

