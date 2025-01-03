import torch
from .bs_context import *

def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    return f"{message}\n"

class Context:

    def __init__(self, BsContext_content_map, gpufhe_content_map):
        for name, value in gpufhe_content_map.items():
            setattr(self, name, value)
        
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
        # if i == m - 1:
        #     return i
        if i == - 1:
            return  m - 1

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

