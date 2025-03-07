from .bs_context import *

class Context:
    def __init__(self, BsContext_content_map, gpufhe_content_map, autoLoadAndSetConfig):
        self.L = get_item("L", gpufhe_content_map)
        self.K = get_item("K", gpufhe_content_map)
        self.M = get_item("M", gpufhe_content_map)
        self.N = get_item("N", gpufhe_content_map)
        self.Nh = get_item("Nh", gpufhe_content_map)
        self.PInvModq = get_item("PInvModq", gpufhe_content_map)
        self.PModq = get_item("PModq", gpufhe_content_map)
        self.PartQlHatInvModq = get_item("PartQlHatInvModq", gpufhe_content_map)
        self.PartQlHatModp = get_item("PartQlHatModp", gpufhe_content_map)
        self.QlQlInvModqlDivqlModq = get_item(
            "QlQlInvModqlDivqlModq", gpufhe_content_map
        )
        self.approxSF = get_item("approxSF", gpufhe_content_map)
        self.automorphism_transform_out = get_item(
            "automorphism_transform_out", gpufhe_content_map
        )
        self.barret_k = get_item("barret_k", gpufhe_content_map)
        self.barret_ratio = get_item("barret_ratio", gpufhe_content_map)
        self.chain_length = get_item("chain_length", gpufhe_content_map)
        self.dmoduliQ = get_item("dmoduliQ", gpufhe_content_map)
        self.h = get_item("h", gpufhe_content_map)
        self.hat_inverse_vec_moddown = get_item(
            "hat_inverse_vec_moddown", gpufhe_content_map
        )
        self.hat_inverse_vec_modup = get_item(
            "hat_inverse_vec_modup", gpufhe_content_map
        )
        self.hat_inverse_vec_shoup_moddown = get_item(
            "hat_inverse_vec_shoup_moddown", gpufhe_content_map
        )
        self.hat_inverse_vec_shoup_modup = get_item(
            "hat_inverse_vec_shoup_modup", gpufhe_content_map
        )
        self.inner_out = get_item("inner_out", gpufhe_content_map)
        self.inner_workspace = get_item("inner_workspace", gpufhe_content_map)
        self.inverse_power_of_roots_div_two = get_item(
            "inverse_power_of_roots_div_two", gpufhe_content_map
        )
        self.inverse_scaled_power_of_roots_div_two = get_item(
            "inverse_scaled_power_of_roots_div_two", gpufhe_content_map
        )
        self.levelBudget = get_item("levelBudget", gpufhe_content_map)
        self.logN = get_item("logN", gpufhe_content_map)
        self.logNh = get_item("logNh", gpufhe_content_map)
        self.logBsSlots_list = get_item("logBsSlots_list", gpufhe_content_map)
        self.auxModSize = get_item("specialMod", gpufhe_content_map)
        self.max_num_moduli = get_item("max_num_moduli", gpufhe_content_map)
        self.moddown_out_ax = get_item("moddown_out_ax", gpufhe_content_map)
        self.moddown_out_bx = get_item("moddown_out_bx", gpufhe_content_map)
        self.moduliP_scalar = get_item("moduliP_scalar", gpufhe_content_map)
        self.moduliQ_scalar = get_item("moduliQ_scalar", gpufhe_content_map)
        self.moduliQ = get_item("moduliQ", gpufhe_content_map)
        self.modup_out = get_item("modup_out", gpufhe_content_map)
        self.mult_swk = get_item("mult_swk", gpufhe_content_map)
        self.num_moduli_after_moddown = get_item(
            "num_moduli_after_moddown", gpufhe_content_map
        )
        self.num_moduli_after_modup = get_item(
            "num_moduli_after_modup", gpufhe_content_map
        )
        self.num_special_moduli = get_item("num_special_moduli", gpufhe_content_map)
        self.p = get_item("p", gpufhe_content_map)
        self.pHatInvModp = get_item("pHatInvModp", gpufhe_content_map)
        self.pHatModp = get_item("pHatModp", gpufhe_content_map)
        self.pHatModq = get_item("pHatModq", gpufhe_content_map)
        self.p_mu = get_item("p_mu", gpufhe_content_map)
        self.power_of_roots = get_item("power_of_roots", gpufhe_content_map)
        self.power_of_roots_shoup = get_item("power_of_roots_shoup", gpufhe_content_map)
        self.power_of_roots_shoup_vec = get_item(
            "power_of_roots_shoup_vec", gpufhe_content_map
        )
        self.power_of_roots_vec = get_item("power_of_roots_vec", gpufhe_content_map)
        self.mult_key_map = get_item("mult_key_map", gpufhe_content_map)
        self.slots_left_rot_key_map = get_item(
            "slots_left_rot_key_map", gpufhe_content_map
        )
        self.slots_precompute_auto_map = get_item(
            "slots_precompute_auto_map", gpufhe_content_map
        )
        self.primes = get_item("primes", gpufhe_content_map)
        self.prod_inv_moddown = get_item("prod_inv_moddown", gpufhe_content_map)
        self.prod_inv_shoup_moddown = get_item(
            "prod_inv_shoup_moddown", gpufhe_content_map
        )
        self.prod_q_i_mod_q_j_moddown = get_item(
            "prod_q_i_mod_q_j_moddown", gpufhe_content_map
        )
        self.prod_q_i_mod_q_j_modup = get_item(
            "prod_q_i_mod_q_j_modup", gpufhe_content_map
        )
        self.qVec = get_item("qVec", gpufhe_content_map)
        self.q_inv_mod_q = get_item("q_inv_mod_q", gpufhe_content_map)
        self.q_inv_mod_q_shoup = get_item("q_inv_mod_q_shoup", gpufhe_content_map)
        self.q_mu = get_item("q_mu", gpufhe_content_map)
        self.qlql_inv_mod_ql_div_ql_mod_q = get_item(
            "qlql_inv_mod_ql_div_ql_mod_q", gpufhe_content_map
        )
        self.qlql_inv_mod_ql_div_ql_mod_q_shoup = get_item(
            "qlql_inv_mod_ql_div_ql_mod_q_shoup", gpufhe_content_map
        )
        self.rescale_out = get_item("rescale_out", gpufhe_content_map)
        self.sigma = get_item("sigma", gpufhe_content_map)
        self.mod_raise_out = get_item("mod_raise_out", gpufhe_content_map)
        self.swk_ax = get_item("swk_ax", gpufhe_content_map)
        self.swk_bx = get_item("swk_bx", gpufhe_content_map)
        self.BsContext_map = {}
        if self.logBsSlots_list[0] != 0:
            for logBsSlots in self.logBsSlots_list:
                _BsContext = BsContext(BsContext_content_map[str(logBsSlots)])
                self.BsContext_map[str(logBsSlots)] = _BsContext
        self.q_mu = torch.tensor(self.q_mu, dtype=torch.uint64)
        self.moduliQ = torch.tensor(self.moduliQ, dtype=torch.uint64)
        self.primes = torch.tensor(self.primes, dtype=torch.uint64)
        self.power_of_roots = torch.tensor(self.power_of_roots, dtype=torch.uint64)
        self.power_of_roots_shoup = torch.tensor(
            self.power_of_roots_shoup, dtype=torch.uint64
        )
        self.inverse_power_of_roots_div_two = torch.tensor(
            self.inverse_power_of_roots_div_two, dtype=torch.uint64
        )
        self.inverse_scaled_power_of_roots_div_two = torch.tensor(
            self.inverse_scaled_power_of_roots_div_two, dtype=torch.uint64
        )
        self.barret_k = torch.tensor(self.barret_k, dtype=torch.uint64)
        self.barret_ratio = torch.tensor(self.barret_ratio, dtype=torch.uint64)
        self.hat_inverse_vec_modup = torch.tensor(
            self.hat_inverse_vec_modup, dtype=torch.uint64
        )
        self.hat_inverse_vec_shoup_modup = torch.tensor(
            self.hat_inverse_vec_shoup_modup, dtype=torch.uint64
        )
        self.prod_q_i_mod_q_j_modup = torch.tensor(
            self.prod_q_i_mod_q_j_modup, dtype=torch.uint64
        )
        self.hat_inverse_vec_moddown = torch.tensor(
            self.hat_inverse_vec_moddown, dtype=torch.uint64
        )
        self.hat_inverse_vec_shoup_moddown = torch.tensor(
            self.hat_inverse_vec_shoup_moddown, dtype=torch.uint64
        )
        self.prod_q_i_mod_q_j_moddown = torch.tensor(
            self.prod_q_i_mod_q_j_moddown, dtype=torch.uint64
        )
        self.prod_inv_moddown = torch.tensor(self.prod_inv_moddown, dtype=torch.uint64)
        self.prod_inv_shoup_moddown = torch.tensor(
            self.prod_inv_shoup_moddown, dtype=torch.uint64
        )
        self.qlql_inv_mod_ql_div_ql_mod_q = torch.tensor(
            self.qlql_inv_mod_ql_div_ql_mod_q, dtype=torch.uint64
        )
        self.qlql_inv_mod_ql_div_ql_mod_q_shoup = torch.tensor(
            self.qlql_inv_mod_ql_div_ql_mod_q_shoup, dtype=torch.uint64
        )
        self.q_inv_mod_q = torch.tensor(self.q_inv_mod_q, dtype=torch.uint64)
        self.q_inv_mod_q_shoup = torch.tensor(
            self.q_inv_mod_q_shoup, dtype=torch.uint64
        )
        self.swk_bx = torch.tensor(self.swk_bx, dtype=torch.uint64)
        self.swk_ax = torch.tensor(self.swk_ax, dtype=torch.uint64)
        self.inner_workspace = torch.tensor(self.inner_workspace, dtype=torch.uint64)
        self.inner_out = torch.tensor(self.inner_out, dtype=torch.uint64)
        self.moddown_out_ax = torch.tensor(self.moddown_out_ax, dtype=torch.uint64)
        self.moddown_out_bx = torch.tensor(self.moddown_out_bx, dtype=torch.uint64)
        self.modup_out = torch.tensor(self.modup_out, dtype=torch.uint64)
        self.rescale_out = torch.tensor(self.rescale_out, dtype=torch.uint64)
        self.automorphism_transform_out = torch.tensor(
            self.automorphism_transform_out, dtype=torch.uint64
        )
        self.mod_raise_out = torch.tensor(self.mod_raise_out, dtype=torch.uint64)
        self.PModq = torch.tensor(self.PModq, dtype=torch.uint64)
        self.mult_key_map = [
            torch.tensor(v, dtype=torch.uint64) for v in self.mult_key_map
        ]

        self.to_cuda()
        self.BsContext = None
        self.left_rot_key_map = {}
        self.precompute_auto_map = {}

        self.autoLoadAndSetConfig = autoLoadAndSetConfig

    def to_cuda(self):
        self.q_mu = self.q_mu.cuda()
        self.moduliQ = self.moduliQ.cuda()
        self.primes = self.primes.cuda()
        self.power_of_roots = self.power_of_roots.cuda()
        self.power_of_roots_shoup = self.power_of_roots_shoup.cuda()
        self.inverse_power_of_roots_div_two = self.inverse_power_of_roots_div_two.cuda()
        self.inverse_scaled_power_of_roots_div_two = (
            self.inverse_scaled_power_of_roots_div_two.cuda()
        )
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
        self.qlql_inv_mod_ql_div_ql_mod_q_shoup = (
            self.qlql_inv_mod_ql_div_ql_mod_q_shoup.cuda()
        )
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

    def norm_rot_index(self, i):
        if i < 0:
            i = self.N // 2 + i
        return i
