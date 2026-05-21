import easyfhe as torch
import numpy as np


class Context:
    def __init__(
        self,
        material,
        device,
        options,
        *,
        native_context_gen=False,
        generation_metadata=None,
        roots_q=None,
        roots_p=None,
    ):
        # Context metadata.
        self.device = device
        self.options = options
        self.native_context_gen = bool(native_context_gen)
        self.context_generation_config = None if generation_metadata is None else dict(generation_metadata)
        if roots_q is not None:
            self.rootsQ = np.asarray(roots_q, dtype=np.uint64)
        if roots_p is not None:
            self.rootsP = np.asarray(roots_p, dtype=np.uint64)

        # Scalar parameters.
        self.L = material.L
        self.dnum = material.dnum
        self.alpha = material.alpha
        self.K = material.K
        self.M = material.M
        self.N = material.N
        self.Nh = material.Nh
        self.approxSF = material.approxSF
        self.h = material.h
        self.levelBudget = material.levelBudget
        self.logN = material.logN
        self.logNh = material.logNh
        self.logBsSlots_list = material.logBsSlots_list
        self.auxModSize = material.auxModSize
        self.scale_mode = material.scale_mode
        self.rescale_policy = material.rescale_policy
        self.dcrtBits = material.dcrtBits
        self.max_num_moduli = material.max_num_moduli
        self.secretKeyDist = material.secretKeyDist
        self.sigma = material.sigma
        self.inBS = material.inBS
        self.moduliP_scalar = material.moduliP_scalar
        self.moduliQ_scalar = material.moduliQ_scalar
        self.scalingFactorsReal = material.scalingFactorsReal
        self.scalingFactorsRealBig = material.scalingFactorsRealBig

        # Device tensors.
        self.primes = material.primes.to(device)
        self.primes_list = [int(p) for p in material.primes.tolist()]
        switch_map = []
        for old_mod in self.primes_list:
            for new_mod in self.primes_list:
                if old_mod > new_mod:
                    diff = new_mod - (old_mod % new_mod)
                else:
                    diff = new_mod - old_mod
                switch_map.append(diff)
        self.switch_modulus_map = torch.tensor(switch_map, dtype=torch.uint64, device=device)
        self.barret_k = material.barret_k.to(device)
        self.barret_ratio = material.barret_ratio.to(device)
        self.q_mu = material.q_mu.to(device)
        self.moduliQ = material.moduliQ.to(device)
        self.PModq = material.PModq.to(device)
        self.max_int_diffs = material.max_int_diffs.to(device)

        self.automorphism_transform_out = material.automorphism_transform_out.to(device)
        self.inner_out = material.inner_out.to(device)
        self.moddown_out_ax = material.moddown_out_ax.to(device)
        self.moddown_out_bx = material.moddown_out_bx.to(device)
        self.modup_out = material.modup_out.to(device)
        self.rescale_out = material.rescale_out.to(device)
        self.mod_raise_out = material.mod_raise_out.to(device)

        self.hat_inverse_vec_moddown = material.hat_inverse_vec_moddown.to(device)
        self.hat_inverse_vec_shoup_moddown = material.hat_inverse_vec_shoup_moddown.to(device)
        self.prod_inv_moddown = material.prod_inv_moddown.to(device)
        self.prod_inv_shoup_moddown = material.prod_inv_shoup_moddown.to(device)
        self.prod_q_i_mod_q_j_moddown = material.prod_q_i_mod_q_j_moddown.to(device)

        self.hat_inverse_vec_modup = material.hat_inverse_vec_modup.to(device)
        self.hat_inverse_vec_shoup_modup = material.hat_inverse_vec_shoup_modup.to(device)
        self.prod_q_i_mod_q_j_modup = material.prod_q_i_mod_q_j_modup.to(device)

        self.inner_workspace = material.inner_workspace.to(device)
        beta = int((self.L + self.alpha - 1) / self.alpha)
        inner_workspace_numel = 16 * (self.L + self.K) * self.N * max(beta, 1)
        if self.inner_workspace.numel() < inner_workspace_numel:
            self.inner_workspace = torch.empty(
                (inner_workspace_numel,),
                dtype=torch.uint64,
                device=device,
            )
        self.mult_swk_ax = material.mult_swk_ax.to(device)
        self.mult_swk_bx = material.mult_swk_bx.to(device)

        self.inverse_power_of_roots_div_two = material.inverse_power_of_roots_div_two.to(device)
        self.inverse_scaled_power_of_roots_div_two = material.inverse_scaled_power_of_roots_div_two.to(device)
        self.power_of_roots = material.power_of_roots.to(device)
        self.power_of_roots_shoup = material.power_of_roots_shoup.to(device)

        self.q_inv_mod_q = material.q_inv_mod_q.to(device)
        self.q_inv_mod_q_shoup = material.q_inv_mod_q_shoup.to(device)
        self.qlql_inv_mod_ql_div_ql_mod_q = material.qlql_inv_mod_ql_div_ql_mod_q.to(device)
        self.qlql_inv_mod_ql_div_ql_mod_q_shoup = material.qlql_inv_mod_ql_div_ql_mod_q_shoup.to(device)

        # Dicts of device tensors.
        self.QmuplusPmu_map = {key: value.to(device) for key, value in material.QmuplusPmu_map.items()}
        self.QplusP_map = {key: value.to(device) for key, value in material.QplusP_map.items()}
        self.QmaxdiffplusPmaxdiff_map = {
            key: value.to(device) for key, value in material.QmaxdiffplusPmaxdiff_map.items()
        }
        self.QbarretKplusPbarretK_map = {
            key: value.to(device) for key, value in material.QbarretKplusPbarretK_map.items()
        }
        self.QbarretRatioplusPbarretRatio_map = {
            key: value.to(device) for key, value in material.QbarretRatioplusPbarretRatio_map.items()
        }

        # Rotation keys stay on CPU unless auto-load moves them to CUDA below.
        self.left_rot_key_map = {
            key: [value[0].to("cpu"), value[1].to("cpu")]
            for key, value in material.left_rot_key_map.items()
        }
        self.precompute_auto_map = {key: value.to("cpu") for key, value in material.precompute_auto_map.items()}
        self.inverse_precompute_auto_map = {
            key: value.to("cpu") for key, value in material.inverse_precompute_auto_map.items()
        }

        # Encode tables.
        self.encode_params_ksiPows = material.encode_params_ksiPows
        self.encode_params_rotGroup = material.encode_params_rotGroup
        self.encode_bitrev_indices = material.encode_bitrev_indices
            
        if options.resolved_auto_load_keys(device) and device == "cuda":
            for key, value in self.left_rot_key_map.items():
                self.left_rot_key_map[key] = [
                    value[0].cuda(), value[1].cuda()
                ]
            for key, value in self.precompute_auto_map.items():
                self.precompute_auto_map[key] = value.cuda()
            for key, value in self.inverse_precompute_auto_map.items():
                self.inverse_precompute_auto_map[key] = value.cuda()

    def construct_copy(self, device):
        return Context(
            self,
            device,
            self.options,
            native_context_gen=self.native_context_gen,
            generation_metadata=self.context_generation_config,
            roots_q=getattr(self, "rootsQ", None),
            roots_p=getattr(self, "rootsP", None),
        )

    def cuda(self):
        return self.construct_copy("cuda")

    def cpu(self):
        return self.construct_copy("cpu")

    def scale_at(self, cur_limbs=None):
        return self.approxSF

    def big_scale_at(self, cur_limbs=None):
        return self.approxSF

    def rescale_divisor_at(self, drop_limb=None):
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

    def get_inverse_precompute_auto(self, key):
        if self.device == "cuda" and not self.inverse_precompute_auto_map[key].is_cuda:
            return self.inverse_precompute_auto_map[key].cuda()
        else:
            return self.inverse_precompute_auto_map[key]

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
        s.append(f"{'scale_mode:':20} {self.scale_mode}")
        s.append(f"{'rescale_policy:':20} {self.rescale_policy}")
        s.append(f"{'secretKeyDist:':20} {self.secretKeyDist}")
        s.append(f"{'device:':20} {self.device}")
        return "<Context>\n" + "\n".join(s)
