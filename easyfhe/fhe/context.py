from dataclasses import fields

from .ciphertext import Plaintext, PreparedPlaintext
from .runtime.scale_policy import split_rescale_tech
from .runtime.instrumentation import instrumentation_from_options
from .material.context_material import RuntimeContextMaterial, runtime_material_from_builder


class Context:
    @classmethod
    def build(cls, spec, device="cpu", options=None):
        from .material.context_factory import build_context

        return build_context(cls, spec, device, options)

    @classmethod
    def _from_builder(cls, builder, device, options):
        return cls(runtime_material_from_builder(builder), device, options)

    def __init__(self, material: RuntimeContextMaterial, device, options):
        L = material.L
        dnum = material.dnum
        alpha = material.alpha
        K = material.K
        M = material.M
        N = material.N
        Nh = material.Nh
        approxSF = material.approxSF
        h = material.h
        levelBudget = material.levelBudget
        logN = material.logN
        logNh = material.logNh
        logBsSlots_list = material.logBsSlots_list
        auxModSize = material.auxModSize
        rescaleTech = material.rescaleTech
        dcrtBits = material.dcrtBits
        max_num_moduli = material.max_num_moduli
        secretKeyDist = material.secretKeyDist
        sigma = material.sigma
        inBS = material.inBS
        primes = material.primes
        barret_k = material.barret_k
        barret_ratio = material.barret_ratio
        q_mu = material.q_mu
        moduliP_scalar = material.moduliP_scalar
        moduliQ_scalar = material.moduliQ_scalar
        moduliQ = material.moduliQ
        scalingFactorsReal = material.scalingFactorsReal
        scalingFactorsRealBig = material.scalingFactorsRealBig
        PModq = material.PModq
        max_int_diffs = material.max_int_diffs
        QmuplusPmu_map = material.QmuplusPmu_map
        QplusP_map = material.QplusP_map
        automorphism_transform_out = material.automorphism_transform_out
        inner_out = material.inner_out
        moddown_out_ax = material.moddown_out_ax
        moddown_out_bx = material.moddown_out_bx
        modup_out = material.modup_out
        rescale_out = material.rescale_out
        mod_raise_out = material.mod_raise_out
        hat_inverse_vec_moddown = material.hat_inverse_vec_moddown
        hat_inverse_vec_shoup_moddown = material.hat_inverse_vec_shoup_moddown
        prod_inv_moddown = material.prod_inv_moddown
        prod_inv_shoup_moddown = material.prod_inv_shoup_moddown
        prod_q_i_mod_q_j_moddown = material.prod_q_i_mod_q_j_moddown
        hat_inverse_vec_modup = material.hat_inverse_vec_modup
        hat_inverse_vec_shoup_modup = material.hat_inverse_vec_shoup_modup
        prod_q_i_mod_q_j_modup = material.prod_q_i_mod_q_j_modup
        inner_workspace = material.inner_workspace
        mult_swk_ax = material.mult_swk_ax
        mult_swk_bx = material.mult_swk_bx
        inverse_power_of_roots_div_two = material.inverse_power_of_roots_div_two
        inverse_scaled_power_of_roots_div_two = material.inverse_scaled_power_of_roots_div_two
        power_of_roots = material.power_of_roots
        power_of_roots_shoup = material.power_of_roots_shoup
        left_rot_key_map = material.left_rot_key_map
        precompute_auto_map = material.precompute_auto_map
        q_inv_mod_q = material.q_inv_mod_q
        q_inv_mod_q_shoup = material.q_inv_mod_q_shoup
        qlql_inv_mod_ql_div_ql_mod_q = material.qlql_inv_mod_ql_div_ql_mod_q
        qlql_inv_mod_ql_div_ql_mod_q_shoup = material.qlql_inv_mod_ql_div_ql_mod_q_shoup
        QmaxdiffplusPmaxdiff_map = material.QmaxdiffplusPmaxdiff_map
        encode_params_ksiPows = material.encode_params_ksiPows
        encode_params_rotGroup = material.encode_params_rotGroup
        encode_bitrev_indices = material.encode_bitrev_indices
        encode_values = material.encode_values
        QbarretKplusPbarretK_map = material.QbarretKplusPbarretK_map
        QbarretRatioplusPbarretRatio_map = material.QbarretRatioplusPbarretRatio_map

        self.device = device
        self.options = options
        self.instrumentation = instrumentation_from_options(options)
        self.inBS = inBS

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
        self.scale_mode, self.rescale_policy = split_rescale_tech(rescaleTech)
        self.dcrtBits = dcrtBits
        self.max_num_moduli = max_num_moduli
        self.secretKeyDist = secretKeyDist
        self.sigma = sigma

        # for common op
        self.primes = primes.to(device)
        self.primes_list = [int(p) for p in primes.tolist()]
        switch_map = []
        for old_mod in self.primes_list:
            for new_mod in self.primes_list:
                if old_mod > new_mod:
                    diff = new_mod - (old_mod % new_mod)
                else:
                    diff = new_mod - old_mod
                switch_map.append(diff)
        import easyfhe as torch
        self.switch_modulus_map = torch.tensor(switch_map, dtype=torch.uint64, device=device)
        self.barret_k = barret_k.to(device)
        self.barret_ratio = barret_ratio.to(device)
        self.q_mu = q_mu.to(device)
        self.moduliP_scalar = moduliP_scalar # for computation on cpu
        self.moduliQ_scalar = moduliQ_scalar # for computation on cpu
        self.moduliQ = moduliQ.to(device)
        self.scalingFactorsReal = scalingFactorsReal
        self.scalingFactorsRealBig = scalingFactorsRealBig

        # for cv_mul
        self.PModq = PModq.to(device)
        self.QmuplusPmu_map = {key: value.to(device) for key, value in QmuplusPmu_map.items()}
        self.QplusP_map =  {key: value.to(device) for key, value in QplusP_map.items()}

        # output space
        self.automorphism_transform_out = automorphism_transform_out.to(device)
        self.inner_out = inner_out.to(device)
        self.moddown_out_ax = moddown_out_ax.to(device)
        self.moddown_out_bx = moddown_out_bx.to(device)
        self.modup_out = modup_out.to(device)
        self.rescale_out = rescale_out.to(device)
        self.mod_raise_out = mod_raise_out.to(device)

        # for moddown
        self.hat_inverse_vec_moddown = hat_inverse_vec_moddown.to(device)
        self.hat_inverse_vec_shoup_moddown = hat_inverse_vec_shoup_moddown.to(device)
        self.prod_inv_moddown = prod_inv_moddown.to(device)
        self.prod_inv_shoup_moddown = prod_inv_shoup_moddown.to(device)
        self.prod_q_i_mod_q_j_moddown = prod_q_i_mod_q_j_moddown.to(device)

        # for modup
        self.hat_inverse_vec_modup = hat_inverse_vec_modup.to(device)
        self.hat_inverse_vec_shoup_modup = hat_inverse_vec_shoup_modup.to(device)
        self.prod_q_i_mod_q_j_modup = prod_q_i_mod_q_j_modup.to(device)

        # for innerproduct
        self.inner_workspace = inner_workspace.to(device)
        self.mult_swk_ax = mult_swk_ax.to(device)
        self.mult_swk_bx = mult_swk_bx.to(device)

        # for ntt&intt
        self.inverse_power_of_roots_div_two = inverse_power_of_roots_div_two.to(device)
        self.inverse_scaled_power_of_roots_div_two = inverse_scaled_power_of_roots_div_two.to(device)
        self.power_of_roots = power_of_roots.to(device)
        self.power_of_roots_shoup = power_of_roots_shoup.to(device)

        # for rotation
        self.left_rot_key_map = {key: [value[0].to("cpu"), value[1].to("cpu")] for key, value in left_rot_key_map.items()}
        self.precompute_auto_map = {key: value.to("cpu") for key, value in precompute_auto_map.items()}

        # for cv_drop
        self.q_inv_mod_q = q_inv_mod_q.to(device)
        self.q_inv_mod_q_shoup = q_inv_mod_q_shoup.to(device)
        self.qlql_inv_mod_ql_div_ql_mod_q = qlql_inv_mod_ql_div_ql_mod_q.to(device)
        self.qlql_inv_mod_ql_div_ql_mod_q_shoup = qlql_inv_mod_ql_div_ql_mod_q_shoup.to(device)

        # for encode
        self.QmaxdiffplusPmaxdiff_map = {key: value.to(device) for key, value in QmaxdiffplusPmaxdiff_map.items()}
        self.QbarretKplusPbarretK_map = {key: value.to(device) for key, value in QbarretKplusPbarretK_map.items()}
        self.QbarretRatioplusPbarretRatio_map = {key: value.to(device) for key, value in QbarretRatioplusPbarretRatio_map.items()}
        self.max_int_diffs = max_int_diffs.to(device)
        self.encode_params_ksiPows = encode_params_ksiPows
        self.encode_params_rotGroup = encode_params_rotGroup
        self.encode_bitrev_indices = encode_bitrev_indices
        self.encode_values = {}
        for key, value in encode_values.items():
            if isinstance(value, Plaintext):
                self.encode_values[key] = value.deep_copy()
                self.encode_values[key].cv = [cv.to(device) for cv in self.encode_values[key].cv]
            elif isinstance(value, PreparedPlaintext):
                self.encode_values[key] = value.deep_copy()
            else:
                raise ValueError("Unknown type in encode_values")
            
        if options.resolved_auto_load_keys(device) and device == "cuda":
            for key, value in self.left_rot_key_map.items():
                self.left_rot_key_map[key] = [
                    value[0].cuda(), value[1].cuda()
                ]
            for key, value in self.precompute_auto_map.items():
                self.precompute_auto_map[key] = value.cuda()

    def construct_copy(self, device):
        copied = Context(self.to_runtime_material(), device, self.options)
        if hasattr(self, "context_generation_config"):
            copied.context_generation_config = dict(self.context_generation_config)
        if hasattr(self, "native_context_gen"):
            copied.native_context_gen = self.native_context_gen
        if hasattr(self, "_key_material"):
            copied._key_material = self._key_material
        if hasattr(self, "_sampler_config"):
            copied._sampler_config = self._sampler_config
        return copied

    def to_runtime_material(self):
        return RuntimeContextMaterial(
            **{field.name: getattr(self, field.name) for field in fields(RuntimeContextMaterial)}
        )

    def cuda(self):
        return self.construct_copy("cuda")

    def cpu(self):
        return self.construct_copy("cpu")

    def _attach_key_material(self, key_material, sampler_config=None):
        self._key_material = key_material
        self._sampler_config = sampler_config
        self.native_context_gen = True
        return self

    def _require_key_material(self):
        key_material = getattr(self, "_key_material", None)
        if key_material is None:
            raise RuntimeError("Context has no key material for native encrypt/decrypt")
        return key_material

    def encrypt(self, x, device=None, scale_deg=1, level=0, slots=0):
        from .material.crypto import encrypt_with_key_material

        return encrypt_with_key_material(
            x,
            self,
            self._require_key_material(),
            device=device,
            scale_deg=scale_deg,
            level=level,
            slots=slots,
        )

    def decrypt_phase(self, cipher):
        from .material.crypto import decrypt_phase_with_key_material

        return decrypt_phase_with_key_material(cipher, self, self._require_key_material())

    def decrypt(self, cipher):
        from .material.crypto import decrypt_with_key_material

        return decrypt_with_key_material(cipher, self, self._require_key_material())
    
    def norm_rot_index(self, i):
        if i < 0:
            i = self.N // 2 + i
        return i

    def scale_at(self, cur_limbs=None):
        if cur_limbs is None:
            cur_limbs = self.L
        lvl = self.L - cur_limbs
        if self.scale_mode == "flexible":
            if lvl >= len(self.scalingFactorsReal):
                return self.approxSF
            return self.scalingFactorsReal[lvl]
        return self.approxSF

    def big_scale_at(self, cur_limbs=None):
        if cur_limbs is None:
            cur_limbs = self.L
        l = self.L - cur_limbs
        if self.scale_mode == "flexible":
            if l >= len(self.scalingFactorsRealBig):
                return self.approxSF
            return self.scalingFactorsRealBig[l]
        return self.approxSF

    def rescale_divisor_at(self, drop_limb=None):
        if drop_limb is None:
            drop_limb = 0
        if self.scale_mode == "flexible":
            return float(self.moduliQ_scalar[drop_limb])
        return self.approxSF

    def GetScalingFactorReal(self, cur_limbs=None):
        return self.scale_at(cur_limbs)

    def GetScalingFactorRealBig(self, cur_limbs=None):
        return self.big_scale_at(cur_limbs)

    def GetModReduceFactor(self, drop_limb=None):
        return self.rescale_divisor_at(drop_limb)

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

    def ensure_rotation_keys(self, rotation_groups):
        from .material.rotation import ensure_rotation_keys

        return ensure_rotation_keys(self, rotation_groups)

    def add_keys(self, key_requirements):
        return self.ensure_rotation_keys(self._rotation_groups_from_key_requirements(key_requirements))

    def addkeys(self, key_requirements):
        return self.add_keys(key_requirements)

    @staticmethod
    def _rotation_groups_from_key_requirements(key_requirements):
        if key_requirements is None:
            return ()

        required_rotations = getattr(key_requirements, "required_rotations", None)
        if required_rotations is not None:
            return (tuple(required_rotations),)

        if _looks_like_generate_result(key_requirements):
            return Context._rotation_groups_from_key_requirements(key_requirements[0])

        if _looks_like_requirements_result(key_requirements):
            return Context._rotation_groups_from_key_requirements(key_requirements[1])

        return key_requirements


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


def _looks_like_generate_result(value):
    if not isinstance(value, tuple) or len(value) != 3:
        return False
    return _is_sequence(value[0]) and getattr(value[2], "required_rotations", None) is not None


def _looks_like_requirements_result(value):
    if not isinstance(value, tuple) or len(value) != 3:
        return False
    extra_depth, rotations, plaintexts = value
    return isinstance(extra_depth, int) and _is_sequence(rotations) and _is_sequence(plaintexts)


def _is_sequence(value):
    if isinstance(value, (str, bytes)):
        return False
    try:
        iter(value)
    except TypeError:
        return False
    return True
