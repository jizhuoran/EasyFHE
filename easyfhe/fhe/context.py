import easyfhe as torch
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class ContextParams:
    depth: int
    log_n: int
    dnum: int
    q_prime_bits: tuple[int, ...]
    q_primes: tuple[int, ...]
    p_primes: tuple[int, ...]
    secret_key_dist: str
    scale_mode: str
    rescale_policy: str


def _validate_device(device):
    device = str(device)
    if device != "cpu" and device != "cuda" and not (
        device.startswith("cuda:") and device[5:].isdigit()
    ):
        raise ValueError(f"device must be 'cpu', 'cuda', or 'cuda:<index>', got {device!r}")
    return device


def _resolved_auto_load_keys(value, device):
    if value is not None:
        return bool(value)
    return str(device) == "cuda"


class Context:
    def __init__(
        self,
        material,
        device,
        *,
        auto_load_keys=None,
        rotation_key_limb_limits=None,
        roots_q=None,
        roots_p=None,
    ):
        # Context metadata.
        self._source_material = getattr(material, "_source_material", material)
        self.device = _validate_device(device)
        self.auto_load_keys = auto_load_keys
        self.auto_load_keys_resolved = _resolved_auto_load_keys(auto_load_keys, device)
        self.rotation_key_limb_limits = dict(rotation_key_limb_limits or {})
        self._roots_q = None if roots_q is None else np.asarray(roots_q, dtype=np.uint64)
        self._roots_p = None if roots_p is None else np.asarray(roots_p, dtype=np.uint64)

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
        self.logN = material.logN
        self.logNh = material.logNh
        self.auxModSize = material.auxModSize
        self.scale_mode = material.scale_mode
        self.rescale_policy = material.rescale_policy
        self.dcrtBits = material.dcrtBits
        self.secretKeyDist = material.secretKeyDist
        self.sigma = material.sigma
        self.moduliP_scalar = material.moduliP_scalar
        self.moduliQ_scalar = material.moduliQ_scalar
        self.scalingFactorsReal = material.scalingFactorsReal
        self.scalingFactorsRealBig = material.scalingFactorsRealBig
        self.params = ContextParams(
            depth=int(self.L) - 1,
            log_n=int(self.logN),
            dnum=int(self.dnum),
            q_prime_bits=tuple(int(bit) for bit in material.dcrtBitsList),
            q_primes=tuple(int(prime) for prime in self.moduliQ_scalar),
            p_primes=tuple(int(prime) for prime in self.moduliP_scalar),
            secret_key_dist=str(self.secretKeyDist),
            scale_mode=str(self.scale_mode),
            rescale_policy=str(self.rescale_policy),
        )

        # Device tensors.
        self.primes = material.primes.to(self.device)
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
        self.key_product_identity_indices = torch.arange((1 << 16) - 1, dtype=torch.int64, device=device)
        self.precompute_auto_maps_cache = {}
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
        self._rotation_key_cuda_cache = {}
        self._precompute_auto_cuda_cache = {}
        self._inverse_precompute_auto_cuda_cache = {}

        # Encode tables.
        self.encode_params_ksiPows = material.encode_params_ksiPows.to(device)
        self.encode_params_rotGroup = material.encode_params_rotGroup.to(device)
        self.encode_bitrev_indices = {
            key: value.to(device) for key, value in material.encode_bitrev_indices.items()
        }
            
    def construct_copy(self, device):
        return Context(
            self._source_material,
            device,
            auto_load_keys=self.auto_load_keys,
            rotation_key_limb_limits=self.rotation_key_limb_limits,
            roots_q=self._roots_q,
            roots_p=self._roots_p,
        )

    def cuda(self):
        return self.construct_copy("cuda")

    def cpu(self):
        return self.construct_copy("cpu")

    @property
    def max_limbs(self):
        return int(self.L)

    @property
    def ring_dim(self):
        return int(self.N)

    @property
    def max_slots(self):
        return int(self.N) // 2

    @property
    def q_prime_bits(self):
        return self.params.q_prime_bits

    @property
    def default_scale(self):
        return float(self.approxSF)

    def scale_at(self, cur_limbs=None):
        if cur_limbs is None:
            cur_limbs = self.L
        if self.scale_mode == "flexible":
            level = self.L - int(cur_limbs)
            if 0 <= level < len(self.scalingFactorsReal):
                return self.scalingFactorsReal[level]
        return self.approxSF

    def big_scale_at(self, cur_limbs=None):
        if cur_limbs is None:
            cur_limbs = self.L
        if self.scale_mode == "flexible":
            level = self.L - int(cur_limbs)
            if 0 <= level < len(self.scalingFactorsRealBig):
                return self.scalingFactorsRealBig[level]
        return self.approxSF

    def rescale_divisor_at(self, drop_limb=None):
        if drop_limb is None:
            drop_limb = 0
        if self.scale_mode == "flexible":
            return float(self.moduliQ_scalar[int(drop_limb)])
        return self.approxSF

    def physical_rescale_divisor_for_limbs(self, cur_limbs, drop_count=1):
        cur_limbs = int(cur_limbs)
        drop_count = int(drop_count)
        if drop_count <= 0:
            raise ValueError(f"drop_count must be positive, got {drop_count}")
        if cur_limbs < drop_count:
            raise ValueError(f"cannot drop {drop_count} limbs from cur_limbs={cur_limbs}")
        divisor = 1
        for idx in range(cur_limbs - drop_count, cur_limbs):
            divisor *= int(self.moduliQ_scalar[idx])
        return float(divisor)

    def get_rotation_key(self, rot_index):
        if self.device == "cuda" and not self.left_rot_key_map[rot_index][0].is_cuda:
            return [self.left_rot_key_map[rot_index][0].cuda(), self.left_rot_key_map[rot_index][1].cuda()]
        else:
            return self.left_rot_key_map[rot_index]

    def get_rotation_key_for_limbs(self, rot_index, cur_limbs):
        cur_limbs = int(cur_limbs)
        available_special_mod_start = self._rotation_key_special_mod_start(rot_index)
        if cur_limbs > available_special_mod_start:
            raise ValueError(
                f"rotation key {rot_index} has {available_special_mod_start} Q limbs, "
                f"but rotation needs {cur_limbs}"
            )
        if self.device != "cuda":
            return self.left_rot_key_map[rot_index], available_special_mod_start
        if not self.auto_load_keys_resolved:
            return self.get_rotation_key(rot_index), available_special_mod_start

        beta = int((cur_limbs + self.alpha - 1) / self.alpha)
        rot_index = int(rot_index)
        cached = self._find_cached_rotation_key(rot_index, cur_limbs, beta)
        if cached is not None:
            cached_key, cached_special_mod_start = cached
            return cached_key, cached_special_mod_start

        cache_key = (rot_index, cur_limbs, beta)
        self._evict_dominated_rotation_key_versions(cache_key)
        swk_bx, swk_ax = self._load_rotation_key_to_cuda(
            rot_index,
            cur_limbs,
            beta,
            available_special_mod_start,
        )
        cached = [swk_bx, swk_ax]
        self._rotation_key_cuda_cache[cache_key] = cached
        return cached, cur_limbs

    def _find_cached_rotation_key(self, rot_index, cur_limbs, beta):
        best = None
        for (cached_rot, cached_special_mod_start, cached_beta), cached_key in self._rotation_key_cuda_cache.items():
            if int(cached_rot) != int(rot_index):
                continue
            if int(cached_special_mod_start) < int(cur_limbs):
                continue
            if int(cached_beta) < int(beta):
                continue
            if best is None or int(cached_special_mod_start) < int(best[0]):
                best = (int(cached_special_mod_start), cached_key)
        if best is None:
            return None
        cached_special_mod_start, cached_key = best
        return cached_key, cached_special_mod_start

    def _evict_dominated_rotation_key_versions(self, retained_key):
        """Keep only non-dominated CUDA versions for one rotation.

        A key compacted for at least as many Q limbs and decomposition digits
        can serve every request handled by a smaller version.  Requests often
        reach a rotation first at a low level and later at a higher one.  Drop
        versions dominated by the requested shape before uploading the larger
        tensors, avoiding both persistent duplication and an unnecessary
        upgrade-time peak.  Tensor destruction remains stream ordered by the
        CUDA caching allocator, so in-flight work keeps its storage alive
        until the stream has passed the last queued use.
        """

        retained_rot, retained_special_mod_start, retained_beta = retained_key
        for candidate in tuple(self._rotation_key_cuda_cache):
            if candidate == retained_key:
                continue
            cached_rot, cached_special_mod_start, cached_beta = candidate
            if int(cached_rot) != int(retained_rot):
                continue
            if int(cached_special_mod_start) > int(retained_special_mod_start):
                continue
            if int(cached_beta) > int(retained_beta):
                continue
            del self._rotation_key_cuda_cache[candidate]

    def _rotation_key_special_mod_start(self, rot_index):
        bx, _ = self.left_rot_key_map[rot_index]
        available_q_limbs = int(bx.shape[1]) - int(self.K)
        configured = self.rotation_key_limb_limits.get(rot_index, self.L)
        return min(int(configured), available_q_limbs)

    def _load_rotation_key_to_cuda(self, rot_index, special_mod_start, beta, available_special_mod_start):
        bx, ax = self.left_rot_key_map[rot_index]
        return [
            self._compact_rotation_key_component_to_cuda(bx, special_mod_start, beta, available_special_mod_start),
            self._compact_rotation_key_component_to_cuda(ax, special_mod_start, beta, available_special_mod_start),
        ]

    def _compact_rotation_key_component_to_cuda(self, component, special_mod_start, beta, available_special_mod_start):
        component = component[:beta]
        q_part = component[:, :special_mod_start, :]
        p_part = component[:, available_special_mod_start:available_special_mod_start + self.K, :]
        compact = torch.cat((q_part, p_part), dim=1).contiguous()
        return compact.cuda()

    def get_precompute_auto(self, key):
        if self.device == "cuda" and not self.precompute_auto_map[key].is_cuda:
            if self.auto_load_keys_resolved:
                cached = self._precompute_auto_cuda_cache.get(key)
                if cached is None:
                    cached = self.precompute_auto_map[key].cuda()
                    self._precompute_auto_cuda_cache[key] = cached
                return cached
            return self.precompute_auto_map[key].cuda()
        return self.precompute_auto_map[key]

    def get_inverse_precompute_auto(self, key):
        if self.device == "cuda" and not self.inverse_precompute_auto_map[key].is_cuda:
            if self.auto_load_keys_resolved:
                cached = self._inverse_precompute_auto_cuda_cache.get(key)
                if cached is None:
                    cached = self.inverse_precompute_auto_map[key].cuda()
                    self._inverse_precompute_auto_cuda_cache[key] = cached
                return cached
            return self.inverse_precompute_auto_map[key].cuda()
        return self.inverse_precompute_auto_map[key]

    def clear_cuda_rotation_cache(self, *, keep_rotations=(), empty_allocator_cache=True):
        """Release reloadable CUDA rotation material.

        Rotation keys and automorphism maps are kept on the host when
        ``auto_load_keys`` is enabled.  Long heterogeneous programs may load
        disjoint key sets for successive operators, so retaining every device
        copy can unnecessarily determine peak memory.  This method removes
        the device-side copies while preserving the host master material; a
        later rotation transparently reloads what it needs.

        ``keep_rotations`` uses the same signed offsets accepted by
        :func:`easyfhe.fhe.homo_rotate`.  The returned counters make cache
        eviction visible to application-level profilers without exposing the
        private cache dictionaries.
        """

        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)

        half_ring = int(self.N) // 2

        def normalize(rotation):
            rotation = int(rotation)
            return half_ring + rotation if rotation < 0 else rotation

        keep_indices = {normalize(rotation) for rotation in keep_rotations}

        def rotation_from_key(key):
            candidate = key[0] if isinstance(key, tuple) and key else key
            try:
                return normalize(candidate)
            except (TypeError, ValueError):
                return None

        stats = {}
        for name in (
            "_rotation_key_cuda_cache",
            "_precompute_auto_cuda_cache",
            "_inverse_precompute_auto_cuda_cache",
        ):
            cache = getattr(self, name, None)
            before = len(cache) if isinstance(cache, dict) else 0
            if isinstance(cache, dict):
                for key in tuple(cache):
                    rotation = rotation_from_key(key)
                    if rotation is None or rotation not in keep_indices:
                        del cache[key]
            stats[f"{name}_entries"] = int(before)
            stats[f"{name}_retained_entries"] = (
                int(len(cache)) if isinstance(cache, dict) else 0
            )

        batch_cache = getattr(self, "precompute_auto_maps_cache", None)
        before = len(batch_cache) if isinstance(batch_cache, dict) else 0
        if isinstance(batch_cache, dict):
            for key in tuple(batch_cache):
                offsets = key[0] if isinstance(key, tuple) and key else ()
                if isinstance(offsets, (int, np.integer)):
                    offsets = (int(offsets),)
                if not offsets or any(
                    normalize(offset) not in keep_indices for offset in offsets
                ):
                    del batch_cache[key]
        stats["precompute_auto_maps_cache_entries"] = int(before)
        stats["precompute_auto_maps_cache_retained_entries"] = (
            int(len(batch_cache)) if isinstance(batch_cache, dict) else 0
        )

        if (
            bool(empty_allocator_cache)
            and str(self.device).startswith("cuda")
        ):
            torch.cuda.empty_cache()
        stats["cpu_master_rotation_keys_retained"] = int(
            len(getattr(self, "left_rot_key_map", {}))
        )
        stats["requested_rotation_indices_kept"] = int(len(keep_indices))
        return stats

    def __repr__(self):
        s = []
        s.append(f"{'max_limbs:':20} {self.max_limbs}")
        s.append(f"{'ring_dim:':20} {self.ring_dim}")
        s.append(f"{'dnum:':20} {self.dnum}")
        s.append(f"{'q_prime_bits:':20} {self.q_prime_bits}")
        s.append(f"{'K:':20} {self.K}")
        s.append(f"{'scale_mode:':20} {self.scale_mode}")
        s.append(f"{'rescale_policy:':20} {self.rescale_policy}")
        s.append(f"{'secretKeyDist:':20} {self.secretKeyDist}")
        s.append(f"{'device:':20} {self.device}")
        return "<Context>\n" + "\n".join(s)
