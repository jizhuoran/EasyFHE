# FHE API Inventory

Snapshot: 2026-05-21. This is a proposed ownership split for the EasyFHE FHE
surface, based on the current Python modules and native `native_functions.yaml`.

The target split is:

1. stable public: `import easyfhe.fhe as fhe`, normal users should call these.
2. expert public: bootstrapping, ResNet20, benchmarks, and advanced workflows can call these.
3. internal Python: implementation modules used by public APIs.
4. native private: `torch.*`/ATen kernels; Python should wrap these before user code sees them.

Current note: package roots should stay aligned with this split; boundary tests
guard against examples importing private implementation modules directly.

## 1. Stable Public

Package target: `easyfhe.fhe as fhe`.

These are the names ordinary users should be able to rely on.

### Data And Context

- `Cipher`
- `Context`
- `CKKSContextSpec`
- `ConstantBundle`
- `generate_client_context`
- `CipherState`

`Context` public methods should stay narrow:

- `cuda()`
- `cpu()`

Client public methods:

- `encrypt(...)`
- `decrypt(...)`

### Cipher Arithmetic

- `homo_add`
- `homo_add_inplace`
- `homo_sub`
- `homo_sub_inplace`
- `homo_mul`
- `homo_mul_rescale`
- `homo_mul_rescale_addscalar`
- `homo_mul_rescale_addpt`

### Plaintext And Scalar Arithmetic

- `homo_add_pt`
- `homo_add_pt_inplace`
- `homo_mul_pt`
- `homo_mul_pt_inplace`
- `homo_add_scalar_double`
- `homo_add_scalar_double_inplace`
- `homo_add_scalar_int`
- `homo_add_scalar_int_inplace`
- `homo_sub_scalar_int`
- `homo_sub_scalar_int_inplace`
- `homo_mul_scalar_double`
- `homo_mul_scalar_double_inplace`
- `homo_mul_scalar_int`
- `homo_mul_scalar_int_inplace`

### Rotation And Shape

- `homo_rotate`
- `fast_rotate`
- `hoisted_mac_sum`
- `giant_rotate_sum`
- `moddown_from_ext`
- `fused_grouped_pairwise_mac`
- `slot_resize`

### Manual Level Control

- `align_to`
- `reduce_noise_to_one`

## 2. Expert Public

These APIs are allowed for bootstrapping, examples, benchmarks, and advanced
manual pipelines, but should not be presented as the default user surface.

### Bootstrapping Packages

`easyfhe.bs.openfhe`:

- `depth`
- `generate`
- `bootstrap`
- `BootstrapPlan`

Notes:

- `depth(...)` is the public name for `bootstrap_depth(...)`; keep one name at
  package root.
- `generate(...)` wraps constant generation and returns
  `(required_rotations, constants, plan)`.
- `bootstrap(...)` calls the runtime and currently returns a cipher reduced to
  `noise_deg == 1`.
- `generate_bootstrap_constants`, `bootstrap_depth`, `bootstrap_approx_depth`,
  `required_rotations`, and `required_plaintexts` are implementation/setup
  helpers, not package-root public API.

`easyfhe.bs.cheddar` should stay hidden for now.

## 3. Internal Python

These should be importable by implementation code, but not documented for user
or benchmark code. The main cleanup goal is to keep these out of package-level
`__all__`.

### FHE Kernel Wrappers

Module: `easyfhe.fhe.ops.kernels`.

- `cv_check`
- `gen_scalar_tensor`
- `cv_neg`
- `cv_add`
- `cv_sub`
- `cv_mul`
- `cv_add_scalar`
- `cv_sub_scalar`
- `cv_mul_scalar`
- `cv_modup`
- `cv_moddown`
- `cv_moddown_write`
- `cv_innerproduct`
- `cv_innerproduct_write`
- `cv_innerproduct_broadcast_cipher_pair`
- `cv_fast_rotate_ext_batch_finalize`
- `cv_fast_rotate_ext_batch_finalize_compact`
- `cv_fast_rotate_batch_finalize`
- `cv_fast_rotate_batch_finalize_compact`
- `cv_keyswitch`
- `cv_hrot`
- `cv_hmul_double_rescale`
- `cv_drop_last_element_and_scale`
- `cv_automorphism_transform`
- `cv_mul_by_monomial`
- `cipher_fused_grouped_pairwise_mac`
- `cipher_fused_broadcast_mac`
- `cipher_scalar_weighted_acc`
- `cipher_grouped_scalar_weighted_acc`

### FHE Primitive Helpers

Module: `easyfhe.fhe.ops.primitives`.

- `_scalar_tensor`
- `_fused_cuda_available`
- `_can_fuse_pairwise`
- `_assign_out`
- `_component_shape`
- `_can_write_out`
- `_metadata_like`
- `_finish_out`
- `_cipher_add`
- `_cipher_add_ext`
- `_cipher_sub`
- `_cipher_sub_ext`
- `_cipher_add_plain`
- `_cipher_mul_plain`
- `_cipher_mul`
- `_cipher_square`
- `_cipher_add_scalar`
- `_cipher_sub_scalar`
- `_cipher_mul_scalar_double`
- `_cipher_mul_scalar_int`
- `_cipher_neg`

### Validation, Runtime, Material, And Fixture Internals

- `validate_cipher_op`
- `validate_binary_cipher_op`
- `validate_cipher_plain_op`
- `validate_cipher_scalar_op`
- `validate_matching_metadata`
- `PreparedPlaintext`
- `Plaintext`
- `state_of`
- `consumed_depth`
- `rescale_one_level`
- `has_target_scale`
- `plan_add_alignment`
- `plan_mul_alignment`
- `plan_reduce_noise_to_one`
- `extract_cv`
- `encode_stage1`
- `encode_stage2`
- `fused_broadcast_mac`
- `homo_mul_double_rescale`
- `homo_square`
- `Client`
- `ContextMaterialBuilder`
- `CkksSamplerConfig`
- `NativeClientMaterial`
- `NativeContextBundle`
- `NativeServerMaterial`
- `sample_native_client_server`
- `sample_native_context`
- `sample_native_rotation_keys`
- `split_native_client_server`
- `_decrypt_phase`
- `mod_inverse`
- `generate_bootstrap_constants`
- `context_requirements`
- `bootstrap_depth`
- `bootstrap_approx_depth`
- `required_rotations`
- `required_plaintexts`

### Bootstrap Internals

Modules:

- `easyfhe.bs.openfhe.internal.*`
- `easyfhe.bs.cheddar.internal.*`

Names:

- `eval_bootstrapping_chebyshev`
- `apply_double_angle_iterations`
- `eval_bootstrap_approx_mod`
- `ChebyshevPSNode`
- `BootstrapApproxPlan`
- `FlatPSSmallSpec`
- `FlatPSCombineSpec`
- `FlatPSPlan`
- `degree`
- `long_division_chebyshev`
- `compile_flat_ps_plan`
- `get_bootstrap_approx_plan`
- `BootstrapTransformStep`
- `BootstrapTransformPlan`
- `CKKS_Boot_Params`
- `round_half_away_from_zero`
- `BsContext`
- `BootstrapFFTParams`
- `LinearTransformPlan`
- `reduce_rotation`
- `select_layers`
- `collapsed_fft_params`
- `coeffs_to_slots_rotation_indices`
- `slots_to_coeffs_rotation_indices`
- `linear_transform_plan`
- `bootstrap_core_rotation_indices`
- `bootstrap_auto_index_map`
- `bootstrap_rotation_indices`
- `coeffs_slots_conversion`
- `eval_coeffs_to_slots`
- `eval_slots_to_coeffs`
- `eval_bootstrap`
- `homo_bootstrap`

## 4. Native Private

These are ATen/native entry points. They should be called through Python wrappers
only. User code, bootstrapping code, and examples should not call them directly.

### Sampling And Encoding

- `torch.fhe_native_sample_ckks`
- `torch.fhe_native_sample_rotation_keys`
- `torch.encode`
- `torch.encrypt`
- `torch.pre_encode`

### Basic Modular Ops

- `torch.neg_mod`
- `torch.neg_mod_`
- `torch.add_mod`
- `torch.add_mod_`
- `torch.sub_mod`
- `torch.sub_mod_`
- `torch.mul_mod`
- `torch.mul_mod_`
- `torch.add_scalar_mod`
- `torch.add_scalar_mod_`
- `torch.sub_scalar_mod`
- `torch.sub_scalar_mod_`
- `torch.mul_scalar_mod`
- `torch.mul_scalar_mod_`

### Fused Two-Component Ops

- `torch.cv_add_pair`
- `torch.cv_add_pair_`
- `torch.cv_sub_pair`
- `torch.cv_sub_pair_`
- `torch.cv_mul_pt_pair`
- `torch.cv_mul_pt_pair_`
- `torch.cv_mul_scalar_pair`
- `torch.cv_mul_scalar_pair_`
- `torch.mul_pt_broadcast`
- `torch.mul_pt_pairwise`
- `torch.add_pt_broadcast`
- `torch.add_pt_pairwise`

### Keyswitch, Rotation, Modup, Moddown

- `torch.modup`
- `torch.moddown`
- `torch.moddown_write`
- `torch.innerproduct`
- `torch.innerproduct_write`
- `torch.innerproduct_write_pair`
- `torch.innerproduct_broadcast_cipher`
- `torch.innerproduct_broadcast_cipher_pair`
- `torch.drop_last_element_and_scale`
- `torch.automorphism_transform`
- `torch.fast_rotate_ext_batch_finalize`
- `torch.fast_rotate_ext_batch_finalize_compact`
- `torch.fast_rotate_ext_batch_finalize_pair`
- `torch.fast_rotate_batch_finalize`
- `torch.fast_rotate_batch_finalize_compact`
- `torch.hrot`
- `torch.mod_raise`
- `torch.extend_ciphertext`
- `torch.mul_by_monomial`
- `torch.mul_by_monomial_`
- `torch.mul_by_monomial.out`

### Multiply And Fused Accumulation

- `torch.hmul_double_rescale`
- `torch.batched_pairwise_mac`
- `torch.fused_broadcast_mac`
- `torch.scalar_weighted_acc`
- `torch.grouped_scalar_weighted_acc`
- `torch.cpmul_broadcast_pt`

## Cleanup Implications

- `easyfhe.fhe.__all__` should become the stable public list only.
- `easyfhe.fhe.ops.__all__` may mirror the stable operation list for backward
  compatibility, but `easyfhe.fhe.__all__` is the canonical stable surface.
- OpenFHE bootstrapping package `__all__` should stay small: `BootstrapPlan`,
  `bootstrap`, `depth`, `generate`.
- Cheddar bootstrapping should stay hidden at package root for now.
- `ops.kernels`, `ops.primitives`, and `bs.*.internal` should be treated as
  unstable implementation modules.
- Native `torch.*` FHE ops should be documented only in native/kernel developer
  notes, not in user-facing API docs.
