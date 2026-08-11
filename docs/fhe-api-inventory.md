# FHE API Inventory

Snapshot: 2026-08-11.

The ownership split is:

1. stable public: names exported by `easyfhe.fhe`;
2. expert public: the small `easyfhe.bs.openfhe` package root;
3. internal Python: implementation modules;
4. native private: ATen entry points wrapped by `ops.kernels`.

## Stable `easyfhe.fhe` API

### Context and values

- `Cipher`, `CipherState`, `EncodedScalar`
- `Client`, `Context`, `ContextParams`
- `CKKSContextSpec`, `PrimeChainPlan`
- `generate_client_context`, `plan_prime_chain`
- `ConstantBundle`, `PackedRaw`, `encode_scalar`

### Alignment

- `align_to`
- `normalize_scale`
- `rescale`

### Arithmetic

- `homo_add`, `homo_add_inplace`
- `homo_sub`, `homo_sub_inplace`
- `homo_mul_no_relin`, `homo_relinearize`, `homo_mul_relin`, `homo_mul_i`
- `homo_add_pt`, `homo_add_pt_inplace`
- `homo_mul_pt`, `homo_mul_pt_inplace`, `homo_mul_pt_rescale`
- `homo_add_scalar`, `homo_add_scalar_inplace`
- `homo_sub_scalar`, `homo_sub_scalar_inplace`
- `homo_mul_scalar`, `homo_mul_scalar_inplace`, `homo_mul_scalar_rescale`
- `homo_mul_relin_rescale_postop`
- `homo_mul_relin_rescale_add_scalar`
- `homo_mul_relin_rescale_add_pt`

### Rotation and grouped operations

- `homo_rotate`, `homo_rotate_add`
- `fast_rotate`, `moddown_from_ext`
- `grouped_pairwise_mac`, `grouped_pairwise_mac_rescale`
- `grouped_scalar_weighted_acc`
- `giant_rotate_sum`
- `hoisted_mac_sum`, `hoisted_mac_sum_rescale`
- `sum_cipher_batch`

Hoist strategies are the strings `normal`, `ext_normal`, and
`ext_double_hoist`; public `HOIST_*` constants do not exist.

### Shape and batching

- `expand_slots`, `fold_slots`
- `pack_cipher_batch`, `unpack_cipher_batch`

## Expert `easyfhe.bs.openfhe` API

- `BootstrapSpec`
- `BootstrapRequirements`
- `BootstrapProgram`
- `requirements`
- `generate`
- `bootstrap`
- `describe_plan`

`BootstrapProgram` owns the generated constants, transform plan, raise target,
mode, and exact output `CipherState`.

## Internal Python

The following categories are intentionally not package-root API:

- stage encoders: `PreparedPlaintext`, `encode_stage1`,
  `encode_stage1_packed`, `encode_stage2`;
- validation and planning helpers under `easyfhe.fhe.ops`;
- primitive operations beginning with `_cipher_`;
- kernel wrappers beginning with `cv_` or `cipher_`;
- key-generation material builders and native samplers;
- OpenFHE generation/runtime modules and their transform schedules;
- direct mod-up, key-switch, automorphism-map, and modulus-raise helpers.

Internal names can change without a compatibility alias. Applications and
examples should import only the two public roots.

## Native private

ATen functions such as modular arithmetic, paired ciphertext kernels,
keyswitch/rotation kernels, encoding, encryption, rescale, modulus raise, and
grouped accumulation are private implementation entry points. Calls belong in
`easyfhe.fhe.ops.kernels` (native key sampling is the one sampler exception),
not in application, bootstrap, or example code.

Boundary tests enforce this ownership split and the package-root allowlists.
