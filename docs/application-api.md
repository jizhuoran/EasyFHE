# EasyFHE Application API

This page describes the API surface intended for application code. It is the
contract for code that uses EasyFHE as a CKKS runtime, not for code that
implements bootstrap internals or native kernels.

Use package roots:

```python
import easyfhe.fhe as fhe
import easyfhe.bs.openfhe as bs
```

Importable submodules under `easyfhe.fhe` and `easyfhe.bs.openfhe` are
implementation details unless their symbols are re-exported by the package root.

Detailed references:

- [EasyFHE FHE API](fhe-api.md)
- [OpenFHE Bootstrap API](openfhe-bootstrap-api.md)

## Contexts

Use `CKKSContextSpec` and `generate_client_context` to build the paired client
and server context.

```python
extra_depth = bs.depth(
    log_bs_slots=14,
    level_budget=(4, 4),
    secret_key_dist="SPARSE_TERNARY",
)

client, ctx = fhe.generate_client_context(
    fhe.CKKSContextSpec(
        depth=10 + extra_depth,
        log_n=16,
        dnum=3,
        dcrt_bits=52,
        first_mod=55,
        secret_key_dist="SPARSE_TERNARY",
        scale_mode="fixed",
        rescale_policy="manual",
        rotations=(-1024, -256, -64, 1, 2, 4),
        auto_load_keys=True,
    ),
    device="cuda",
)
```

Application-facing client methods:

```python
client.encrypt(values, device=None, scale_deg=1, level=0, slots=0)
client.decrypt(cipher)
```

Application-facing context methods:

```python
ctx.cuda()
ctx.cpu()
ctx.scale_at(cur_limbs=None)
ctx.big_scale_at(cur_limbs=None)
ctx.rescale_divisor_at(drop_limb=None)
```

The following are internal/debug helpers and should not be used by application
code:

```python
ctx.get_rotation_key(...)
ctx.get_precompute_auto(...)
ctx.get_inverse_precompute_auto(...)
```

## Constants

Use `ConstantBundle` for reusable plaintext constants and weights.

```python
weights = fhe.ConstantBundle(
    vectors={
        "kernel": raw_kernel_vectors,
        "bias": raw_bias_vector,
    },
    scalars={
        "scale": 0.125,
    },
    cache_mode="plain",
    plain_cache_limit_gb=None,
    plain_cache_policy="first_fit",
)

pt = weights.plaintext(
    "kernel",
    level=ctx.L - cipher.state.cur_limbs,
    slots=cipher.slots,
    cryptoContext=ctx,
    scale=1.0,
    is_ext=False,
)
```

Application-facing constant methods:

```python
bundle.plaintext(name, level, slots, ctx, scale=1.0, is_ext=False, cache=True)
bundle.encoded_scalars(names, cur_limbs, noise_deg, ctx, mode="double", absolute=False, cache=True)
bundle.cache_info()
bundle.memory_info()
bundle.clear_cache()
bundle.set_cache_mode(mode, clear=True)
bundle.set_plain_cache_limit_gb(limit_gb, clear=False)
bundle.set_plain_cache_policy(policy)
```

Raw scalar reads, middle encoding, and vector padding are internal implementation
details. Application code should access constants by name through
`plaintext(...)` or `encoded_scalars(...)` and let the bundle manage preparation,
materialization, batching, and caching.

Use `cache_mode="plain"` to cache only final plaintexts, `cache_mode="middle"`
to cache only prepared middle encodings, and `cache_mode="both"` to cache both.
These three modes do not support a plaintext cache limit; if cached plaintexts
do not fit in memory, allocation fails normally.

Use `cache_mode="mix"` with `plain_cache_limit_gb=<size>` when you want a
bounded plaintext cache. Plaintexts are cached until the limit is reached; later
constants keep only their prepared middle encoding and run stage2 on demand.
Cached plaintexts do not keep duplicate middle encodings.

Set `plain_cache_policy="small_first"` to prefer smaller plaintexts when the
`mix` plaintext cache is full. In `cache_mode="mix"`, evicted plaintexts keep
their middle encoding cached as the fallback.

## Basic Homomorphic Ops

Use these for normal ciphertext arithmetic:

```python
fhe.homo_add(a, b, ctx)
fhe.homo_sub(a, b, ctx)
fhe.homo_mul_relin(a, b, ctx)

fhe.homo_add_pt(cipher, plaintext, ctx)
fhe.homo_mul_pt(cipher, plaintext, ctx)

encoded = bundle.encoded_scalars("scale", cipher.state.cur_limbs, cipher.state.noise_deg, ctx, mode="double")[0]
fhe.homo_mul_scalar_double(cipher, encoded, ctx)

shift = bundle.encoded_scalars("shift", cipher.state.cur_limbs, 0, ctx, mode="int")[0]
fhe.homo_add_scalar_int(cipher, shift, ctx)

fhe.homo_rotate(cipher, offset, ctx)
fhe.homo_rotate_add(cipher, offset, ctx, addend=other)
fhe.expand_slots(cipher, slots, ctx)
fhe.fold_slots(cipher, slots, ctx, mask=mask)
```

`expand_slots(...)` only updates ciphertext metadata and requires the target
slot count to be at least the current slot count. `fold_slots(...)` reduces the
slot count with an explicit plaintext mask encoded at the source slot count.

Use `align_to` when an application needs explicit state alignment:

```python
target = fhe.CipherState(cur_limbs=12, noise_deg=1, scaling_factor=None)
cipher = fhe.align_to(cipher, target, ctx)
```

## Hoisted Ops

These are advanced application APIs. They are useful for BSGS-style matrix,
convolution, and linear-transform code.

```python
rotated = fhe.fast_rotate(cipher, offsets, ctx, output_ext=False)
partials = fhe.grouped_pairwise_mac(rotated, plaintexts, groups, ctx)  # batched Cipher
weighted = fhe.grouped_scalar_weighted_acc(rotated, encoded_scalars, ctx)  # batched Cipher
result = fhe.giant_rotate_sum(partials, giant_offset, ctx, strategy=fhe.HOIST_NORMAL)

result = fhe.hoisted_mac_sum(
    cipher,
    baby_offsets,
    plaintexts,
    giant_offset,
    giant_count,
    ctx,
    strategy=fhe.HOIST_NORMAL,
)
```

Strategies:

```python
fhe.HOIST_NORMAL           # fast rotate to normal ciphertexts, normal giant rotations
fhe.HOIST_EXT_NORMAL       # fast rotate to ext, multiply in ext, moddown before giants
fhe.HOIST_EXT_DOUBLE_HOIST # fast rotate to ext, multiply in ext, double-hoisted giants
```

Avoid using lower-level rotation helpers such as modup, moddown, automorphism
precompute maps, and raw rotation-key access. Those are runtime internals.

## Bootstrapping

OpenFHE-compatible bootstrapping lives in `easyfhe.bs.openfhe`.

```python
import easyfhe.bs.openfhe as bs

level_budget = (4, 4)
extra_depth = bs.depth(
    log_bs_slots=14,
    level_budget=level_budget,
    secret_key_dist="SPARSE_TERNARY",
)
rotations = bs.plan_rot_keys(
    log_n=16,
    log_bs_slots=14,
    level_budget=level_budget,
    strategy="double_hoist",
)

client, ctx = fhe.generate_client_context(
    ... depth=post_bootstrap_levels + extra_depth, rotations=rotations ...
)

bs_constants, bs_plan = bs.generate(
    ctx,
    log_bs_slots=14,
    level_budget=level_budget,
    post_bootstrap_levels=post_bootstrap_levels,
    baby_step=None,
    strategy="double_hoist",
)

cipher = bs.bootstrap(
    cipher,
    ctx,
    bs_constants,
    bs_plan,
    L0=cipher.state.cur_limbs,
    bootstrap_mode="modraise_first",
)
```

Application-facing bootstrap API:

```python
bs.depth(...)
bs.plan_rot_keys(...)
bs.generate(...)
bs.bootstrap(...)
bs.describe_plan(...)
bs.BootstrapPlan
```

Detailed reference: [OpenFHE Bootstrap API](openfhe-bootstrap-api.md).

`BootstrapPlan` is returned by `generate(...)` and passed back to
`bootstrap(...)`. Treat it as an opaque plan object; application code should not
depend on its internal fields. Use `bs.describe_plan(plan)` when debugging the
generated bootstrapping schedule.

## Non-API Internals

The following categories are intentionally outside the application API:

- OpenFHE bootstrap implementation modules such as `easyfhe.bs.openfhe.runtime.*`
  and `easyfhe.bs.openfhe.generation.*`
- native kernel wrappers under `easyfhe.fhe.ops.kernels`
- raw encoding stages
- raw key-switch, modup, moddown, automorphism, and precompute-map helpers
- debug/profiling-only helpers such as decryption phase
- OpenFHE-style compatibility aliases when a newer EasyFHE name exists

If application code needs one of these directly, the public API should probably
gain a small, named operation instead of exposing the internal helper.
