# EasyFHE Application API

This page describes the API surface intended for application code. It is the
contract for code that uses EasyFHE as a CKKS runtime, not for code that
implements bootstrap internals or native kernels.

Use package roots:

```python
import easyfhe.fhe as fhe
import easyfhe.bs.openfhe as bs
```

Importable submodules under `easyfhe.fhe` and `easyfhe.bs.openfhe.internal` are
implementation details unless their symbols are re-exported by the package root.

## Contexts

Use `CKKSContextSpec` and `generate_context` to build a context.

```python
extra_depth = bs.depth(
    log_bs_slots=14,
    level_budget=(4, 4),
    secret_key_dist="SPARSE_TERNARY",
)

ctx = fhe.generate_context(
    fhe.CKKSContextSpec(
        depth=10 + extra_depth,
        log_n=16,
        dnum=3,
        dcrt_bits=52,
        first_mod=55,
        secret_key_dist="SPARSE_TERNARY",
        rescale_tech="FIXEDMANUAL",
        rotations=(-1024, -256, -64, 1, 2, 4),
    ),
    device="cuda",
    options=fhe.RuntimeOptions(auto_load_keys=True),
)
```

Application-facing context methods:

```python
ctx.encrypt(values, device=None, scale_deg=1, level=0, slots=0)
ctx.decrypt(cipher)
ctx.cuda()
ctx.cpu()
ctx.add_keys(key_requirements)
ctx.scale_at(cur_limbs=None)
ctx.big_scale_at(cur_limbs=None)
ctx.rescale_divisor_at(drop_limb=None)
```

The following are internal/debug helpers and should not be used by application
code:

```python
ctx.decrypt_phase(...)
ctx.norm_rot_index(...)
ctx.get_rotation_key(...)
ctx.get_precompute_auto(...)
ctx.ensure_rotation_keys(...)
ctx.addkeys(...)
ctx.GetScalingFactorReal(...)
ctx.GetScalingFactorRealBig(...)
ctx.GetModReduceFactor(...)
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
)

pt = weights.plaintext(
    "kernel",
    level=ctx.L - cipher.cur_limbs,
    slots=cipher.slots,
    cryptoContext=ctx,
    scale=1.0,
    is_ext=False,
)
```

Application-facing constant methods:

```python
bundle.plaintext(name, level, slots, ctx, scale=1.0, is_ext=False, cache=True)
bundle.cache_info()
bundle.clear_cache()
bundle.set_cache_mode(mode, clear=True)
```

`encoded_scalars(...)`, raw scalar reads, middle encoding, and vector padding are
internal implementation hooks. Application code should pass named vectors to
`plaintext(...)` and let the bundle manage preparation, plaintext materialization,
batching, and caching.

## Basic Homomorphic Ops

Use these for normal ciphertext arithmetic:

```python
fhe.homo_add(a, b, ctx)
fhe.homo_sub(a, b, ctx)
fhe.homo_mul(a, b, ctx)
fhe.homo_square(a, ctx)

fhe.homo_add_pt(cipher, plaintext, ctx)
fhe.homo_mul_pt(cipher, plaintext, ctx)

fhe.homo_add_scalar_double(cipher, value, ctx)
fhe.homo_add_scalar_int(cipher, value, ctx)
fhe.homo_mul_scalar_double(cipher, value, ctx)
fhe.homo_mul_scalar_int(cipher, value, ctx)

fhe.homo_rotate(cipher, offset, ctx)
fhe.slot_resize(cipher, slots, ctx)
```

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
partials = fhe.fused_grouped_pairwise_mac(rotated, plaintexts, groups, ctx)
result = fhe.giant_rotate_sum(partials, giant_offset, ctx, strategy="normal")

result = fhe.hoisted_mac_sum(
    cipher,
    baby_offsets,
    plaintexts,
    giant_offset,
    giant_count,
    ctx,
    strategy="normal",
)
```

Strategies:

```python
"normal"           # fast rotate to normal ciphertexts, normal giant rotations
"ext_normal"       # fast rotate to ext, multiply in ext, moddown before giants
"ext_double_hoist" # fast rotate to ext, multiply in ext, double-hoisted giants
```

Avoid using lower-level rotation helpers such as modup, moddown, automorphism
precompute maps, and raw rotation-key access. Those are runtime internals.

`fused_broadcast_mac` is not recommended for application code. A single
ciphertext multiplied by many plaintexts and summed is usually better expressed
by pre-summing the plaintext constants and using one `homo_mul_pt`.

## Bootstrapping

OpenFHE-compatible bootstrapping lives in `easyfhe.bs.openfhe`.

```python
import easyfhe.bs.openfhe as bs

extra_depth = bs.depth(
    log_bs_slots=14,
    level_budget=(4, 4),
    secret_key_dist="SPARSE_TERNARY",
)

ctx = fhe.generate_context(... depth=max_levels_remaining + extra_depth ...)

bs_keys, bs_constants, bs_plan = bs.generate(
    ctx,
    log_bs_slots=14,
    level_budget=(4, 4),
    max_levels_remaining=max_levels_remaining,
    baby_step=None,
    strategy="double_hoist",
)
ctx.add_keys(bs_keys)

cipher = bs.bootstrap(cipher, ctx, bs_constants, bs_plan, L0=cipher.cur_limbs)
```

Application-facing bootstrap API:

```python
bs.depth(...)
bs.generate(...)
bs.bootstrap(...)
bs.BootstrapPlan
```

`BootstrapPlan` is returned by `generate(...)` and passed back to
`bootstrap(...)`. Treat it as an opaque plan object; application code should not
depend on its internal fields.

## Non-API Internals

The following categories are intentionally outside the application API:

- `easyfhe.bs.openfhe.internal.*`
- native kernel wrappers under `easyfhe.fhe.ops.kernels`
- raw encoding stages
- raw key-switch, modup, moddown, automorphism, and precompute-map helpers
- debug/profiling-only helpers such as decryption phase
- OpenFHE-style compatibility aliases when a newer EasyFHE name exists

If application code needs one of these directly, the public API should probably
gain a small, named operation instead of exposing the internal helper.
