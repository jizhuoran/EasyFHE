# EasyFHE Application API

Application code should import the two package roots:

```python
import easyfhe.fhe as fhe
import easyfhe.bs.openfhe as bs
```

Submodules are implementation details unless a symbol is re-exported from one
of these roots.

## Context and client

The regular u64 context form is:

```python
bootstrap_spec = bs.BootstrapSpec(
    log_slots=14,
    level_budget=(4, 4),
    output_levels=10,
)
requirements = bs.requirements(bootstrap_spec, log_n=16)

client, context = fhe.generate_client_context(
    fhe.CKKSContextSpec(
        depth=requirements.context_depth,
        log_n=16,
        dnum=3,
        dcrt_bits=59,
        first_mod=60,
        scale_mode="fixed",
        rescale_policy="manual",
        rotations=requirements.rotations,
    ),
    device="cuda",
)
```

Use `limb_specs=(60, 59, 59, ...)` instead of
`depth`/`dcrt_bits`/`first_mod` when each Q prime must be specified explicitly.
Every u64 rescale removes exactly one Q prime. Fixed mode requires equal-size
rescale primes; heterogeneous chains require flexible mode.

Client operations are:

```python
cipher = client.encrypt(values, slots=..., cur_limbs=None, scaling_factor=None)
values = client.decrypt(cipher)
```

Fresh ciphertexts have `scale_degree == 1`. Runtime state lives in:

```python
cipher.state.cur_limbs
cipher.state.scale_degree
cipher.state.scaling_factor
```

## Constants

`ConstantBundle` accepts slot-ready tensor constants. Padding and layout belong
in the application, so the requested `slots` must match the stored tensor.

```python
weights = fhe.ConstantBundle(
    vectors={"kernel": fhe.PackedRaw(kernel_tensor)},
    scalars={"gain": 0.125, "count": 4},
    cache_mode="mix_of_middle_plain",
    plain_cache_limit_gb=8,
    plain_cache_policy="small_first",
)

kernel = weights.plaintext(
    "kernel",
    state=cipher.state,
    slots=cipher.slots,
    context=context,
)
```

Plaintext state is explicit. For multiplication, use a normalized
`scale_degree=1` state at the ciphertext's limb count and scale. For addition,
use the exact output state that will receive the constant.

Scalar encoding also returns a typed value with metadata:

```python
gain = weights.encoded_scalars(
    "gain",
    cur_limbs=cipher.state.cur_limbs,
    scale_degree=1,
    scaling_factor=cipher.state.scaling_factor,
    context=context,
    mode="scaled",
)[0]

count = weights.encoded_scalars(
    "count",
    cur_limbs=cipher.state.cur_limbs,
    scale_degree=0,
    context=context,
    mode="integer",
)[0]
```

For a one-off value, use `fhe.encode_scalar(...)`. Raw residue tensors are not
accepted by scalar homomorphic operations.

Cache modes are `none`, `middle`, `plain`, `both`, and
`mix_of_middle_plain`. Only `mix_of_middle_plain` supports a plaintext cache
limit. Cache inspection and control methods are:

```python
bundle.cache_info()
bundle.memory_info()
bundle.clear_cache()
bundle.set_cache_mode(mode, clear=True)
bundle.set_plain_cache_limit_gb(limit_gb, clear=False)
bundle.set_plain_cache_policy(policy)
```

## Arithmetic and state control

```python
fhe.homo_add(a, b, context)
fhe.homo_sub(a, b, context)
fhe.homo_mul_no_relin(a, b, context)
fhe.homo_mul_relin(a, b, context)

fhe.homo_add_pt(cipher, plaintext, context)
fhe.homo_mul_pt(cipher, plaintext, context)
fhe.homo_add_scalar(cipher, encoded_scalar, context)
fhe.homo_sub_scalar(cipher, encoded_scalar, context)
fhe.homo_mul_scalar(cipher, encoded_scalar, context)
```

Add/subtract scalar metadata must match the ciphertext. Multiplication accepts
an integer scalar (`scale_degree=0`) or a normalized scaled scalar
(`scale_degree=1`). In-place variants use the `_inplace` suffix and require
already aligned inputs.

State operations are:

```python
target = fhe.CipherState(cur_limbs=12, scale_degree=1, scaling_factor=None)
cipher = fhe.align_to(cipher, target, context)
cipher = fhe.normalize_scale(cipher, context)
cipher = fhe.rescale(cipher, context)
```

`rescale` always consumes one u64 Q prime and reduces `scale_degree` by one.
`normalize_scale` brings a pending degree-two result back to degree one.

The common multiply-and-rescale forms are named directly:

```python
fhe.homo_mul_pt_rescale(cipher, plaintext, context)
fhe.homo_mul_scalar_rescale(cipher, scalar, context)
fhe.grouped_pairwise_mac_rescale(ciphers, plaintexts, groups, context)
fhe.hoisted_mac_sum_rescale(
    cipher, baby_offsets, plaintexts, giant_offset, giant_count, context,
    strategy="normal",
)
```

Ciphertext multiplication also has a fused relin/rescale/post-op API:

```python
fhe.homo_mul_relin_rescale_postop(
    a, b, context, add=None, sub=None, scalar=None, plaintext=None
)
```

At most one post-op may be supplied.

## Rotation, batching, and layouts

```python
fhe.homo_rotate(cipher, offset, context)
fhe.homo_rotate_add(cipher, offset, context, addend=other)
rotated = fhe.fast_rotate(cipher, offsets, context, output_ext=False)

partials = fhe.grouped_pairwise_mac(rotated, plaintexts, groups, context)
weighted = fhe.grouped_scalar_weighted_acc(rotated, scalars, context)
result = fhe.giant_rotate_sum(partials, giant_offset, context, strategy="normal")
result = fhe.hoisted_mac_sum(
    cipher, baby_offsets, plaintexts, giant_offset, giant_count, context,
    strategy="normal",
)
```

Strategy is an explicit string: `normal`, `ext_normal`, or
`ext_double_hoist`. There are no public strategy constants.

Cipher batching and slot metadata APIs are:

```python
batch = fhe.pack_cipher_batch(ciphers)
items = fhe.unpack_cipher_batch(batch)
total = fhe.sum_cipher_batch(batch, context)

expanded = fhe.expand_slots(cipher, target_slots, context)
folded = fhe.fold_slots(expanded, target_slots, context, mask=mask)
```

`expand_slots` changes metadata only. `fold_slots` requires an explicit mask at
the source slot count and performs a plaintext multiplication.

## Bootstrapping

Bootstrap planning belongs to the bootstrap package, not the application:

```python
spec = bs.BootstrapSpec(
    log_slots=14,
    level_budget=(4, 4),
    output_levels=10,
    strategy="normal_giant",
    mode="modraise_first",
)
requirements = bs.requirements(spec, log_n=16)
program = bs.generate(context, spec)
bootstrapped = bs.bootstrap(cipher, context, program)
```

The program owns constants, C2S/S2C schedules, the raise target, runtime mode,
and output state. Applications do not select transform limb layouts, H/L rails,
drop counts, or runtime overrides.

See [FHE API](fhe-api.md) and
[OpenFHE bootstrap API](openfhe-bootstrap-api.md) for the complete contracts.
