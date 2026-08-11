# EasyFHE FHE API

The supported application surface is exported from one package root:

```python
import easyfhe.fhe as fhe
```

Importable submodules are implementation details unless their symbols are
re-exported by `easyfhe.fhe.__all__`.

## u64 model

This frontend represents each Q prime as one physical u64 limb. Consequently:

- `CipherState.cur_limbs` is the number of active Q primes;
- `CipherState.scale_degree` records CKKS scale degree;
- every `rescale(...)` drops one Q prime and one scale degree;
- no H/L rail, backend-width, composite-limb, or public drop-count setting is
  needed.

## Context and client

### `CKKSContextSpec`

```python
fhe.CKKSContextSpec(
    log_n: int,
    dnum: int,
    depth: int | None = None,
    dcrt_bits: int | None = None,
    first_mod: int | None = None,
    secret_key_dist: str = "SPARSE_TERNARY",
    scale_mode: str = "fixed",
    rescale_policy: str = "manual",
    rotations: tuple[int, ...] = (),
    auto_load_keys: bool | None = None,
    rotation_random_mode: str = "fresh",
    rotation_key_limb_limits: Mapping[int, int] = {},
    exact_q_primes: Sequence[int] | None = None,
    limb_specs: Sequence[int] | None = None,
)
```

The regular form uses `depth`, `dcrt_bits`, and `first_mod`. The explicit form
uses one integer bit-size per Q prime:

```python
regular = fhe.CKKSContextSpec(
    depth=3, log_n=16, dnum=1, dcrt_bits=59, first_mod=60,
)
explicit = fhe.CKKSContextSpec(
    log_n=16, dnum=1, limb_specs=(60, 59, 59, 59),
)
```

Do not combine `limb_specs` with `dcrt_bits` or `first_mod`. Fixed scale mode
requires equal bit sizes after the first prime. `scale_mode` is `fixed` or
`flexible`; `rescale_policy` is `manual` or `auto`.

`fhe.plan_prime_chain(...)` validates an explicit chain without generating
keys.

### `generate_client_context`

```python
client, context = fhe.generate_client_context(spec, device="cuda")
```

The returned `Client` owns encryption/decryption material. `Context` contains
server-side parameters and evaluation keys.

```python
cipher = client.encrypt(
    values,
    slots=...,
    device=None,
    cur_limbs=None,
    scaling_factor=None,
)
values = client.decrypt(cipher, complex_output=False)
```

Fresh plaintexts and ciphertexts have `scale_degree=1`. `cur_limbs` defaults to
the full context chain.

Useful context properties and methods are `max_limbs`, `ring_dim`, `max_slots`,
`q_prime_bits`, `default_scale`, `scale_at(...)`, `big_scale_at(...)`,
`rescale_divisor_at(...)`, `cuda()`, and `cpu()`.

## State and constants

### `CipherState`

```python
state = fhe.CipherState(
    cur_limbs=12,
    scale_degree=1,
    scaling_factor=context.scale_at(12),
)
updated = state.replace(cur_limbs=10)
```

`cur_limbs` and `scale_degree` must be positive. `scaling_factor` may be omitted
for fixed-mode alignment targets, but encoded plaintext and scalar values carry
an explicit positive scale.

### `EncodedScalar`

`EncodedScalar` couples CRT residues to `cur_limbs`, `scale_degree`, and
`scaling_factor`. Scalar homomorphic operations require this type; raw tensors
and Python numbers are rejected.

### `ConstantBundle`

```python
bundle = fhe.ConstantBundle(
    vectors={"weight": fhe.PackedRaw(weight_tensor)},
    scalars={"gain": 0.5, "count": 4},
    cache_mode="plain",
)

plaintext = bundle.plaintext(
    "weight",
    state=cipher.state,
    slots=cipher.slots,
    context=context,
    is_ext=False,
    cache=True,
)
```

`PackedRaw` is already arranged at the requested slot count; EasyFHE does not
silently pad or repack it.

Scaled and integer scalar encoding are explicit:

```python
gain = bundle.encoded_scalars(
    "gain",
    cur_limbs=cipher.state.cur_limbs,
    scale_degree=1,
    scaling_factor=cipher.state.scaling_factor,
    context=context,
    mode="scaled",
)[0]

count = fhe.encode_scalar(
    4,
    cur_limbs=cipher.state.cur_limbs,
    scale_degree=0,
    context=context,
    mode="integer",
)
```

`mode="integer"` requires `scale_degree=0` and uses scale 1. A scaled scalar
uses the supplied actual scale once; `scale_degree` is metadata, not an
exponent applied by the encoder.

Cache modes are `none`, `middle`, `plain`, `both`, and
`mix_of_middle_plain`. See `cache_info()`, `memory_info()`, `clear_cache()`,
`set_cache_mode(...)`, `set_plain_cache_limit_gb(...)`, and
`set_plain_cache_policy(...)`.

## Alignment and rescaling

```python
cipher = fhe.align_to(cipher, target_state, context)
cipher = fhe.normalize_scale(cipher, context)
cipher = fhe.rescale(cipher, context)
```

`rescale` requires a non-extended ciphertext with at least two limbs and
`scale_degree > 1`. It consumes one limb. `normalize_scale` rescale-normalizes a
pending multiplication result to `scale_degree=1`. `align_to` may additionally
drop inactive high limbs to reach a requested state.

Flexible mode requires explicit target scales and does not implicitly align
additions or multiplications. Fixed mode may align compatible non-in-place
operands. In-place operations always require matching metadata.

## Arithmetic

### Ciphertext operations

```python
fhe.homo_add(a, b, context)
fhe.homo_add_inplace(a, b, context)
fhe.homo_sub(a, b, context)
fhe.homo_sub_inplace(a, b, context)
fhe.homo_mul_no_relin(a, b, context)
fhe.homo_mul_relin(a, b, context)
```

`homo_mul_no_relin` returns a three-component product. `homo_mul_relin`
relinearizes to two components. Neither operation rescales.

### Plaintext operations

```python
fhe.homo_add_pt(cipher, plaintext, context)
fhe.homo_add_pt_inplace(cipher, plaintext, context)
fhe.homo_mul_pt(cipher, plaintext, context)
fhe.homo_mul_pt_inplace(cipher, plaintext, context)
fhe.homo_mul_pt_rescale(cipher, plaintext, context)
```

`homo_mul_pt_rescale` is the u64 multiply-plus-one-rescale composition.

### Scalar operations

```python
fhe.homo_add_scalar(cipher, scalar, context)
fhe.homo_add_scalar_inplace(cipher, scalar, context)
fhe.homo_sub_scalar(cipher, scalar, context)
fhe.homo_sub_scalar_inplace(cipher, scalar, context)
fhe.homo_mul_scalar(cipher, scalar, context)
fhe.homo_mul_scalar_inplace(cipher, scalar, context)
fhe.homo_mul_scalar_rescale(cipher, scaled_scalar, context)
```

Add/subtract require matching scalar and ciphertext metadata. Multiplication
accepts either an integer (`scale_degree=0`) or normalized scaled
(`scale_degree=1`) `EncodedScalar`. The rescale composition requires the latter.

### Cipher multiply/relin/rescale/post-op

```python
fhe.homo_mul_relin_rescale_postop(
    a,
    b,
    context,
    *,
    apply_double=False,
    add=None,
    sub=None,
    scalar=None,
    plaintext=None,
)
```

This multiplies, relinearizes, consumes one rescale limb, and applies at most
one post-op. Convenience wrappers are
`homo_mul_relin_rescale_add_scalar(...)` and
`homo_mul_relin_rescale_add_pt(...)`.

## Rotation and hoisted operations

```python
fhe.homo_rotate(cipher, offset, context)
fhe.homo_rotate_add(cipher, offset, context, addend=None)
rotated = fhe.fast_rotate(cipher, offsets, context, output_ext=False)
```

Hoisted linear algebra APIs are:

```python
partials = fhe.grouped_pairwise_mac(ciphers, plaintexts, groups, context)
partials = fhe.grouped_pairwise_mac_rescale(
    ciphers, plaintexts, groups, context
)
weighted = fhe.grouped_scalar_weighted_acc(ciphers, scalars, context)
result = fhe.giant_rotate_sum(ciphers, offset, context, strategy="normal")
result = fhe.hoisted_mac_sum(
    cipher, baby_offsets, plaintexts, giant_offset, giant_count, context,
    strategy="normal",
)
result = fhe.hoisted_mac_sum_rescale(
    cipher, baby_offsets, plaintexts, giant_offset, giant_count, context,
    strategy="normal",
)
```

Allowed strategy strings are `normal`, `ext_normal`, and
`ext_double_hoist`. Extended-domain results can be converted with
`moddown_from_ext(...)`.

## Slot and batch layout

```python
expanded = fhe.expand_slots(cipher, slots, context)
folded = fhe.fold_slots(expanded, target_slots, context, mask=mask)

batch = fhe.pack_cipher_batch(ciphers)
items = fhe.unpack_cipher_batch(batch)
total = fhe.sum_cipher_batch(batch, context)
```

`expand_slots` changes metadata only. `fold_slots` multiplies by an explicit
source-slot mask and folds rotations. Batching requires compatible state,
slots, component count, and domain metadata.

## Migration from the older API

| Older form | Current form |
|---|---|
| `noise_deg` | `scale_degree` |
| `reduce_noise_to_one` / `reduce_level_to_one` | `normalize_scale` |
| `rescale_one_level` | `rescale` |
| raw scalar residue tensor | `EncodedScalar` |
| `homo_*_scalar_int/double` | typed `homo_*_scalar` |
| positional constant level/scale arguments | `plaintext(..., state=..., slots=..., context=...)` |
| `mode="int"` / `mode="double"` | `mode="integer"` / `mode="scaled"` |
| `HOIST_*` constants | strategy strings |
| public stage1/stage2 encoding | `PackedRaw` plus `ConstantBundle` |
| separate bootstrap constants and plan | context-bound `BootstrapProgram` |

See [OpenFHE bootstrap API](openfhe-bootstrap-api.md) for bootstrap planning and
execution.
