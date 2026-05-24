# EasyFHE FHE API

This page documents the public API exported from `easyfhe.fhe`.
Import from the package root:

```python
import easyfhe.fhe as fhe
```

Submodules under `easyfhe.fhe` may be importable, but they are implementation
details unless their symbols are re-exported by `easyfhe.fhe.__all__`.

## Context And Client

### `CKKSContextSpec`

```python
fhe.CKKSContextSpec(
    depth: int,
    log_n: int,
    dnum: int,
    dcrt_bits: int,
    first_mod: int,
    secret_key_dist: str = "SPARSE_TERNARY",
    scale_mode: str = "fixed",
    rescale_policy: str = "manual",
    rotations: tuple[int, ...] = (),
    auto_load_keys: bool | None = None,
    rotation_random_mode: str = "fresh",
    rotation_key_limb_limits: Mapping[int, int] = {},
)
```

Specifies CKKS parameters for native EasyFHE context generation.

- `depth`: number of available Q limbs / multiplicative depth.
- `log_n`: ring dimension log; the ring dimension is `1 << log_n`.
- `dnum`: decomposition count for key switching.
- `dcrt_bits`: bit size for ordinary CRT primes.
- `first_mod`: bit size for the first / special modulus.
- `secret_key_dist`: secret-key distribution string.
- `scale_mode`: `"fixed"` or `"flexible"`.
- `rescale_policy`: `"manual"` or `"auto"`.
- `rotations`: rotation offsets whose keys should be generated.
- `auto_load_keys`: when true, eagerly moves rotation keys to CUDA contexts.
  When `None`, CUDA contexts auto-load keys and CPU contexts do not.
- `rotation_random_mode`: randomness mode used for rotation-key generation.
- `rotation_key_limb_limits`: optional per-rotation key limb limit map.

Returns an immutable dataclass instance used by `generate_client_context(...)`.

### `generate_client_context(spec, device="cpu")`

Builds a paired client and runtime context.

```python
client, ctx = fhe.generate_client_context(spec, device="cuda")
```

Parameters:

- `spec`: a `CKKSContextSpec`.
- `device`: `"cpu"`, `"cuda"`, or another device accepted by the tensor backend.

Returns:

- `Client`: owns secret/public material and encrypt/decrypt methods.
- `Context`: server-side runtime material and evaluation keys.

### `Client`

Client-side encryption and decryption handle.

```python
cipher = client.encrypt(values, device=None, scale_deg=1, level=0, slots=0)
plain = client.decrypt(cipher)
```

`encrypt(...)` parameters:

- `values`: array-like plaintext values.
- `device`: target ciphertext device. Defaults to CPU.
- `scale_deg`: plaintext scale degree. Usually `1`.
- `level`: encode level. Use `0` for a fresh ciphertext.
- `slots`: active CKKS slot count. `0` lets encoding use its default.

`encrypt(...)` returns a `Cipher`.

`decrypt(cipher)` returns a tensor of decoded `float64` values on the
ciphertext device.

Property:

- `client.N`: ring dimension, equal to `1 << client.log_n`.

### `Context`

Runtime server context used by homomorphic operations.

Application-facing methods:

```python
ctx.construct_copy(device)
ctx.cuda()
ctx.cpu()
ctx.scale_at(cur_limbs=None)
ctx.big_scale_at(cur_limbs=None)
ctx.rescale_divisor_at(drop_limb=None)
ctx.get_rotation_key(rot_index)
ctx.get_precompute_auto(key)
ctx.get_inverse_precompute_auto(key)
```

- `construct_copy(device)` returns a context copy on `device`.
- `cuda()` / `cpu()` return a copy of the context on that device.
- `scale_at(cur_limbs)` returns the active CKKS scale for a limb count.
- `big_scale_at(cur_limbs)` returns the corresponding big scale.
- `rescale_divisor_at(drop_limb)` returns the divisor used when rescaling.
- `get_rotation_key(...)`, `get_precompute_auto(...)`, and
  `get_inverse_precompute_auto(...)` are advanced runtime accessors used by
  rotation kernels and bootstrap internals.

Most context attributes are runtime parameters or precomputed tables. Treat
direct access as advanced debugging unless an operation explicitly asks for the
context.

### `CipherState`

```python
fhe.CipherState(cur_limbs: int, noise_deg: int, scaling_factor: float | None = None)
```

Tracks ciphertext scale metadata.

- `cur_limbs`: active Q limbs.
- `noise_deg`: scale/noise degree used by validation and encoding.
- `scaling_factor`: concrete scale factor when available.

Method:

```python
state.replace(cur_limbs=None, noise_deg=None, scaling_factor=None)
```

Returns a new `CipherState` with selected fields changed.

### `Cipher`

Ciphertext/plaintext container used by the runtime.

Public fields:

- `cv`: component tensors. Ciphertexts normally have two components;
  plaintexts have one component.
- `state`: `CipherState`.
- `slots`: active CKKS slot count.
- `is_ext`: whether the ciphertext is in the extended QP domain.
- `batch_size`: number of packed ciphertexts in the leading batch dimension.

Methods:

```python
cipher.cipher_like(cv, state=None, slots=None, is_ext=None, batch_size=None)
cipher.deep_copy()
cipher.shallow_copy()
cipher.replace_with(other)
cipher.cuda()
cipher.cpu()
```

Most application code receives `Cipher` objects from `Client.encrypt(...)`,
`ConstantBundle.plaintext(...)`, or FHE operations instead of constructing them
manually.

## Constants

### `ConstantBundle`

```python
bundle = fhe.ConstantBundle(
    scalars={"scale": 0.125},
    vectors={"weights": values},
    cache_mode="plain",
    plain_cache_limit_gb=None,
    plain_cache_policy="first_fit",
)
```

Stores named scalar and vector constants and materializes reusable plaintexts or
encoded scalar tensors.

`cache_mode` can be:

- `"none"`: cache nothing.
- `"middle"`: cache prepared vector encodings.
- `"plain"`: cache plaintexts and encoded scalars.
- `"both"`: cache plaintexts first; when the plaintext cache limit is reached,
  keep prepared vector encodings as a fallback. A vector that already has a
  cached plaintext does not also keep its middle encoding.

`plain_cache_limit_gb` limits how much plaintext cache `ConstantBundle` may
keep. `None` means no explicit limit. The limit applies only to plaintext
entries; middle encodings and encoded scalar tensors are still reported in
`cache_info()`.

`plain_cache_policy` controls admission when the plaintext cache is full:

- `"first_fit"`: keep existing entries and skip new plaintexts that do not fit.
- `"small_first"`: replace the largest cached plaintext when the new plaintext
  is smaller. In `"both"` mode, evicted plaintexts keep their middle encoding
  cached as the fallback.

Methods:

```python
len(bundle)
bundle.plaintext(name, level, slots, cryptoContext, scale=1.0, is_ext=False, cache=True)
bundle.encoded_scalars(names, cur_limbs, noise_deg, cryptoContext, mode="double", absolute=False, cache=True)
bundle.cache_info()
bundle.memory_info()
bundle.clear_cache()
bundle.set_cache_mode(cache_mode, clear=True)
bundle.set_plain_cache_limit_gb(limit_gb, clear=False)
bundle.set_plain_cache_limit_bytes(limit_bytes, clear=False)
bundle.set_plain_cache_policy(policy)
```

`len(bundle)` returns the number of named vector constants.

`plaintext(...)` parameters:

- `name`: vector constant name.
- `level`: encoding level, usually `ctx.L - cipher.state.cur_limbs`.
- `slots`: plaintext slot count.
- `cryptoContext`: runtime `Context`.
- `scale`: scalar multiplier applied while preparing the vector.
- `is_ext`: encode into the extended QP domain when true.
- `cache`: whether this call may read/write the bundle cache.

Returns a plaintext represented as a one-component `Cipher`.

`encoded_scalars(...)` parameters:

- `names`: one scalar name or an iterable of scalar names.
- `cur_limbs`: active limb count the scalar will be used at.
- `noise_deg`: current ciphertext noise degree.
- `cryptoContext`: runtime `Context`.
- `mode`: `"double"` for approximate CKKS constants or `"int"` for exact
  integer CRT constants.
- `absolute`: encode absolute scalar values when true.
- `cache`: whether this call may read/write the scalar cache.

Returns a tensor with shape `[len(names), cur_limbs]`. Scalar homomorphic
operations expect one encoded row from this tensor, not a raw Python number.

## Alignment

### `align_to(cipher, target, cryptoContext)`

Aligns a ciphertext to a target `CipherState`.

```python
target = fhe.CipherState(cur_limbs=12, noise_deg=1)
cipher = fhe.align_to(cipher, target, ctx)
```

The operation may rescale or reduce scale metadata as needed. It returns a new
`Cipher`.

### `reduce_noise_to_one(cipher, cryptoContext)`

Reduces a ciphertext to `noise_deg == 1` when possible and returns the aligned
`Cipher`.

## Basic Cipher Operations

All operations take a runtime `Context` as the last positional argument. Non
in-place binary operations align compatible inputs automatically. In-place
variants require the input metadata to already match.

### Ciphertext Arithmetic

```python
fhe.homo_add(in0, in1, cryptoContext)
fhe.homo_add_inplace(in0, in1, cryptoContext)
fhe.homo_sub(in0, in1, cryptoContext)
fhe.homo_sub_inplace(in0, in1, cryptoContext)
fhe.homo_mul_relin(in0, in1, cryptoContext)
```

- `in0`, `in1`: `Cipher` objects with matching slot counts.
- Returns: a `Cipher`.
- In-place variants mutate and return `in0`.
- `homo_mul_relin(...)` multiplies two ciphertexts and relinearizes back to two
  components. It does not rescale by itself.

### Multiply, Rescale, Then Optional Post-Op

```python
fhe.homo_mul_relin_rescale_postop(
    in0,
    in1,
    cryptoContext,
    *,
    apply_double=False,
    add=None,
    sub=None,
    scalar=None,
    plaintext=None,
)
```

Multiplies, relinearizes, rescales one level, and optionally fuses one post-op.
At most one of `add`, `sub`, `scalar`, or `plaintext` may be supplied.

Parameters:

- `in0`, `in1`: non-extended single ciphertexts with matching slot counts.
- `apply_double`: whether the fused kernel applies its double path.
- `add`: optional ciphertext to add after rescale.
- `sub`: optional ciphertext to subtract after rescale.
- `scalar`: optional encoded scalar row.
- `plaintext`: optional plaintext to add after rescale.

Returns a rescaled `Cipher`.

Convenience wrappers:

```python
fhe.homo_mul_relin_rescale_add_scalar(in0, in1, scalar, cryptoContext)
fhe.homo_mul_relin_rescale_add_pt(in0, in1, plaintext, cryptoContext)
```

### Plaintext Operations

```python
fhe.homo_add_pt(cipher, plaintext, cryptoContext)
fhe.homo_add_pt_inplace(cipher, plaintext, cryptoContext)
fhe.homo_mul_pt(cipher, plaintext, cryptoContext)
fhe.homo_mul_pt_inplace(cipher, plaintext, cryptoContext)
```

- `cipher`: a non-extended `Cipher`.
- `plaintext`: a plaintext produced by `ConstantBundle.plaintext(...)`.
- Returns: a `Cipher`.
- In-place variants mutate and return `cipher`.

The ciphertext and plaintext must have matching `cur_limbs`, `scaling_factor`,
and `slots`.

### Encoded Scalar Operations

```python
fhe.homo_add_scalar_double(cipher, constant, cryptoContext)
fhe.homo_add_scalar_double_inplace(cipher, constant, cryptoContext)
fhe.homo_add_scalar_int(cipher, scalar, cryptoContext)
fhe.homo_add_scalar_int_inplace(cipher, scalar, cryptoContext)
fhe.homo_sub_scalar_int(cipher, scalar, cryptoContext)
fhe.homo_sub_scalar_int_inplace(cipher, scalar, cryptoContext)
fhe.homo_mul_scalar_double(cipher, constant, cryptoContext)
fhe.homo_mul_scalar_double_inplace(cipher, constant, cryptoContext)
fhe.homo_mul_scalar_int(cipher, scalar, cryptoContext)
fhe.homo_mul_scalar_int_inplace(cipher, scalar, cryptoContext)
```

`constant` and `scalar` must be encoded CRT scalar rows, typically from
`ConstantBundle.encoded_scalars(...)`:

```python
encoded = bundle.encoded_scalars(
    "scale",
    cipher.state.cur_limbs,
    cipher.state.noise_deg,
    ctx,
    mode="double",
)[0]
out = fhe.homo_mul_scalar_double(cipher, encoded, ctx)
```

Returns a `Cipher`; in-place variants mutate and return `cipher`.

## Rotation And Hoisting

### Single Rotation

```python
fhe.homo_rotate(cipher, offset, cryptoContext)
fhe.homo_rotate_add(cipher, offset, cryptoContext, addend=None)
```

- `cipher`: non-extended two-component `Cipher`.
- `offset`: rotation offset. A matching rotation key must exist in the context.
- `addend`: optional ciphertext added during the rotation.

Returns a rotated `Cipher`. `homo_rotate(...)` is equivalent to
`homo_rotate_add(..., addend=None)`.

### Batched Fast Rotation

```python
fhe.fast_rotate(cipher, offsets, cryptoContext, *, output_ext=False)
```

Computes several rotations of one ciphertext using one hoisted modup.

- `offsets`: an int or non-empty iterable of ints. At least one offset must be
  nonzero.
- `output_ext`: when true, returns an extended-domain batched ciphertext.

Returns a batched `Cipher` with `batch_size == len(offsets)`.

### Extended-Domain Moddown

```python
fhe.moddown_from_ext(cipher, cryptoContext)
```

Converts an extended-domain `Cipher` (`is_ext=True`) back to normal Q domain and
returns a new `Cipher`.

### Hoisted MAC And Giant Sum

```python
fhe.grouped_pairwise_mac(ciphers, plaintexts, groups, cryptoContext)
fhe.grouped_scalar_weighted_acc(ciphers, scalars, cryptoContext)
fhe.giant_rotate_sum(ciphers, offset, cryptoContext, *, strategy=fhe.HOIST_NORMAL)
fhe.hoisted_mac_sum(
    cipher,
    baby_offsets,
    plaintexts,
    giant_offset,
    giant_count,
    cryptoContext,
    *,
    strategy,
)
```

`grouped_pairwise_mac(...)`:

- `ciphers`: batched `Cipher`.
- `plaintexts`: batched plaintext whose batch size is
  `groups * ciphers.batch_size`.
- `groups`: number of output groups.
- Returns: batched `Cipher` with `batch_size == groups`.

`grouped_scalar_weighted_acc(...)`:

- `ciphers`: batched `Cipher`.
- `scalars`: encoded scalar tensor. The first dimension is the output group
  count.
- Returns: batched `Cipher` with `batch_size == scalars.shape[0]`.

`giant_rotate_sum(...)`:

- `ciphers`: batched `Cipher` with `batch_size > 1`.
- `offset`: nonzero giant rotation offset.
- `strategy`: one of the hoist strategy constants below.
- Returns: one accumulated `Cipher`.

`hoisted_mac_sum(...)` combines baby rotations, plaintext MAC, and giant
rotation accumulation for BSGS-style linear transforms.

Hoist strategy constants:

```python
fhe.HOIST_NORMAL
fhe.HOIST_EXT_NORMAL
fhe.HOIST_EXT_DOUBLE_HOIST
```

- `HOIST_NORMAL`: fast rotate to normal ciphertexts, then normal giant
  rotations.
- `HOIST_EXT_NORMAL`: fast rotate to extended ciphertexts, MAC in extended
  domain, moddown before giant rotations.
- `HOIST_EXT_DOUBLE_HOIST`: fast rotate to extended ciphertexts, MAC in extended
  domain, and double-hoist the giant rotations.

## Slot And Batch Layout

### Slot Metadata

```python
fhe.expand_slots(cipher, slots, cryptoContext)
fhe.fold_slots(cipher, slots, cryptoContext, *, mask)
```

`expand_slots(...)` increases the ciphertext slot metadata and returns a copied
`Cipher`. It does not change encrypted values. `slots` must be at least the
current slot count.

`fold_slots(...)` reduces the active slot count by multiplying with an explicit
mask plaintext and adding folded rotations. Parameters:

- `cipher`: non-extended `Cipher`.
- `slots`: target slot count, smaller than the current slot count.
- `mask`: plaintext encoded at the source slot count.

Returns a `Cipher` with `slots` set to the target count.

### Cipher Batching

```python
fhe.pack_cipher_batch(ciphers)
fhe.unpack_cipher_batch(cipher)
```

`pack_cipher_batch(...)` takes a non-empty iterable of compatible `Cipher`
objects and concatenates them into one batched `Cipher`.

`unpack_cipher_batch(...)` splits a batched `Cipher` into a tuple of individual
`Cipher` objects. A single unbatched ciphertext is returned as a one-element
tuple.

## Common Usage Pattern

```python
import easyfhe.fhe as fhe

spec = fhe.CKKSContextSpec(
    depth=10,
    log_n=16,
    dnum=3,
    dcrt_bits=52,
    first_mod=55,
    rotations=(1, 2, 4, -1),
    auto_load_keys=True,
)
client, ctx = fhe.generate_client_context(spec, device="cuda")

cipher = client.encrypt([1.0, 2.0, 3.0, 4.0], device="cuda", slots=4)
constants = fhe.ConstantBundle(
    vectors={"mask": [1.0, 1.0, 0.0, 0.0]},
    scalars={"gain": 0.5},
)

gain = constants.encoded_scalars(
    "gain",
    cipher.state.cur_limbs,
    cipher.state.noise_deg,
    ctx,
    mode="double",
)[0]
scaled = fhe.homo_mul_scalar_double(cipher, gain, ctx)
rotated = fhe.homo_rotate(scaled, 1, ctx)
out = fhe.homo_add(scaled, rotated, ctx)

decoded = client.decrypt(out)
```
