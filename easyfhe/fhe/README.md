# easyfhe.fhe

`easyfhe.fhe` is the CKKS-oriented public frontend layered on top of the
retained EasyFHE tensor runtime. It owns encrypted tensor state, context
generation, plaintext preparation, homomorphic operators, bootstrapping support,
and native key material generation.

## Public Entry Points

For the application-facing API contract, see
[`docs/application-api.md`](../../docs/application-api.md).

Use the package root for stable FHE workflows:

```python
import easyfhe
import easyfhe.fhe as fhe
```

The main entry points are:

- `CKKSContextSpec`, `plan_prime_chain`, and `generate_client_context` for client/context
  construction. Use
  `easyfhe.bs.openfhe` to plan bootstrap extra depth and bootstrap rotation keys
  before constructing the context.
- `Context`, `CipherState`, and `ConstantBundle` for runtime state and reusable
  scalar/vector constant packs. Plaintext materialization should go through
  `ConstantBundle.plaintext(...)`; raw encoding stages are internal helpers.
- ciphertext slot helpers.
- `homo_add`, `homo_sub`, `homo_mul_no_relin`, `homo_mul_relin`,
  scalar/plaintext variants,
  rotations, and `align_to` for explicit ciphertext state alignment.

The supported external API is the allowlist in `easyfhe.fhe.__all__`, sourced
from `easyfhe.fhe._public_api.PUBLIC_API`. Importable submodules are
implementation details unless their symbols are re-exported by the package root.

Bootstrap APIs live in concrete sibling packages such as `easyfhe.bs.openfhe`.
Describe each bootstrap with `bs.BootstrapSpec(...)`, then call
`bs.requirements(...)` before context construction. Use its `context_depth` and
`rotations` in `CKKSContextSpec`. After context construction,
`bs.generate(ctx, spec)` returns a context-bound program, which runs as
`bs.bootstrap(cipher, ctx, program)`.

The current frontend uses the u64 prime backend: every Q prime is one physical
limb and every rescale removes exactly one limb. Regular chains can keep using
`depth`/`dcrt_bits`/`first_mod`; use `limb_specs=(first_bits, ..., last_bits)`
when an explicit per-prime chain is useful. Composite or paired limb specs are
not part of the u64 API.

## Package Map

- `context.py` and `ciphertext.py`: context construction and ciphertext/plaintext
  state containers.
- `context_factory.py`: CKKS context spec and client/context generation.
- `ops/`: homomorphic arithmetic, encoding, key switching, rotation, alignment,
  operation validation, fused operations, and thin native-kernel wrappers.
- `../bs/`: public bootstrapping specs, planning, constants, runtime helpers,
  and OpenFHE-specific internal implementation code.
- `_keygen/`: native sampler integration and context material assembly.

## Refactor Direction

Keep new public FHE APIs re-exported from `easyfhe.fhe.__init__` and keep
submodules focused on one responsibility. General PyTorch model-training,
optimizer, export, and quantization surfaces should stay outside this package or
remain explicit compatibility stubs elsewhere in `easyfhe`.
