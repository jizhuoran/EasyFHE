# easyfhe.fhe

`easyfhe.fhe` is the CKKS-oriented public frontend layered on top of the
retained EasyFHE tensor runtime. It owns encrypted tensor state, context
generation, plaintext preparation, homomorphic operators, bootstrapping support,
runtime options, and native key material generation.

## Public Entry Points

For the application-facing API contract, see
[`docs/application-api.md`](../../docs/application-api.md).

Use the package root for stable FHE workflows:

```python
import easyfhe
import easyfhe.fhe as fhe
```

The main entry points are:

- `CKKSContextSpec` and `generate_client_context` for client/context
  construction. Use
  `easyfhe.bs.openfhe` to plan bootstrap extra depth and bootstrap rotation keys
  before constructing the context.
- `Context`, `CipherState`, and `ConstantBundle` for runtime state and reusable
  scalar/vector constant packs. Plaintext materialization should go through
  `ConstantBundle.plaintext(...)`; raw encoding stages are internal helpers.
- ciphertext slot helpers.
- `homo_add`, `homo_sub`, `homo_mul`, `homo_square`, scalar/plaintext variants,
  rotations, and `align_to` for explicit ciphertext state alignment.
- `RuntimeOptions` for runtime control.

The supported external API is the allowlist in `easyfhe.fhe.__all__`, sourced
from `easyfhe.fhe._public_api.PUBLIC_API`. Importable submodules are
implementation details unless their symbols are re-exported by the package root.

Bootstrap APIs live in concrete sibling packages such as `easyfhe.bs.openfhe`.
New OpenFHE-compatible bootstrap code should call `bs.depth(...)` before context
construction and add it to the application's remaining-depth budget when
choosing `CKKSContextSpec.depth`. Applications may still provide their own depth
directly. Call `bs.plan_rot_keys(...)` before key generation and include those
offsets in `CKKSContextSpec.rotations`; key generation then returns separate
client/server material, and `Context` is built from the server material only.
After context construction, call `bs.generate(ctx, ...)` to generate bootstrap
constants and a bootstrap plan, then call
`bs.bootstrap(cipher, ctx, constants, plan, L0=...)` at runtime.

## Package Map

- `runtime/`: context specs, runtime options, rescale policy helpers, and
  operation validation.
- `context.py` and `ciphertext.py`: context construction and ciphertext/plaintext
  state containers.
- `ops/`: homomorphic arithmetic, encoding, key switching, rotation, alignment,
  fused operations, and thin native-kernel wrappers.
- `../bs/`: public bootstrapping specs, planning, constants, runtime helpers,
  and OpenFHE-specific internal implementation code.
- `_keygen/`: native sampler integration and context material assembly.

## Refactor Direction

Keep new public FHE APIs re-exported from `easyfhe.fhe.__init__` and keep
submodules focused on one responsibility. General PyTorch model-training,
optimizer, export, and quantization surfaces should stay outside this package or
remain explicit compatibility stubs elsewhere in `easyfhe`.
