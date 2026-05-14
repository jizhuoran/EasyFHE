# easyfhe.fhe

`easyfhe.fhe` is the CKKS-oriented public frontend layered on top of the
retained EasyFHE tensor runtime. It owns encrypted tensor state, context
generation, plaintext preparation, homomorphic operators, bootstrapping support,
runtime options, and native key material generation.

## Public Entry Points

Use the package root for stable FHE workflows:

```python
import easyfhe
import easyfhe.fhe as fhe
```

The main entry points are:

- `CKKSContextSpec`, `BootstrapSpec`, `bootstrap_depth`, and `generate_context`
  for context construction.
- `Context`, `CipherState`, and `PreparedPlaintext` for runtime state.
- `prepare_plaintext`, `make_plaintext`, `encode`, and ciphertext slot helpers.
- `homo_add`, `homo_sub`, `homo_mul`, `homo_square`, scalar/plaintext variants,
  rotations, rescale, and noise-level alignment helpers.
- `generate_bootstrap_constants` and `homo_bootstrap` for bootstrapping.
- `RuntimeOptions`, `profile`, and CLI helpers for runtime control.

## Package Map

- `runtime/`: user-facing context specs, runtime options, validation, CLI, and
  instrumentation.
- `context.py` and `ciphertext.py`: context construction and ciphertext/plaintext
  state containers.
- `ops/`: homomorphic arithmetic, encoding, key switching, rotation, alignment,
  fused operations, and thin native-kernel wrappers.
- `bootstrap/`: bootstrapping plans, constants, rotations, approximation, and
  runtime execution.
- `material/`: native sampler integration, CKKS key material, context material,
  rotation plans, and sample arithmetic.
- `dev_tools/` and `logger/`: debugging and instrumentation helpers used by
  development workflows.

## Refactor Direction

Keep new public FHE APIs re-exported from `easyfhe.fhe.__init__` and keep
submodules focused on one responsibility. General PyTorch model-training,
optimizer, export, and quantization surfaces should stay outside this package or
remain explicit compatibility stubs elsewhere in `easyfhe`.
