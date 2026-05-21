# EasyFHE

EasyFHE is a tensor runtime for Fully Homomorphic Encryption (FHE). See the [homepage and installation guide](https://jizhuoran.github.io/EasyFHE/) for CUDA wheel selection and installation commands. EasyFHE keeps the parts of a PyTorch-derived runtime that encrypted tensor programs need: dispatch, storage, CUDA execution, Python bindings, profiling hooks, and selected NCCL-oriented infrastructure. Around that runtime, EasyFHE adds a CKKS-oriented FHE layer with native key material generation and GPU accelerated homomorphic operators.

```python
import easyfhe
import easyfhe.fhe as fhe
```

EasyFHE is not intended to be a full PyTorch replacement. The goal is narrower: make FHE programs easier to express while keeping the implementation close enough to the encrypted data layout, RNS levels, key material, and GPU kernels to optimize end-to-end execution.

## Design

The design philosophy follows the framework described in [A Framework for Developing and Optimizing Fully Homomorphic Encryption Programs on GPUs](https://doi.org/10.1145/3779212.3790120). EasyFHE treats FHE programs as tensor programs, but exposes enough runtime structure for GPU-oriented optimization: ciphertext layout, modulus levels, precomputed material, rotation keys, bootstrapping constants, and fused homomorphic kernels are all first-class parts of the system.

## Highlights

- `easyfhe.fhe` frontend for CKKS context specs, ciphertext state, plaintext preparation, arithmetic, rotation, rescale, fused ops, and bootstrapping.
- Native sampler and material generation for CKKS public keys, multiplication keys, rotation keys, and encryption/decryption material.
- CUDA accelerated FHE kernels for encoding, NTT, automorphism, key switching, modulus movement, ciphertext arithmetic, and bootstrapping paths.
- A PyTorch-derived tensor core with storage, dispatch, CUDA runtime, Python bindings, selected profiling, and selected NCCL-oriented infrastructure.

## Install

EasyFHE is intended to install with one pip command. Use the [EasyFHE install page](https://jizhuoran.github.io/EasyFHE/) to choose the CUDA wheel channel for your machine.

## Minimal Example

This computes `x^32 + 3.14*x^16 + 1` on encrypted CKKS slots and performs one bootstrap after `x^16`.

```python
import numpy as np
import easyfhe
import easyfhe.fhe as fhe


device = "cuda"
slots = 1 << 12
max_levels_after_bootstrap = 6
input_limbs = 6
bootstrap = fhe.BootstrapSpec(log_bs_slots=12, level_budget=(3, 3))

client, ctx = fhe.generate_client_context(
    fhe.CKKSContextSpec(
        depth=fhe.bootstrap_depth(max_levels_after_bootstrap, [bootstrap]),
        log_n=16,
        dnum=3,
        dcrt_bits=58,
        first_mod=60,
        secret_key_dist="SPARSE_TERNARY",
        scale_mode="fixed",
        rescale_policy="manual",
    ),
    device=device,
)

bootstrap_constants = fhe.generate_bootstrap_constants(
    ctx,
    log_bs_slots=bootstrap.log_bs_slots,
    level_budget=bootstrap.level_budget,
    maxLevelsRemaining=max_levels_after_bootstrap,
)


def square(cipher):
    return fhe.rescale_one_level(fhe.homo_square(cipher, ctx), ctx)


values = np.full(slots, 0.2, dtype=np.float64)
x = ctx.encrypt(
    values,
    device=device,
    scale_deg=1,
    level=ctx.L - input_limbs,
    slots=slots,
)

x2 = square(x)
x4 = square(x2)
x8 = square(x4)
x16 = square(x8)

x16 = fhe.homo_bootstrap(
    x16,
    ctx,
    bootstrap_constants,
    L0=max_levels_after_bootstrap,
)

x32 = square(x16)
term = fhe.rescale_one_level(fhe.homo_mul_scalar_double(x16, 3.14, ctx), ctx)
result = fhe.homo_add_scalar_double(fhe.homo_add(x32, term, ctx), 1.0, ctx)

print(ctx.decrypt(result)[:8])
```

## Reference Application

The ResNet-20 AESPA reference application is maintained separately:

- [jizhuoran/easyfhe-resnet20-aespa](https://github.com/jizhuoran/easyfhe-resnet20-aespa)

## Repository Map

- `easyfhe/`: Python package and retained tensor runtime surface.
- `easyfhe/fhe/`: FHE frontend, context generation, ops, bootstrapping, runtime options, and material handling.
- `aten/src/ATen/native/fhe/`: native FHE kernels and native sampler.
- `packaging/`: manylinux wheel and container packaging scripts.

## Current Status

EasyFHE is an alpha research system. APIs, build flags, ciphertext layout, and packaging details may change quickly. Do not treat it as a production cryptographic library without independent review.

## Direction

EasyFHE is being reduced from a general PyTorch fork into an FHE-focused tensor runtime. The long-term direction is to keep tensor storage, dispatch, CUDA execution, selected profiling, selected NCCL runtime pieces, and FHE operators, while removing model-training, optimizer, export, mobile, quantization, and other general machine-learning surfaces.

## Contributors

EasyFHE is primarily developed by contributors from Shandong University.

## License

EasyFHE is distributed under the GPL-3.0 license.

EasyFHE is derived from PyTorch and keeps portions of the PyTorch runtime. See the repository license files and retained third-party notices for upstream license information.
