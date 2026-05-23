# EasyFHE manylinux wheel build

This directory contains the reproducible Linux wheel build path for EasyFHE.
It builds `easyfhe` wheels inside a `manylinux_2_28_x86_64` container with a
selected CUDA toolkit installed from NVIDIA's RHEL8 repository.

## Build

Requires Docker or Podman.

```bash
packaging/manylinux/build_wheel.sh
```

Common overrides:

```bash
MANYLINUX_PLAT=manylinux_2_28_x86_64 CUDA_VERSION=12.9 \
  PYTHON_TAG=cp312-cp312 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=24 \
  packaging/manylinux/build_wheel.sh
```

Ubuntu 24.04 / CUDA 13.2 target:

```bash
MANYLINUX_PLAT=manylinux_2_28_x86_64 CUDA_VERSION=13.2 CUDA_FLAVOR=cu132 \
  PYTHON_TAG=cp312-cp312 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=24 \
  packaging/manylinux/build_wheel.sh
```

Build a CUDA matrix:

```bash
CUDA_VERSIONS="12.4 12.6 12.8 12.9 13.2" \
  packaging/manylinux/build_all_cuda_wheels.sh
```

With Podman:

```bash
CONTAINER_ENGINE=podman packaging/manylinux/build_wheel.sh
```

The repaired wheel is written to a CUDA-specific wheelhouse:

```text
wheelhouse/manylinux_2_28_x86_64/cu129/
```

The unrepaired build output remains in:

```text
dist/
```

## Current target

- Platform tag: `manylinux_2_28_x86_64`
- CUDA toolkit: selected by `CUDA_VERSION`
- Default Python: CPython 3.12
- Default GPU arch list: `8.0`

## Ubuntu compatibility

Do not build separate wheels for Ubuntu 20.04, 22.04, and 24.04 unless a
specific system library forces it. Build against the oldest compatible
manylinux baseline instead:

| Target Ubuntu versions | Recommended wheel baseline |
| --- | --- |
| Ubuntu 20.04, 22.04, 24.04 | `manylinux_2_28_x86_64` |
| Ubuntu 18.04 and newer | `manylinux2014_x86_64` if the CUDA/toolchain stack supports it |

Example for the default modern Ubuntu line:

```bash
MANYLINUX_PLAT=manylinux_2_28_x86_64 CUDA_VERSIONS="12.4 12.6 12.8 12.9 13.2" \
  packaging/manylinux/build_all_cuda_wheels.sh
```

If Ubuntu 18.04 support becomes mandatory, try:

```bash
MANYLINUX_PLAT=manylinux2014_x86_64 CUDA_VERSION=12.4 \
  packaging/manylinux/build_wheel.sh
```

Treat `manylinux2014` as a separate compatibility target: it may require older
CUDA/toolchain choices and more auditwheel exclusions.

## CUDA variants and package versions

Each CUDA build gets a local version label by default, for example:

```text
easyfhe-0.1.1+cu129.git6e869e1-cp312-cp312-manylinux_2_28_x86_64.whl
```

This prevents `cu124`, `cu128`, `cu129`, and `cu132` wheels from colliding
when they have the same Python and platform tags.

PyPI does not accept every PyTorch-style local-version workflow cleanly. For
CUDA variants, the practical choices are:

1. Use separate wheel indexes, PyTorch-style.
2. Publish CPU-only on PyPI and CUDA wheels on your own index.
3. Use distinct package names per CUDA runtime if you really need a single
   public PyPI index.

## CUDA policy

The script excludes CUDA driver/runtime libraries from `auditwheel repair` by
default. This keeps the wheel small and avoids bundling driver-facing libraries,
but it means the target machine must provide a compatible CUDA runtime.

Before publishing broadly, choose one policy:

1. Document a system CUDA requirement for CUDA wheels.
2. Depend on NVIDIA's PyPI CUDA runtime packages.
3. Build separate CPU and CUDA wheel channels.

For public PyPI, option 2 or separate wheel indexes are usually cleaner.
