# Publishing The EasyFHE Install Page

This directory is a static GitHub Pages site for EasyFHE installation.

## One-time GitHub Pages setup

1. Push `docs/` to the default branch.
2. Open the repository on GitHub.
3. Go to `Settings` -> `Pages`.
4. Set `Source` to `Deploy from a branch`.
5. Set `Branch` to `main` and folder to `/docs`.
6. Save. The project page will be:

```text
https://jizhuoran.github.io/EasyFHE/
```

If the default branch is not `main`, choose the actual default branch.

## Release wheel flow

Build wheels first:

```bash
CUDA_VERSIONS="12.4 13.2" \
  PYTHON_TAG=cp312-cp312 \
  TORCH_CUDA_ARCH_LIST=8.0 \
  packaging/manylinux/build_all_cuda_wheels.sh
```

Expected outputs:

```text
wheelhouse/manylinux_2_28_x86_64/cu124/*.whl
wheelhouse/manylinux_2_28_x86_64/cu132/*.whl
```

Do not commit the wheel files into `docs/`. They are too large for normal Git
hosting. Upload them as GitHub Release assets or to object storage.

For the current first release, create a GitHub Release named:

```text
v2.13.0a0
```

Upload these assets:

```text
easyfhe-2.13.0a0+cu124.git6e869e1-cp312-cp312-manylinux_2_28_x86_64.whl
easyfhe-2.13.0a0+cu132.git6e869e1-cp312-cp312-manylinux_2_28_x86_64.whl
```

Then update:

- `docs/install-matrix.json`
- `docs/whl/cu124/index.html`
- `docs/whl/cu132/index.html`

The install page uses `pip --find-links`, so PyPI remains available for normal
Python dependencies while EasyFHE is selected from the CUDA wheel channel.

## Test the public install command

Use a clean virtual environment:

```bash
python3.12 -m venv /tmp/easyfhe-install-test
source /tmp/easyfhe-install-test/bin/activate
python -m pip install --upgrade pip
python -m pip install "easyfhe==2.13.0a0+cu132.git6e869e1" \
  --find-links https://jizhuoran.github.io/EasyFHE/whl/cu132
python -c "import easyfhe; print(easyfhe.__version__)"
```

Repeat with `cu124` before announcing the release.

## Custom domain later

When the first page works, it is worth moving wheels to a stable download
domain such as:

```text
https://download.easyfhe.org/whl/cu132
https://download.easyfhe.org/whl/cu124
```

At that point, only `docs/install-matrix.json` and the small wheel index pages
need to change.
