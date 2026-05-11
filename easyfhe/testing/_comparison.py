import easyfhe as torch


def default_tolerances(
    *inputs,
    dtype=None,
    rtol=None,
    atol=None,
):
    if rtol is not None or atol is not None:
        return 0.0 if rtol is None else rtol, 0.0 if atol is None else atol

    dtype = dtype or next(
        (getattr(input, "dtype", None) for input in inputs if getattr(input, "dtype", None) is not None),
        torch.float32,
    )
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-3, 1e-5
    if dtype in (torch.float64, torch.complex128):
        return 1e-7, 1e-7
    return 1e-5, 1e-8


def assert_close(actual, expected, *, rtol=None, atol=None, equal_nan=False, msg=None, **kwargs):
    if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
        rtol, atol = default_tolerances(actual, expected, rtol=rtol, atol=atol)
        if not torch.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan):
            raise AssertionError(msg or "Tensor-likes are not close")
        return

    if actual != expected:
        raise AssertionError(msg or f"{actual!r} != {expected!r}")


assert_allclose = assert_close

