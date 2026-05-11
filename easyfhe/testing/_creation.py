import easyfhe as torch


def make_tensor(
    *shape,
    dtype,
    device,
    low=None,
    high=None,
    requires_grad=False,
    noncontiguous=False,
    exclude_zero=False,
    memory_format=None,
):
    if len(shape) == 1 and isinstance(shape[0], (torch.Size, list, tuple)):
        shape = tuple(shape[0])

    if dtype is torch.bool:
        result = torch.randint(0, 2, shape, device=device, dtype=dtype)
    elif dtype.is_floating_point:
        low = -9 if low is None else low
        high = 9 if high is None else high
        result = torch.empty(*shape, dtype=dtype, device=device).uniform_(low, high)
    elif dtype.is_complex:
        low = -9 if low is None else low
        high = 9 if high is None else high
        real = torch.empty(*shape, dtype=torch.float32, device=device).uniform_(low, high)
        imag = torch.empty(*shape, dtype=torch.float32, device=device).uniform_(low, high)
        result = torch.complex(real, imag).to(dtype)
    else:
        low = 0 if low is None else int(low)
        high = 10 if high is None else int(high)
        result = torch.randint(low, high, shape, device=device, dtype=dtype)

    if exclude_zero:
        result = torch.where(result == 0, torch.ones((), dtype=dtype, device=device), result)
    if noncontiguous and result.ndim:
        result = result.transpose(0, -1)
    if memory_format is not None:
        result = result.contiguous(memory_format=memory_format)
    return result.requires_grad_(requires_grad)

