class TritonKernelWrapperFunctional:
    pass


class TritonKernelWrapperMutation:
    pass


class TraceableTritonKernelWrapper:
    pass


def get_kernel(*args, **kwargs):
    raise RuntimeError("triton higher-order kernels are disabled in EasyFHE")
