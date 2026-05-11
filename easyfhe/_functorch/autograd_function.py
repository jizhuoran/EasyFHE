class AutogradFunctionApply:
    pass


def custom_function_call(*args, **kwargs):
    raise RuntimeError("functorch is disabled in EasyFHE")


def custom_function_call_vmap_helper(*args, **kwargs):
    raise RuntimeError("functorch is disabled in EasyFHE")
