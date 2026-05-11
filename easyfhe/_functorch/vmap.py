def lazy_load_decompositions():
    return None


def restore_vmap(*args, **kwargs):
    raise RuntimeError("functorch vmap is disabled in EasyFHE")


def unwrap_batched(*args, **kwargs):
    raise RuntimeError("functorch vmap is disabled in EasyFHE")


def wrap_batched(*args, **kwargs):
    raise RuntimeError("functorch vmap is disabled in EasyFHE")
