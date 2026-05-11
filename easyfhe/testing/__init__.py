try:
    from easyfhe._C import FileCheck as FileCheck
except ImportError:

    class FileCheck:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("FileCheck is disabled in EasyFHE fast build")


from ._comparison import assert_allclose, assert_close, default_tolerances
from ._creation import make_tensor


__all__ = [
    "FileCheck",
    "assert_allclose",
    "assert_close",
    "default_tolerances",
    "make_tensor",
]

