try:
    from torch._C import FileCheck as FileCheck
except ImportError:
    class FileCheck:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("FileCheck is disabled in EasyFHE fast build")

from . import _utils

# pyrefly: ignore [deprecated]
from ._comparison import assert_allclose, assert_close as assert_close
from ._creation import make_tensor as make_tensor
