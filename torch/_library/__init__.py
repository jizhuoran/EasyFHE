import torch._library.autograd
import torch._library.fake_impl
import torch._library.simple_registry
import torch._library.utils
from torch._library.fake_class_registry import register_fake_class
try:
    from torch._library.triton import capture_triton, triton_op, wrap_triton
except Exception:
    def capture_triton(*args, **kwargs):
        raise RuntimeError("Triton custom ops are disabled in EasyFHE fast build")

    triton_op = capture_triton
    wrap_triton = capture_triton
