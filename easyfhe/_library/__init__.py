import easyfhe as torch
import easyfhe._library.utils
import easyfhe as torch
import easyfhe._library.simple_registry
import easyfhe as torch
import easyfhe._library.autograd
import easyfhe as torch
import easyfhe._library.fake_impl
from easyfhe._library.fake_class_registry import register_fake_class
try:
    from easyfhe._library.triton import capture_triton, triton_op, wrap_triton
except Exception:
    def capture_triton(*args, **kwargs):
        raise RuntimeError("Triton custom ops are disabled in EasyFHE fast build")

    triton_op = capture_triton
    wrap_triton = capture_triton
