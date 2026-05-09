# mypy: allow-untyped-defs
from enum import Enum
from typing import Any, Callable


_DEFAULT_SPARSE_BLOCK_SIZE = 128
FlexKernelOptions = dict[str, Any]


def _disabled(*args, **kwargs):
    raise RuntimeError("torch.nn.attention.flex_attention is disabled in EasyFHE")


class _Backend(Enum):
    EAGER = "eager"
    COMPILED = "compiled"
    FLASH = "flash"


class _MaskModWrapper:
    def __init__(self, fn: Callable, spec=None):
        self.fn = fn
        self.spec = spec

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class BlockMask:
    def __init__(self, *args, **kwargs):
        _disabled()


class AuxRequest:
    def __init__(self, *args, **kwargs):
        _disabled()


class AuxOutput:
    def __init__(self, *args, **kwargs):
        _disabled()


def _mask_mod_signature(*args, **kwargs):
    _disabled()


def _score_mod_signature(*args, **kwargs):
    _disabled()


def _vmap_for_bhqkv(*args, **kwargs):
    _disabled()


def _identity(*args, **kwargs):
    return args[0] if args else None


def _extract_callable_pytree(fn):
    return (), None, fn


def _create_empty_block_mask(*args, **kwargs):
    _disabled()


def _register_blockmask_pytree(*args, **kwargs):
    return None


create_block_mask = _disabled
create_nested_block_mask = _disabled
create_mask = _disabled
flex_attention = _disabled
flex_attention_hop = _disabled
noop_mask = _disabled
or_masks = _disabled
and_masks = _disabled
