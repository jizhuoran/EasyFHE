"""Higher-order op stubs for EasyFHE."""


def _disabled(*args, **kwargs):
    raise RuntimeError("higher-order ops are disabled in EasyFHE")


cond = _disabled
while_loop = _disabled
while_loop_stack_output = _disabled
scan = _disabled
map = _disabled
local_map_hop = _disabled
associative_scan = _disabled
invoke_quant_packed = _disabled


class InvokeQuant:
    pass
