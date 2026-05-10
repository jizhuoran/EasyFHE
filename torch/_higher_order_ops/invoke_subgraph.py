class NestedCompileRegionOptions:
    pass


class InvokeSubgraphHOP:
    pass


def invoke_subgraph(*args, **kwargs):
    raise RuntimeError("invoke_subgraph is disabled in EasyFHE")


def mark_compile_region(fn=None, **kwargs):
    if fn is None:
        return lambda inner: inner
    return fn
