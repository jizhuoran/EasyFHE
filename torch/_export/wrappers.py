def mark_subclass_constructor_exportable_experimental(fn=None, **kwargs):
    if fn is None:
        return lambda inner: inner
    return fn
