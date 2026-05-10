def compatibility(*args, **kwargs):
    def decorator(fn):
        return fn

    return decorator
