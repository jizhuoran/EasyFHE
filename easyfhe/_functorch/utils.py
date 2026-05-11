def unwrap_dead_wrappers(args):
    return args


def exposed_in(_):
    def decorator(fn):
        return fn

    return decorator
