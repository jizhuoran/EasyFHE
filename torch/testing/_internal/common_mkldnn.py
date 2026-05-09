# mypy: ignore-errors
import functools


def bf32_is_not_fp32():
    return False


def tf32_is_not_fp32():
    return False


def reduced_f32_on_and_off(bf32_precision=1e-2, tf32_precision=1e-5):
    def wrapper(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            return f(*args, **kwargs)

        return wrapped

    return wrapper
