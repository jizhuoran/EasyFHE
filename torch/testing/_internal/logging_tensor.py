from contextlib import contextmanager


class LoggingTensorMode:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@contextmanager
def capture_logs(*args, **kwargs):
    logs = []
    yield logs

