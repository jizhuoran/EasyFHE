from contextlib import contextmanager

_FX_METADATA_REGISTRY = {}


def has_preserved_node_meta():
    return False


def _is_preserving_node_seq_nr():
    return False


@contextmanager
def preserve_node_meta(*args, **kwargs):
    yield
