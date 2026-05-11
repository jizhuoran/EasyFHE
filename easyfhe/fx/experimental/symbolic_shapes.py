from enum import Enum


class GuardOnDataDependentSymNode(RuntimeError):
    pass


class DimDynamic(Enum):
    STATIC = 0
    DYNAMIC = 1
    DUCK = 2


class ShapeEnv:
    pass


class SymbolicContext:
    pass


class SubclassSymbolicContext(SymbolicContext):
    pass


SymTypes = ()
TrackedFake = object
uninteresting_files = set()


def _as_bool(value, default=False):
    try:
        return bool(value)
    except Exception:
        return default


def expect_true(value, *args, **kwargs):
    return _as_bool(value, True)


def guard_or_false(value):
    return _as_bool(value, False)


def guard_or_true(value):
    return _as_bool(value, True)


def sym_eq(a, b):
    return a == b


def sym_and(*args):
    return all(_as_bool(a, False) for a in args)


def sym_or(*args):
    return any(_as_bool(a, False) for a in args)


def statically_known_true(value):
    return _as_bool(value, False)


def has_free_unbacked_symbols(value):
    return False


def free_unbacked_symbols(value):
    return set()


def constrain_range(*args, **kwargs):
    return None


def _constrain_range_for_size(*args, **kwargs):
    return None


def _advise_is_size(*args, **kwargs):
    return True


def _advise_is_bounded(*args, **kwargs):
    return True


def is_symbolic(value):
    return False


def is_nested_int(value):
    return False


def has_guarding_hint(value):
    return True


def guard_int(value):
    return int(value)


def _iterate_exprs(value):
    return ()


def _iterate_nodes(value):
    return ()
