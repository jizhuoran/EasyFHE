def cond(*args, **kwargs):
    raise RuntimeError("cond is disabled in EasyFHE")


class CondOp:
    pass


cond_op = CondOp()
