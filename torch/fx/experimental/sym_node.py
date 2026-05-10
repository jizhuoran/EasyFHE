class SymNode:
    def __init__(self, node):
        self.node = node


class DynamicInt(int):
    pass


def wrap_node(node):
    return node


def to_node(reference, value):
    return getattr(value, "node", value)
