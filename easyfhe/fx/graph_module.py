FX_GRAPH_MODULE_FILE_PREFIX = "fx_graph_module"


class GraphModule:
    def __init__(self, root=None, graph=None, class_name="GraphModule"):
        self.root = root
        self.graph = graph
        self.__class__.__name__ = class_name


def _assign_attr(*args, **kwargs):
    return None


class _Loader:
    def __call__(self, *args, **kwargs):
        raise RuntimeError("torch.fx is disabled in EasyFHE")


_loader = _Loader()
