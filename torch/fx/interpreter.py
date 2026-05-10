class Interpreter:
    def __init__(self, module):
        self.module = module

    def run(self, *args, **kwargs):
        raise RuntimeError("torch.fx is disabled in EasyFHE")
