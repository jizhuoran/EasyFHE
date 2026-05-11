"""Tiny EasyFHE stub for the disabled inductor backend."""


def compile(*args, **kwargs):
    raise RuntimeError("torch.compile is disabled in EasyFHE")


def list_mode_options(*args, **kwargs):
    return {}


def list_options(*args, **kwargs):
    return []

