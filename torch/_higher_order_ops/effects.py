def new_token_tensor(*args, **kwargs):
    import torch

    return torch.empty((), dtype=torch.uint8)


def _get_effect(*args, **kwargs):
    return None
