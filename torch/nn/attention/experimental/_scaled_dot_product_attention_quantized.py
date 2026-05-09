# mypy: allow-untyped-defs


def __getattr__(name):
    raise AttributeError(
        "torch.nn.attention.experimental._scaled_dot_product_attention_quantized."
        f"{name} is disabled in EasyFHE"
    )
