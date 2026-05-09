def _disabled(*args, **kwargs):
    raise RuntimeError("FlexAttention higher-order ops are disabled in EasyFHE")


flex_attention = _disabled
flex_attention_backward = _disabled


__all__ = ["flex_attention", "flex_attention_backward"]
