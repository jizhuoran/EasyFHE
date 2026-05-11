def _disabled(*args, **kwargs):
    raise RuntimeError("functorch forward AD is disabled in EasyFHE")


_add_batch_dim = _disabled
_remove_batch_dim = _disabled
_vmap_increment_nesting = _disabled
_vmap_decrement_nesting = _disabled
_make_dual = _disabled
_unpack_dual = _disabled
_jvp_increment_nesting = _disabled
_jvp_decrement_nesting = _disabled
_unwrap_for_grad = _disabled
_enter_dual_level = _disabled
_exit_dual_level = _disabled
