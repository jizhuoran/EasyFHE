def active_limbs(cipher, context):
    """Return the physical limb width of a regular or extended ciphertext."""
    return int(cipher.state.cur_limbs) + (int(context.K) if cipher.is_ext else 0)
