def split_rescale_tech(rescale_tech):
    if rescale_tech == "FIXEDMANUAL":
        return "fixed", "manual"
    if rescale_tech == "FIXEDAUTO":
        return "fixed", "auto"
    if rescale_tech == "FLEXIBLEAUTO":
        return "flexible", "auto"
    raise ValueError(f"Unsupported rescale technique: {rescale_tech}")
