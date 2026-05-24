def format_accuracy(correct, total):
    percent = 100.0 * correct / total if total else 0.0
    return f"{correct}/{total} ({percent:.2f}%)"


def format_bytes(num_bytes):
    units = ("B", "KiB", "MiB", "GiB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024.0


def format_seconds(seconds):
    return f"{seconds:.3f}s"
