from __future__ import annotations

import sys
from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parent
EXAMPLES_ROOT = BENCHMARK_ROOT.parent
PROJECT_ROOT = EXAMPLES_ROOT.parent
DATA_ROOT = BENCHMARK_ROOT / "data"


def ensure_repo_on_path() -> None:
    for path in (PROJECT_ROOT, EXAMPLES_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path
