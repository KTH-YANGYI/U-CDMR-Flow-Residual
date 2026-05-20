from __future__ import annotations

from pathlib import Path


DOMAIN_TO_INDEX = {"camera": 0, "phone": 1, "dphone": 2}
INDEX_TO_DOMAIN = {value: key for key, value in DOMAIN_TO_INDEX.items()}


def resolve_existing(path_value: str | Path, *, dataset_root: Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute() and path.exists():
        return path
    if path.exists():
        return path
    if dataset_root is not None:
        candidate = dataset_root / path
        if candidate.exists():
            return candidate
    return path

