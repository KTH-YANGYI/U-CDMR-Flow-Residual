from __future__ import annotations

from pathlib import Path
import re


DOMAIN_TO_INDEX = {"camera": 0, "phone": 1, "dphone": 2}
INDEX_TO_DOMAIN = {value: key for key, value in DOMAIN_TO_INDEX.items()}

DPHONE_SPLIT_ID = 270
DPHONE_LOW_DOMAIN = "dphone_lt270"
DPHONE_HIGH_DOMAIN = "dphone_ge270"
MASK_DOMAIN_TO_INDEX = {
    "camera": 0,
    "phone": 1,
    DPHONE_LOW_DOMAIN: 2,
    DPHONE_HIGH_DOMAIN: 3,
}
INDEX_TO_MASK_DOMAIN = {value: key for key, value in MASK_DOMAIN_TO_INDEX.items()}
_DPHONE_ID_RE = re.compile(r"dphone[_-]?(\d+)", re.IGNORECASE)


def dphone_id_from_text(value: object) -> int | None:
    match = _DPHONE_ID_RE.search(str(value or ""))
    if match is None:
        return None
    return int(match.group(1))


def dphone_id_from_row(row: dict[str, object]) -> int | None:
    for key in (
        "dataset_relative_path",
        "annotation_relative_path",
        "sample_id",
        "source_template_id",
        "normal_source_path",
        "source_normal_path",
        "m_raw_path",
        "m_gate_path",
        "image_path",
        "synthetic_image_path",
    ):
        value = row.get(key)
        if value:
            found = dphone_id_from_text(value)
            if found is not None:
                return found
    return None


def base_domain_name(domain: str) -> str:
    if domain in {DPHONE_LOW_DOMAIN, DPHONE_HIGH_DOMAIN}:
        return "dphone"
    return domain


def effective_domain(row: dict[str, object] | str) -> str:
    if isinstance(row, str):
        domain = row
        row_dict: dict[str, object] = {}
    else:
        domain = str(row.get("domain") or row.get("dataset_group") or "")
        row_dict = row
    if domain in MASK_DOMAIN_TO_INDEX:
        return domain
    if base_domain_name(domain) != "dphone":
        return domain
    dphone_id = dphone_id_from_row(row_dict) if row_dict else dphone_id_from_text(domain)
    if dphone_id is None:
        return domain
    if dphone_id < DPHONE_SPLIT_ID:
        return DPHONE_LOW_DOMAIN
    return DPHONE_HIGH_DOMAIN


def residual_domain_index(domain: str) -> int:
    base_domain = base_domain_name(domain)
    if base_domain not in DOMAIN_TO_INDEX:
        raise ValueError(f"Unknown residual domain={domain!r} (base={base_domain!r})")
    return DOMAIN_TO_INDEX[base_domain]


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
