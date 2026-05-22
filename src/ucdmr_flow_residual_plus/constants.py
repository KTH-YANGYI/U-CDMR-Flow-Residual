from __future__ import annotations

from pathlib import Path
import math
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
_DPHONE_PATTERNS = [
    re.compile(r"dphone[_-]?(\d+)", re.IGNORECASE),
    re.compile(r"dphone[/\\]+(?:id[_-]?)?(\d+)", re.IGNORECASE),
    re.compile(r"dphone[^/\\]*?[_-](?:id[_-]?)?(\d+)(?:$|[/_\\.-])", re.IGNORECASE),
    re.compile(r"(?:^|[/_\\-])id[_-]?(\d+)(?:$|[/_\\.-])", re.IGNORECASE),
    re.compile(r"(?:^|[/_\\-])(\d+)_dphone(?:$|[/_\\.-])", re.IGNORECASE),
]
_DPHONE_ID_FIELDS = ("dphone_id", "device_id", "image_id", "id")
_DPHONE_TEXT_FIELDS = (
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
)


def _int_value(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d+", text):
        return int(text)
    if re.fullmatch(r"\d+\.0+", text):
        return int(float(text))
    return None


def _text_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def _first_text(row: dict[str, object], keys: tuple[str, ...]) -> str:
    for key in keys:
        text = _text_value(row.get(key))
        if text:
            return text
    return ""


def dphone_id_from_text(value: object) -> int | None:
    text = _text_value(value)
    for pattern in _DPHONE_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            return int(match.group(1))
    return None


def dphone_id_from_row(row: dict[str, object]) -> int | None:
    for key in _DPHONE_ID_FIELDS:
        found = _int_value(row.get(key))
        if found is not None:
            return found
    for key in _DPHONE_TEXT_FIELDS:
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


def split_dphone_domain_from_id(dphone_id: int) -> str:
    return DPHONE_LOW_DOMAIN if int(dphone_id) < DPHONE_SPLIT_ID else DPHONE_HIGH_DOMAIN


def mask_domain_from_row(row: dict[str, object]) -> str:
    """Strict domain for mask-flow training/sampling and normal-mask matching."""
    domain = _first_text(row, ("effective_domain", "domain", "dataset_group"))
    if domain in MASK_DOMAIN_TO_INDEX:
        return domain
    if base_domain_name(domain) != "dphone":
        if domain in {"camera", "phone"}:
            return domain
        raise ValueError(f"Unknown mask domain={domain!r}")
    dphone_id = dphone_id_from_row(row)
    if dphone_id is None:
        raise ValueError(f"Cannot split dphone mask domain; missing dphone id. row keys={sorted(row.keys())}")
    return split_dphone_domain_from_id(dphone_id)


def residual_domain_from_row_or_domain(row_or_domain: dict[str, object] | str) -> str:
    """Domain for the residual model checkpoint. Always camera, phone, or dphone."""
    if isinstance(row_or_domain, str):
        return base_domain_name(row_or_domain)
    domain = str(
        _first_text(row_or_domain, ("residual_domain", "effective_domain", "domain", "dataset_group"))
    )
    return base_domain_name(domain)


def materialize_domain_fields(row: dict[str, object], *, set_domain: bool = True) -> dict[str, object]:
    record = dict(row)
    mask_domain = mask_domain_from_row(record)
    dphone_id = dphone_id_from_row(record) if base_domain_name(mask_domain) == "dphone" else None
    base_domain = base_domain_name(mask_domain)
    record["dphone_id"] = "" if dphone_id is None else dphone_id
    record["effective_domain"] = mask_domain
    record["base_domain"] = base_domain
    record["residual_domain"] = base_domain
    if set_domain:
        record["domain"] = mask_domain
    return record


def effective_domain(row: dict[str, object] | str) -> str:
    if isinstance(row, str):
        domain = row
        row_dict: dict[str, object] = {}
    else:
        domain = _first_text(row, ("effective_domain", "domain", "dataset_group"))
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
    base_domain = residual_domain_from_row_or_domain(domain)
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
