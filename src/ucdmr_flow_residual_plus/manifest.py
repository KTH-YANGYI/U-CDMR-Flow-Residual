from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from ucdmr_flow_residual_plus.constants import materialize_domain_fields
from ucdmr_flow_residual_plus.io_utils import read_csv_records


KEEP_LABELS = {"crack", "normal"}
IGNORE_LABELS = {"broken"}


REQUIRED_COLUMNS = [
    "dataset_group",
    "label",
    "dataset_relative_path",
    "annotation_relative_path",
    "image_width",
    "image_height",
    "video_name",
]


def load_merged_manifest(path: str | Path) -> list[dict[str, str]]:
    rows = read_csv_records(path)
    missing = [column for column in REQUIRED_COLUMNS if rows and column not in rows[0]]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    return rows


def filter_valid_rows(rows: list[dict[str, str]], *, max_samples: int | None = None) -> list[dict[str, str]]:
    kept = []
    for row in rows:
        if row.get("label") not in KEEP_LABELS:
            continue
        record = dict(row)
        record["domain"] = record.get("dataset_group", "")
        record = materialize_domain_fields(record)
        kept.append(record)
    if max_samples is not None:
        return kept[:max_samples]
    return kept


def limit_per_domain(rows: list[dict[str, str]], max_per_domain: int | None) -> list[dict[str, str]]:
    if max_per_domain is None:
        return rows
    counts: dict[str, int] = defaultdict(int)
    limited = []
    for row in rows:
        domain = row.get("effective_domain", row.get("domain", row.get("dataset_group", "")))
        if counts[domain] >= max_per_domain:
            continue
        counts[domain] += 1
        limited.append(row)
    return limited


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    labels = Counter(row.get("label", "") for row in rows)
    domains = Counter(row.get("effective_domain", row.get("domain", row.get("dataset_group", ""))) for row in rows)
    by_domain_label = Counter((row.get("effective_domain", row.get("domain", row.get("dataset_group", ""))), row.get("label", "")) for row in rows)
    sizes = Counter(
        (
            row.get("effective_domain", row.get("domain", row.get("dataset_group", ""))),
            row.get("label", ""),
            row.get("image_width", ""),
            row.get("image_height", ""),
        )
        for row in rows
    )
    return {
        "row_count": len(rows),
        "labels": dict(sorted(labels.items())),
        "domains": dict(sorted(domains.items())),
        "by_domain_label": {f"{domain}/{label}": count for (domain, label), count in sorted(by_domain_label.items())},
        "sizes": {f"{domain}/{label}/{width}x{height}": count for (domain, label, width, height), count in sorted(sizes.items())},
        "contains_broken": any(row.get("label") == "broken" for row in rows),
    }
