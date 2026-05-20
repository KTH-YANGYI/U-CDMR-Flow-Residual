from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from ucdmr_flow_residual_plus.io_utils import write_csv_records, write_json


NUMERIC_FIELDS = [
    "m_raw_ratio",
    "bbox_w",
    "bbox_h",
    "skeleton_length",
    "skeleton_ratio",
    "component_count",
    "main_orientation",
    "thickness_mean",
    "thickness_p90",
    "center_x",
    "center_y",
]


def _as_float(value: object) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except ValueError:
        return None


def summarize_domain_stats(rows: Iterable[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("domain", row.get("dataset_group", ""))].append(row)

    csv_rows: list[dict[str, object]] = []
    histogram: dict[str, object] = {}
    for domain, group in sorted(grouped.items()):
        record: dict[str, object] = {"domain": domain, "count": len(group)}
        histogram[domain] = {}
        for field in NUMERIC_FIELDS:
            values = [value for row in group if (value := _as_float(row.get(field))) is not None]
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            record[f"{field}_mean"] = round(float(arr.mean()), 10)
            record[f"{field}_median"] = round(float(np.median(arr)), 10)
            record[f"{field}_p10"] = round(float(np.percentile(arr, 10)), 10)
            record[f"{field}_p90"] = round(float(np.percentile(arr, 90)), 10)
            record[f"{field}_min"] = round(float(arr.min()), 10)
            record[f"{field}_max"] = round(float(arr.max()), 10)
            hist, edges = np.histogram(arr, bins=20)
            histogram[domain][field] = {
                "hist": [int(v) for v in hist],
                "edges": [round(float(v), 10) for v in edges],
            }
        csv_rows.append(record)
    return csv_rows, histogram


def write_domain_stats(rows: list[dict[str, str]], output_root: Path) -> dict[str, object]:
    csv_rows, histogram = summarize_domain_stats(rows)
    stats_path = output_root / "stats" / "domain_mask_stats.csv"
    hist_path = output_root / "stats" / "domain_mask_histograms.json"
    write_csv_records(stats_path, csv_rows)
    write_json(hist_path, histogram)
    return {"stats": str(stats_path), "histograms": str(hist_path), "domain_count": len(csv_rows)}

