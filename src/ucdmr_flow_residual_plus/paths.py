from __future__ import annotations

from pathlib import Path


DEFAULT_DATASET_ROOT = Path("/Users/yangyi/Desktop/masterthesis/dataset0505_crop640_roi_dphone")
DEFAULT_CONFIG = Path("configs/methods/u_cdmr_flow_residual_plus/data.yaml")
DEFAULT_METHOD_ROOT = Path("artifacts/dataset0505_crop640_roi_dphone/methods/u_cdmr_flow_residual_plus")


def dataset_path(dataset_root: Path, value: str | Path) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw
    return dataset_root / raw

