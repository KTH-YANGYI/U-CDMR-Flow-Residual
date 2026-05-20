from __future__ import annotations

from pathlib import Path
import random

import numpy as np
from PIL import Image

from ucdmr_flow_residual_plus.image_utils import bool_mask, load_labelme_mask

from ucdmr_flow_residual_plus.constants import DOMAIN_TO_INDEX, resolve_existing
from ucdmr_flow_residual_plus.paths import dataset_path


CONDITION_KEYS = [
    "m_raw_path",
    "m_inpaint_path",
    "m_band_path",
    "m_gate_path",
    "m_skeleton_path",
    "m_sdf_path",
    "m_thickness_path",
]
M_RAW_INDEX = 0
M_INPAINT_INDEX = 1
M_BAND_INDEX = 2
M_GATE_INDEX = 3
CONDITION_CHANNELS = len(CONDITION_KEYS)


def load_rgb01(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_mask01(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def crop(arr: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    if arr.ndim == 2:
        return arr[y : y + size, x : x + size]
    return arr[y : y + size, x : x + size, :]


def crop_fixed(arr: np.ndarray, x: int, y: int, size: int, *, pad_value: float = 0.0, image_pad: bool = False) -> np.ndarray:
    if arr.ndim not in {2, 3}:
        raise ValueError(f"Expected 2D/3D array, got shape={arr.shape}")
    cropped = crop(arr, x, y, size)
    pad_h = max(0, size - cropped.shape[0])
    pad_w = max(0, size - cropped.shape[1])
    if pad_h == 0 and pad_w == 0:
        return cropped
    pad_spec: tuple[tuple[int, int], ...]
    if arr.ndim == 2:
        pad_spec = ((0, pad_h), (0, pad_w))
    else:
        pad_spec = ((0, pad_h), (0, pad_w), (0, 0))
    if image_pad and cropped.size:
        return np.pad(cropped, pad_spec, mode="edge")
    return np.pad(cropped, pad_spec, mode="constant", constant_values=pad_value)


def choose_crop(width: int, height: int, mask: np.ndarray, rng: random.Random, size: int, focus_probability: float) -> tuple[int, int]:
    if rng.random() < focus_probability and np.any(mask > 0.5):
        ys, xs = np.where(mask > 0.5)
        pick = rng.randrange(len(xs))
        cx = int(xs[pick])
        cy = int(ys[pick])
    else:
        cx = rng.randrange(width)
        cy = rng.randrange(height)
    x = max(0, min(width - size, cx - size // 2))
    y = max(0, min(height - size, cy - size // 2))
    return x, y


class PlusResidualDataset:
    def __init__(
        self,
        *,
        pseudo_rows: list[dict[str, str]],
        dataset_root: Path,
        tile_size: int,
        samples_per_epoch: int,
        seed: int,
        focus_probability: float = 0.85,
        style_dim: int = 16,
        style_dropout: float = 0.5,
    ) -> None:
        if not pseudo_rows:
            raise ValueError("No pseudo-normal rows were provided")
        self.rows = pseudo_rows
        self.dataset_root = dataset_root
        self.tile_size = int(tile_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.focus_probability = float(focus_probability)
        self.style_dim = int(style_dim)
        self.style_dropout = float(style_dropout)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, np.ndarray | int | str]:
        rng = random.Random(self.seed + index * 1000003)
        row = self.rows[index % len(self.rows)]
        target = load_rgb01(dataset_path(self.dataset_root, row["dataset_relative_path"]))
        context = load_rgb01(resolve_existing(row["pseudo_image_path"]))
        height, width = target.shape[:2]
        m_raw = load_mask01(resolve_existing(row["m_raw_path"]))
        size = self.tile_size
        x, y = choose_crop(width, height, m_raw, rng, size, self.focus_probability)
        conditions = [crop_fixed(load_mask01(resolve_existing(row[key])), x, y, size) for key in CONDITION_KEYS]
        style = np.asarray([rng.gauss(0.0, 1.0) for _ in range(self.style_dim)], dtype=np.float32)
        if rng.random() < self.style_dropout:
            style[:] = 0.0
        domain = row.get("domain", row.get("dataset_group", ""))
        return {
            "context": np.transpose(crop_fixed(context, x, y, size, image_pad=True), (2, 0, 1)).astype(np.float32),
            "target": np.transpose(crop_fixed(target, x, y, size, image_pad=True), (2, 0, 1)).astype(np.float32),
            "condition": np.stack(conditions, axis=0).astype(np.float32),
            "m_raw": conditions[M_RAW_INDEX][None, ...].astype(np.float32),
            "m_band": conditions[M_BAND_INDEX][None, ...].astype(np.float32),
            "m_gate": conditions[M_GATE_INDEX][None, ...].astype(np.float32),
            "style": style,
            "domain_idx": DOMAIN_TO_INDEX.get(domain, 0),
        }


def load_real_segmentation(row: dict[str, str], *, dataset_root: Path) -> tuple[np.ndarray, np.ndarray]:
    image_path = dataset_path(dataset_root, row["dataset_relative_path"])
    image = load_rgb01(image_path)
    height, width = image.shape[:2]
    if row.get("label") == "crack" and row.get("annotation_relative_path"):
        mask = bool_mask(load_labelme_mask(dataset_path(dataset_root, row["annotation_relative_path"]), (width, height))).astype(np.float32)
    else:
        mask = np.zeros((height, width), dtype=np.float32)
    return image, mask


def load_synthetic_segmentation(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    image = load_rgb01(resolve_existing(row.get("image_path", row.get("synthetic_image_path", ""))))
    mask = load_mask01(resolve_existing(row.get("mask_path", row.get("synthetic_mask_path", ""))))
    return image, (mask > 0.5).astype(np.float32)


class PlusSegmentationDataset:
    def __init__(
        self,
        *,
        real_rows: list[dict[str, str]],
        synthetic_rows: list[dict[str, str]],
        dataset_root: Path,
        tile_size: int,
        samples_per_epoch: int,
        seed: int,
        focus_probability: float = 0.8,
        synthetic_weight: int = 1,
    ) -> None:
        self.samples: list[dict[str, object]] = []
        for row in real_rows:
            self.samples.append({"kind": "real", "row": row})
        for row in synthetic_rows:
            for _ in range(max(0, int(synthetic_weight))):
                self.samples.append({"kind": "synthetic", "row": row})
        if not self.samples:
            raise ValueError("No segmentation samples were provided")
        self.dataset_root = dataset_root
        self.tile_size = int(tile_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.focus_probability = float(focus_probability)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, np.ndarray | str]:
        rng = random.Random(self.seed + index * 1000003)
        sample = self.samples[index % len(self.samples)]
        row = sample["row"]
        if not isinstance(row, dict):
            raise TypeError("invalid sample row")
        if sample["kind"] == "synthetic":
            image, mask = load_synthetic_segmentation(row)
            kind = "synthetic"
        else:
            image, mask = load_real_segmentation(row, dataset_root=self.dataset_root)
            kind = f"real_{row.get('label', '')}"
        height, width = image.shape[:2]
        size = self.tile_size
        x, y = choose_crop(width, height, mask, rng, size, self.focus_probability)
        return {
            "image": np.transpose(crop_fixed(image, x, y, size, image_pad=True), (2, 0, 1)).astype(np.float32),
            "mask": crop_fixed(mask, x, y, size)[None, ...].astype(np.float32),
            "sample_kind": kind,
        }
