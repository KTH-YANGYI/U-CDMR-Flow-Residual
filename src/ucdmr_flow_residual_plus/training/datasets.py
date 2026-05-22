from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image

from ucdmr_flow_residual_plus.image_utils import bool_mask, load_labelme_mask

from ucdmr_flow_residual_plus.constants import residual_domain_index, resolve_existing
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


def _float_from_row(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value in {"", None}:
            return default
        return float(value)
    except ValueError:
        return default


def _int_from_row(row: dict[str, str], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        try:
            value = row.get(key, "")
            if value not in {"", None}:
                return int(float(value))
        except ValueError:
            continue
    return None


def native_size_from_row(row: dict[str, str], *, dataset_root: Path | None = None) -> tuple[int, int]:
    height = _int_from_row(row, ("native_height", "image_height", "mask_height", "height"))
    width = _int_from_row(row, ("native_width", "image_width", "mask_width", "width"))
    if height is not None and width is not None and height > 0 and width > 0:
        return height, width
    if dataset_root is not None and row.get("dataset_relative_path"):
        with Image.open(dataset_path(dataset_root, row["dataset_relative_path"])) as image:
            width, height = image.size
        row.setdefault("native_height", str(height))
        row.setdefault("native_width", str(width))
        return height, width
    raise ValueError(f"Cannot infer native size for row={row.get('sample_id') or row.get('dataset_relative_path', '<unknown>')}")


def _pad_chw(arr: np.ndarray, height: int, width: int, *, image_pad: bool) -> np.ndarray:
    pad_h = max(0, height - arr.shape[-2])
    pad_w = max(0, width - arr.shape[-1])
    if pad_h == 0 and pad_w == 0:
        return arr
    pad_spec = ((0, 0), (0, pad_h), (0, pad_w))
    if image_pad:
        return np.pad(arr, pad_spec, mode="edge")
    return np.pad(arr, pad_spec, mode="constant", constant_values=0.0)


def native_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    max_hw_by_key: dict[str, tuple[int, int]] = {}
    for item in batch:
        for key, value in item.items():
            if isinstance(value, np.ndarray) and value.ndim == 3:
                h, w = int(value.shape[-2]), int(value.shape[-1])
                old_h, old_w = max_hw_by_key.get(key, (0, 0))
                max_hw_by_key[key] = (max(old_h, h), max(old_w, w))

    out: dict[str, Any] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        first = values[0]
        if isinstance(first, np.ndarray):
            if first.ndim == 3:
                h, w = max_hw_by_key[key]
                image_pad = key in {"context", "target", "image"}
                out[key] = torch.from_numpy(np.stack([_pad_chw(value, h, w, image_pad=image_pad) for value in values], axis=0))
            else:
                out[key] = torch.from_numpy(np.stack(values, axis=0))
        elif isinstance(first, (int, np.integer)):
            out[key] = torch.tensor(values, dtype=torch.long)
        elif isinstance(first, (float, np.floating)):
            out[key] = torch.tensor(values, dtype=torch.float32)
        else:
            out[key] = values
    return out


def strict_native_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    shapes: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    for item in batch:
        for key, value in item.items():
            if isinstance(value, np.ndarray):
                shapes[key].add(tuple(value.shape))
    mixed = {key: sorted(value) for key, value in shapes.items() if len(value) > 1}
    if mixed:
        sample_ids = [str(item.get("sample_id", "")) for item in batch]
        raise ValueError(f"strict_native_collate received mixed tensor shapes: {mixed}; sample_ids={sample_ids}")
    return native_collate(batch)


class SizeBucketBatchSampler:
    """Batch pseudo rows by native HxW so residual-flow training never pads small images."""

    def __init__(
        self,
        *,
        rows: list[dict[str, str]],
        dataset_root: Path,
        samples_per_epoch: int,
        batch_size: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = True,
    ) -> None:
        if not rows:
            raise ValueError("SizeBucketBatchSampler requires at least one row")
        self.rows = rows
        self.samples_per_epoch = int(samples_per_epoch)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if self.world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        self.row_size_keys = [native_size_from_row(row, dataset_root=dataset_root) for row in rows]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def bucket_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for height, width in self.row_size_keys:
            key = f"{height}x{width}"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def _global_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1009)
        indices = list(range(self.samples_per_epoch))
        rng.shuffle(indices)
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index in indices:
            buckets[self.row_size_keys[index % len(self.rows)]].append(index)
        batches: list[list[int]] = []
        for key in sorted(buckets):
            bucket = buckets[key]
            rng.shuffle(bucket)
            full = (len(bucket) // self.batch_size) * self.batch_size
            if full:
                for start in range(0, full, self.batch_size):
                    batches.append(bucket[start : start + self.batch_size])
            if not self.drop_last and full < len(bucket):
                batches.append(bucket[full:])
        rng.shuffle(batches)
        usable = (len(batches) // self.world_size) * self.world_size
        return batches[:usable] if self.drop_last else batches

    def __iter__(self):
        batches = self._global_batches()
        return iter(batches[self.rank :: self.world_size])

    def __len__(self) -> int:
        batches = self._global_batches()
        return len(batches[self.rank :: self.world_size])


class PlusResidualDataset:
    def __init__(
        self,
        *,
        pseudo_rows: list[dict[str, str]],
        dataset_root: Path,
        samples_per_epoch: int,
        seed: int,
        style_dim: int = 16,
        style_dropout: float = 0.5,
    ) -> None:
        if not pseudo_rows:
            raise ValueError("No pseudo-normal rows were provided")
        self.rows = pseudo_rows
        self.dataset_root = dataset_root
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.style_dim = int(style_dim)
        self.style_dropout = float(style_dropout)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, np.ndarray | int | float | str]:
        rng = random.Random(self.seed + index * 1000003)
        row = self.rows[index % len(self.rows)]
        target = load_rgb01(dataset_path(self.dataset_root, row["dataset_relative_path"]))
        context = load_rgb01(resolve_existing(row["pseudo_image_path"]))
        sample_id = row.get("sample_id", "")
        if target.shape[:2] != context.shape[:2]:
            raise ValueError(f"target/context shape mismatch for {sample_id}: {target.shape} vs {context.shape}")
        height, width = target.shape[:2]
        conditions = [load_mask01(resolve_existing(row[key])) for key in CONDITION_KEYS]
        for key, cond in zip(CONDITION_KEYS, conditions):
            if cond.shape != target.shape[:2]:
                raise ValueError(f"{key} shape mismatch for {sample_id}: {cond.shape} vs {target.shape[:2]}")
        target_chw = np.transpose(target, (2, 0, 1)).astype(np.float32)
        context_chw = np.transpose(context, (2, 0, 1)).astype(np.float32)
        valid_mask = np.ones((1, height, width), dtype=np.float32)
        style = np.asarray([rng.gauss(0.0, 1.0) for _ in range(self.style_dim)], dtype=np.float32)
        if rng.random() < self.style_dropout:
            style[:] = 0.0
        domain = row.get("domain", row.get("dataset_group", ""))
        domain_idx = residual_domain_index(domain)
        return {
            "context": context_chw,
            "target": target_chw,
            "condition": np.stack(conditions, axis=0).astype(np.float32),
            "m_raw": conditions[M_RAW_INDEX][None, ...].astype(np.float32),
            "m_band": conditions[M_BAND_INDEX][None, ...].astype(np.float32),
            "m_gate": conditions[M_GATE_INDEX][None, ...].astype(np.float32),
            "valid_mask": valid_mask,
            "style": style,
            "domain_idx": domain_idx,
            "sample_id": sample_id,
            "domain": domain,
            "dataset_relative_path": row.get("dataset_relative_path", ""),
            "pseudo_image_path": row.get("pseudo_image_path", ""),
            "native_height": height,
            "native_width": width,
            "m_raw_ratio": _float_from_row(row, "m_raw_ratio"),
            "m_band_ratio": _float_from_row(row, "m_band_ratio"),
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
        samples_per_epoch: int,
        seed: int,
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
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, np.ndarray | str]:
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
        image_chw = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask_chw = mask[None, ...].astype(np.float32)
        valid_mask = np.ones((1, height, width), dtype=np.float32)
        return {
            "image": image_chw,
            "mask": mask_chw,
            "valid_mask": valid_mask,
            "sample_kind": kind,
        }
