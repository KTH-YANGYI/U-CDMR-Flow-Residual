from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ucdmr_flow_residual_plus.image_utils import save_mask
from ucdmr_flow_residual_plus.io_utils import ensure_dir, read_csv_records, write_csv_records, write_json

from ucdmr_flow_residual_plus.config import load_config, resolve_dataset_root, resolve_output_root
from ucdmr_flow_residual_plus.training.datasets import load_real_segmentation


def dry_run_summary(args: Any) -> dict[str, object]:
    config = load_config(args.config)
    output_root = resolve_output_root(config, args.output_root)
    return {
        "stage": "plus_eval_downstream",
        "checkpoint": str(args.checkpoint or output_root / args.stage_name / "checkpoints" / "latest.pt"),
        "split": args.split,
        "tile_size": args.tile_size,
        "dry_run": True,
    }


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    starts = list(range(0, max(length - tile_size + 1, 1), stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _predict(torch_module: Any, model: Any, image: np.ndarray, *, device: Any, tile_size: int, overlap: int) -> np.ndarray:
    h, w = image.shape[:2]
    th = min(tile_size, h)
    tw = min(tile_size, w)
    accum = np.zeros((h, w), dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)
    for y in _tile_starts(h, th, overlap):
        for x in _tile_starts(w, tw, overlap):
            tile = image[y : y + th, x : x + tw]
            tensor = torch_module.from_numpy(np.transpose(tile, (2, 0, 1))[None, ...]).to(device=device, dtype=torch_module.float32)
            logits = model(tensor).squeeze(0).squeeze(0).detach().cpu().numpy()
            accum[y : y + th, x : x + tw] += logits
            weights[y : y + th, x : x + tw] += 1.0
    logits = accum / np.maximum(weights, 1.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = float((pred & target).sum())
    fp = float((pred & ~target).sum())
    fn = float((~pred & target).sum())
    pred_sum = float(pred.sum())
    target_sum = float(target.sum())
    dice = 1.0 if target_sum == 0 and pred_sum == 0 else (2.0 * tp / max(2.0 * tp + fp + fn, 1e-8))
    iou = 1.0 if target_sum == 0 and pred_sum == 0 else (tp / max(tp + fp + fn, 1e-8))
    return {
        "dice": dice,
        "iou": iou,
        "precision": tp / max(tp + fp, 1e-8),
        "recall": tp / max(tp + fn, 1e-8),
        "boundary_f1": _boundary_f1(pred, target),
        "normal_false_positive_rate": pred_sum / max(float(pred.size), 1.0) if target_sum == 0 else 0.0,
    }


def _erode(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant")
    out = np.ones(mask.shape, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(max(0, int(radius))):
        padded = np.pad(out, 1, mode="constant")
        next_out = np.zeros_like(out)
        for dy in range(3):
            for dx in range(3):
                next_out |= padded[dy : dy + out.shape[0], dx : dx + out.shape[1]]
        out = next_out
    return out


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    return mask & ~_erode(mask)


def _boundary_f1(pred: np.ndarray, target: np.ndarray, *, tolerance: int = 2) -> float:
    pred_b = _boundary(pred)
    target_b = _boundary(target)
    pred_sum = float(pred_b.sum())
    target_sum = float(target_b.sum())
    if pred_sum == 0 and target_sum == 0:
        return 1.0
    if pred_sum == 0 or target_sum == 0:
        return 0.0
    precision = float((pred_b & _dilate(target_b, tolerance)).sum()) / max(pred_sum, 1.0)
    recall = float((target_b & _dilate(pred_b, tolerance)).sum()) / max(target_sum, 1.0)
    return 2.0 * precision * recall / max(precision + recall, 1e-8)


def _bucket(mask: np.ndarray) -> str:
    area = float(mask.sum())
    if area == 0:
        return "normal"
    ratio = area / max(float(mask.size), 1.0)
    if ratio < 1e-5:
        return "tiny"
    if ratio < 1e-4:
        return "small"
    if ratio < 1e-3:
        return "medium"
    return "large"


def _mean(rows: list[dict[str, object]], key: str) -> float:
    vals = [float(row[key]) for row in rows]
    return round(float(np.mean(vals)), 8) if vals else 0.0


def evaluate(args: Any) -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for plus evaluation.") from exc

    from ucdmr_flow_residual_plus.models.segmentation import SegmenterPlus

    config = load_config(args.config)
    dataset_root = resolve_dataset_root(config, args.dataset_root)
    output_root = resolve_output_root(config, args.output_root)
    split_manifest = Path(args.split_manifest) if args.split_manifest else output_root / "data" / "manifest_splits.csv"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_root / args.stage_name / "checkpoints" / "latest.pt"
    eval_root = Path(args.eval_output) if args.eval_output else output_root / args.stage_name / f"eval_{args.split}"
    ensure_dir(eval_root)
    rows = [row for row in read_csv_records(split_manifest) if row.get("split", args.split) == args.split and row.get("label") in {"crack", "normal"}]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model = SegmenterPlus(encoder_name=str(args.encoder or ckpt_args.get("encoder", "resnet34")), pretrained=False, base_channels=int(args.base_channels or ckpt_args.get("base_channels", 48))).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    records: list[dict[str, object]] = []
    pred_dir = eval_root / "pred_masks"
    with torch.no_grad():
        for row in rows:
            image, target = load_real_segmentation(row, dataset_root=dataset_root)
            prob = _predict(torch, model, image, device=device, tile_size=args.tile_size, overlap=args.tile_overlap)
            pred = prob >= args.threshold
            domain = row.get("domain", row.get("dataset_group", ""))
            sample_id = f"{domain}__{Path(row['dataset_relative_path']).stem}"
            pred_path = ""
            if args.save_predictions:
                pred_path = str(pred_dir / f"{sample_id}.png")
                save_mask(pred_path, pred)
            records.append({"sample_id": sample_id, "domain": domain, "label": row.get("label", ""), "size_bucket": _bucket(target), "prediction_path": pred_path, **{k: round(float(v), 8) for k, v in _metrics(pred, target > 0.5).items()}})
    csv_path = eval_root / "segmentation_eval.csv"
    write_csv_records(csv_path, records)
    by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_bucket: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in records:
        by_domain[str(row["domain"])].append(row)
        by_bucket[str(row["size_bucket"])].append(row)
    keys = ["dice", "iou", "precision", "recall", "boundary_f1", "normal_false_positive_rate"]
    summary = {
        "sample_count": len(records),
        "checkpoint": str(checkpoint_path),
        "per_sample": str(csv_path),
        "overall": {key: _mean(records, key) for key in keys},
        "by_domain": {name: {key: _mean(group, key) for key in keys} for name, group in sorted(by_domain.items())},
        "by_size_bucket": {name: {key: _mean(group, key) for key in keys} for name, group in sorted(by_bucket.items())},
    }
    write_json(eval_root / "segmentation_eval_summary.json", summary)
    print(summary)
