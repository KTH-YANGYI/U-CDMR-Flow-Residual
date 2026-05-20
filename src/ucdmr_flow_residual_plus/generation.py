from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import random

import numpy as np
from PIL import Image

from ucdmr_flow_residual_plus.image_utils import gray_from_rgb, save_mask
from ucdmr_flow_residual_plus.io_utils import ensure_dir, read_csv_records, write_csv_records, write_json

from ucdmr_flow_residual_plus.config import load_config, resolve_dataset_root, resolve_output_root
from ucdmr_flow_residual_plus.constants import DOMAIN_TO_INDEX, resolve_existing
from ucdmr_flow_residual_plus.paths import dataset_path
from ucdmr_flow_residual_plus.training.datasets import CONDITION_CHANNELS, CONDITION_KEYS, M_GATE_INDEX, M_RAW_INDEX, load_mask01, load_rgb01


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    starts = list(range(0, max(length - tile_size + 1, 1), stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(path)


def _same_shape(row: dict[str, str], width: int, height: int) -> bool:
    return int(float(row.get("mask_width", row.get("image_width", 0)))) == width and int(float(row.get("mask_height", row.get("image_height", 0)))) == height


def _condition(row: dict[str, str]) -> np.ndarray:
    return np.stack([load_mask01(resolve_existing(row[key])) for key in CONDITION_KEYS], axis=0).astype(np.float32)


def _tiled_delta(
    *,
    torch_module: Any,
    model: Any,
    image: np.ndarray,
    condition: np.ndarray,
    domain: str,
    style: np.ndarray,
    device: Any,
    tile_size: int,
    overlap: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    tile_h = min(tile_size, height)
    tile_w = min(tile_size, width)
    accum = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width, 1), dtype=np.float32)
    domain_idx = torch_module.tensor([DOMAIN_TO_INDEX.get(domain, 0)], device=device)
    style_tensor = torch_module.from_numpy(style[None, ...]).to(device=device, dtype=torch_module.float32)
    for y in _tile_starts(height, tile_h, overlap):
        for x in _tile_starts(width, tile_w, overlap):
            img_tile = image[y : y + tile_h, x : x + tile_w]
            cond_tile = condition[:, y : y + tile_h, x : x + tile_w]
            img_tensor = torch_module.from_numpy(np.transpose(img_tile, (2, 0, 1))[None, ...]).to(device=device, dtype=torch_module.float32)
            cond_tensor = torch_module.from_numpy(cond_tile[None, ...]).to(device=device, dtype=torch_module.float32)
            pred = model(img_tensor, cond_tensor, domain_idx, style_tensor).squeeze(0).detach().cpu().numpy()
            pred = np.transpose(pred, (1, 2, 0))
            accum[y : y + tile_h, x : x + tile_w] += pred
            weights[y : y + tile_h, x : x + tile_w] += 1.0
    return accum / np.maximum(weights, 1.0)


def _tiled_flow_delta(
    *,
    torch_module: Any,
    model: Any,
    image: np.ndarray,
    condition: np.ndarray,
    domain: str,
    style: np.ndarray,
    device: Any,
    tile_size: int,
    overlap: int,
    steps: int,
    sampler: str,
    sigma: float,
    seed: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    tile_h = min(tile_size, height)
    tile_w = min(tile_size, width)
    accum = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width, 1), dtype=np.float32)
    domain_idx = torch_module.tensor([DOMAIN_TO_INDEX.get(domain, 0)], device=device)
    style_tensor = torch_module.from_numpy(style[None, ...]).to(device=device, dtype=torch_module.float32)
    step_count = max(1, int(steps))
    dt = 1.0 / float(step_count)
    tile_index = 0
    for y in _tile_starts(height, tile_h, overlap):
        for x in _tile_starts(width, tile_w, overlap):
            img_tile = image[y : y + tile_h, x : x + tile_w]
            cond_tile = condition[:, y : y + tile_h, x : x + tile_w]
            img_tensor = torch_module.from_numpy(np.transpose(img_tile, (2, 0, 1))[None, ...]).to(device=device, dtype=torch_module.float32)
            cond_tensor = torch_module.from_numpy(cond_tile[None, ...]).to(device=device, dtype=torch_module.float32)
            gate = cond_tensor[:, M_GATE_INDEX : M_GATE_INDEX + 1].clamp(0.0, 1.0)
            try:
                generator = torch_module.Generator(device=device)
            except TypeError:
                generator = torch_module.Generator()
            generator.manual_seed(int(seed) + tile_index * 1009)
            x_t = torch_module.randn((1, 3, tile_h, tile_w), generator=generator, device=device, dtype=torch_module.float32)
            x_t = x_t * float(sigma) * gate
            for step in range(step_count):
                t0 = step / float(step_count)
                t = torch_module.full((1, 1, 1, 1), t0, device=device, dtype=torch_module.float32)
                v0 = model(x_t, t, img_tensor, cond_tensor, domain_idx, style_tensor)
                if sampler == "heun":
                    x_pred = (x_t + dt * v0) * gate
                    t1 = torch_module.full((1, 1, 1, 1), min(1.0, t0 + dt), device=device, dtype=torch_module.float32)
                    v1 = model(x_pred, t1, img_tensor, cond_tensor, domain_idx, style_tensor)
                    x_t = x_t + 0.5 * dt * (v0 + v1)
                else:
                    x_t = x_t + dt * v0
                x_t = x_t * gate
            pred = x_t.squeeze(0).detach().cpu().numpy()
            pred = np.transpose(pred, (1, 2, 0))
            accum[y : y + tile_h, x : x + tile_w] += pred
            weights[y : y + tile_h, x : x + tile_w] += 1.0
            tile_index += 1
    return accum / np.maximum(weights, 1.0)


def dry_run_generate_summary(args: Any) -> dict[str, object]:
    config = load_config(args.config)
    output_root = resolve_output_root(config, args.output_root)
    masks_manifest = output_root / "sampled_masks" / "sampled_masks_manifest.csv" if args.mask_source == "descriptor_flow" else output_root / "masks" / "masks_manifest.csv"
    default_checkpoint = output_root / "residual_flow_plus" / "checkpoints" / "latest.pt" if args.residual_source == "flow" else output_root / "residual_renderer_plus" / "checkpoints" / "latest.pt"
    return {
        "stage": "plus_generate_synthetic",
        "output_root": str(output_root),
        "residual_source": args.residual_source,
        "checkpoint": str(args.checkpoint or default_checkpoint),
        "masks_manifest": str(args.masks_manifest or masks_manifest),
        "mask_source": args.mask_source,
        "flow_steps": args.flow_steps,
        "flow_sampler": args.flow_sampler,
        "flow_sigma": args.flow_sigma,
        "teacher_checkpoint": str(args.teacher_checkpoint or output_root / args.teacher_stage_name / "checkpoints" / "latest.pt"),
        "max_samples": args.max_samples,
        "dry_run": True,
    }


def generate(args: Any) -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for plus synthetic generation.") from exc

    from ucdmr_flow_residual_plus.models.residual_flow import ResidualFlowUNet
    from ucdmr_flow_residual_plus.models.residual_renderer import ResidualRendererPlus
    from ucdmr_flow_residual_plus.evaluation import _metrics as segmentation_metrics
    from ucdmr_flow_residual_plus.evaluation import _predict as predict_segmentation
    from ucdmr_flow_residual_plus.models.residual_renderer import SegmenterPlus

    config = load_config(args.config)
    dataset_root = resolve_dataset_root(config, args.dataset_root)
    output_root = resolve_output_root(config, args.output_root)
    split_manifest = Path(args.split_manifest) if args.split_manifest else output_root / "data" / "manifest_splits.csv"
    default_masks = output_root / "sampled_masks" / "sampled_masks_manifest.csv" if args.mask_source == "descriptor_flow" else output_root / "masks" / "masks_manifest.csv"
    masks_manifest = Path(args.masks_manifest) if args.masks_manifest else default_masks
    default_checkpoint = output_root / "residual_flow_plus" / "checkpoints" / "latest.pt" if args.residual_source == "flow" else output_root / "residual_renderer_plus" / "checkpoints" / "latest.pt"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint
    synthetic_root = Path(args.synthetic_output) if args.synthetic_output else output_root / "synthetic" / "raw"
    ensure_dir(synthetic_root)
    normals = [row for row in read_csv_records(split_manifest) if row.get("label") == "normal" and row.get("split", args.split) == args.split]
    masks = [
        row
        for row in read_csv_records(masks_manifest)
        if row.get("split") in {args.split, "generated"}
    ]
    if args.max_samples is not None:
        normals = normals[: args.max_samples]
    if not normals:
        raise SystemExit(f"No normal rows found for split={args.split}: {split_manifest}")
    if not masks:
        raise SystemExit(f"No mask rows found: {masks_manifest}")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    style_dim = int(args.style_dim or ckpt_args.get("style_dim", 16))
    if args.residual_source == "flow":
        model = ResidualFlowUNet(
            condition_channels=CONDITION_CHANNELS,
            base_channels=int(args.base_channels or ckpt_args.get("base_channels", 48)),
            style_dim=style_dim,
            time_dim=int(args.time_dim or ckpt_args.get("time_dim", 128)),
            max_velocity=float(args.max_velocity if args.max_velocity is not None else ckpt_args.get("max_velocity", 0.0)),
        ).to(device)
    else:
        model = ResidualRendererPlus(
            encoder_name=str(args.encoder or ckpt_args.get("encoder", "resnet34")),
            pretrained=False,
            base_channels=int(args.base_channels or ckpt_args.get("base_channels", 48)),
            condition_channels=CONDITION_CHANNELS,
            style_dim=style_dim,
            max_delta=float(args.max_delta or ckpt_args.get("max_delta", 1.0)),
        ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    teacher = None
    teacher_checkpoint = Path(args.teacher_checkpoint) if args.teacher_checkpoint else output_root / args.teacher_stage_name / "checkpoints" / "latest.pt"
    if teacher_checkpoint.exists():
        try:
            teacher_ckpt = torch.load(teacher_checkpoint, map_location=device, weights_only=False)
        except TypeError:
            teacher_ckpt = torch.load(teacher_checkpoint, map_location=device)
        teacher_args = teacher_ckpt.get("args", {})
        teacher = SegmenterPlus(
            encoder_name=str(teacher_args.get("encoder", "resnet34")),
            pretrained=False,
            base_channels=int(teacher_args.get("base_channels", 48)),
        ).to(device)
        teacher.load_state_dict(teacher_ckpt["model"])
        teacher.eval()
    seed_mask = int(args.seed_mask if args.seed_mask is not None else args.seed)
    seed_residual = int(args.seed_residual if args.seed_residual is not None else args.seed + 100000)
    rng_mask = random.Random(seed_mask)
    rng_residual = random.Random(seed_residual)
    records: list[dict[str, object]] = []
    with torch.no_grad():
        for idx, normal in enumerate(normals):
            domain = normal.get("domain", normal.get("dataset_group", ""))
            image = load_rgb01(dataset_path(dataset_root, normal["dataset_relative_path"]))
            height, width = image.shape[:2]
            candidates = [row for row in masks if row.get("domain", row.get("dataset_group", "")) == domain and _same_shape(row, width, height)]
            if not candidates:
                continue
            mask_row = rng_mask.choice(candidates)
            condition = _condition(mask_row)
            style = np.asarray([rng_residual.gauss(0.0, 1.0) for _ in range(style_dim)], dtype=np.float32)
            sample_seed_residual = seed_residual + idx * 1000003
            sample_seed_mask = seed_mask + idx * 1000003
            if args.residual_source == "flow":
                delta = _tiled_flow_delta(
                    torch_module=torch,
                    model=model,
                    image=image,
                    condition=condition,
                    domain=domain,
                    style=style,
                    device=device,
                    tile_size=args.tile_size,
                    overlap=args.tile_overlap,
                    steps=args.flow_steps,
                    sampler=args.flow_sampler,
                    sigma=args.flow_sigma,
                    seed=sample_seed_residual,
                )
            else:
                delta = _tiled_delta(torch_module=torch, model=model, image=image, condition=condition, domain=domain, style=style, device=device, tile_size=args.tile_size, overlap=args.tile_overlap)
            gate = condition[M_GATE_INDEX]
            synthetic = np.clip(image + gate[..., None] * delta, 0.0, 1.0)
            change = np.abs(synthetic - image)
            raw = condition[M_RAW_INDEX] > 0.5
            outside = gate <= 0.001
            inside_change = float(change[raw].mean()) if np.any(raw) else 0.0
            outside_change = float(change[outside].mean()) if np.any(outside) else 0.0
            support = gray_from_rgb(change * 255.0) > args.support_threshold
            mask_residual_iou = float((support & raw).sum() / max(int((support | raw).sum()), 1))
            teacher_dice: str | float = ""
            teacher_recall: str | float = ""
            teacher_fp: str | float = ""
            if teacher is not None:
                prob = predict_segmentation(torch, teacher, synthetic, device=device, tile_size=args.tile_size, overlap=args.tile_overlap)
                teacher_pred = prob >= args.teacher_threshold
                teacher_parts = segmentation_metrics(teacher_pred, raw)
                outside_mask = ~raw
                teacher_dice = round(float(teacher_parts["dice"]), 8)
                teacher_recall = round(float(teacher_parts["recall"]), 8)
                teacher_fp = round(float((teacher_pred & outside_mask).sum() / max(int(outside_mask.sum()), 1)), 8)
            synthetic_id = f"plus_syn_{idx:06d}_{domain}_{Path(normal['dataset_relative_path']).stem}_{mask_row.get('sample_id', idx)}"
            image_path = synthetic_root / "images" / f"{synthetic_id}.png"
            mask_path = synthetic_root / "masks" / f"{synthetic_id}.png"
            residual_path = synthetic_root / "residual_abs" / f"{synthetic_id}.png"
            _save_rgb(image_path, synthetic)
            save_mask(mask_path, raw)
            ensure_dir(residual_path.parent)
            Image.fromarray(np.clip(gray_from_rgb(np.abs(delta) * 255.0), 0, 255).astype(np.uint8), mode="L").save(residual_path)
            records.append(
                {
                    "synthetic_id": synthetic_id,
                    "domain": domain,
                    "source_split": normal.get("split", ""),
                    "normal_source_path": normal["dataset_relative_path"],
                    "mask_source_id": mask_row.get("sample_id", ""),
                    "native_width": width,
                    "native_height": height,
                    "mask_source": args.mask_source,
                    "residual_source": "flow" if args.residual_source == "flow" else "deterministic_renderer",
                    "residual_flow_checkpoint": str(checkpoint_path) if args.residual_source == "flow" else "",
                    "renderer_checkpoint": str(checkpoint_path) if args.residual_source == "renderer" else "",
                    "flow_steps": args.flow_steps if args.residual_source == "flow" else "",
                    "flow_sampler": args.flow_sampler if args.residual_source == "flow" else "",
                    "flow_sigma": args.flow_sigma if args.residual_source == "flow" else "",
                    "seed_residual": sample_seed_residual,
                    "seed_mask": sample_seed_mask,
                    "m_raw_ratio": round(float(raw.mean()), 10),
                    "mask_area_ratio": round(float(raw.mean()), 10),
                    "outside_change": round(outside_change, 8),
                    "inside_change": round(inside_change, 8),
                    "residual_leak": round(outside_change / max(inside_change, 1e-8), 8),
                    "mask_residual_iou": round(mask_residual_iou, 8),
                    "teacher_dice": teacher_dice,
                    "teacher_recall_on_mask": teacher_recall,
                    "teacher_fp_outside_mask": teacher_fp,
                    "quality_score": "",
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "residual_path": str(residual_path),
                    "generation_formula": "I_syn = I_normal + gate(M_syn) * Delta_residual",
                }
            )
    manifest_path = synthetic_root / "synthetic_manifest.csv"
    write_csv_records(manifest_path, records)
    summary = {
        "sample_count": len(records),
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path),
        "residual_source": args.residual_source,
        "mean_residual_leak": round(float(np.mean([row["residual_leak"] for row in records])) if records else math.nan, 8),
    }
    write_json(synthetic_root / "synthetic_generation_summary.json", summary)
    print(summary)


def dry_run_filter_summary(args: Any) -> dict[str, object]:
    config = load_config(args.config)
    output_root = resolve_output_root(config, args.output_root)
    return {
        "stage": "plus_filter_synthetic",
        "synthetic_manifest": str(args.synthetic_manifest or output_root / "synthetic" / "raw" / "synthetic_manifest.csv"),
        "dry_run": True,
    }


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except ValueError:
        return default


def filter_synthetic(args: Any) -> None:
    config = load_config(args.config)
    output_root = resolve_output_root(config, args.output_root)
    synthetic_manifest = Path(args.synthetic_manifest) if args.synthetic_manifest else output_root / "synthetic" / "raw" / "synthetic_manifest.csv"
    output_dir = Path(args.filtered_output) if args.filtered_output else output_root / "synthetic" / "filtered"
    rows = read_csv_records(synthetic_manifest)
    scored = []
    kept = []
    for row in rows:
        leak = _float(row.get("residual_leak"), math.inf)
        outside = _float(row.get("outside_change"), math.inf)
        area = _float(row.get("mask_area_ratio"), 0.0)
        iou = _float(row.get("mask_residual_iou"), 0.0)
        teacher = _float(row.get("teacher_dice"), math.nan)
        topology = _float(row.get("topology_score"), math.nan)
        keep = leak <= args.max_residual_leak and outside <= args.max_outside_change and args.min_mask_area <= area <= args.max_mask_area and iou >= args.min_mask_residual_iou
        score = 1.0 - min(leak / max(args.max_residual_leak, 1e-8), 2.0) * 0.3 - min(outside / max(args.max_outside_change, 1e-8), 2.0) * 0.3 + min(iou, 1.0) * 0.4
        if not math.isnan(teacher):
            score += min(max(teacher, 0.0), 1.0) * 0.1
        if not math.isnan(topology):
            score += min(max(topology, 0.0), 1.0) * 0.1
        record = {**row, "filter_keep": "1" if keep else "0", "quality_score": round(float(score), 8), "filter_reason": "kept" if keep else "threshold_failed"}
        scored.append(record)
        if keep:
            kept.append(record)
    ensure_dir(output_dir)
    scored_path = output_dir / "synthetic_scored.csv"
    kept_path = output_dir / "synthetic_filtered.csv"
    write_csv_records(scored_path, scored)
    write_csv_records(kept_path, kept)
    summary = {"input_count": len(rows), "kept_count": len(kept), "scored": str(scored_path), "filtered": str(kept_path)}
    write_json(output_dir / "synthetic_filter_report.json", summary)
    print(summary)
