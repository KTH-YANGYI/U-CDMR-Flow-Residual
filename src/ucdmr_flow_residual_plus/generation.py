from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import random

import numpy as np
from PIL import Image

from ucdmr_flow_residual_plus.image_utils import gray_from_rgb, save_mask
from ucdmr_flow_residual_plus.io_utils import ensure_dir, read_csv_records, write_csv_records, write_json

from ucdmr_flow_residual_plus.config import load_config, nested_get, resolve_dataset_root, resolve_output_root
from ucdmr_flow_residual_plus.constants import base_domain_name, effective_domain, residual_domain_index, resolve_existing
from ucdmr_flow_residual_plus.paths import dataset_path
from ucdmr_flow_residual_plus.training.datasets import CONDITION_CHANNELS, CONDITION_KEYS, M_GATE_INDEX, M_RAW_INDEX, load_mask01, load_rgb01


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(path)


def _same_shape(row: dict[str, str], width: int, height: int) -> bool:
    return int(float(row.get("mask_width", row.get("image_width", 0)))) == width and int(float(row.get("mask_height", row.get("image_height", 0)))) == height


def _condition(row: dict[str, str]) -> np.ndarray:
    return np.stack([load_mask01(resolve_existing(row[key])) for key in CONDITION_KEYS], axis=0).astype(np.float32)


def _float_or_none(value: object) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except ValueError:
        return None


def _row_value(row: dict[str, str], key: str) -> str:
    return row.get(key) or row.get(f"source_template_{key}", "")


def _crop_box(row: dict[str, str]) -> tuple[float, float, float, float] | None:
    values = [_float_or_none(_row_value(row, key)) for key in ("crop_left", "crop_top", "crop_right", "crop_bottom")]
    if any(value is None for value in values):
        return None
    left, top, right, bottom = values
    if left is None or top is None or right is None or bottom is None or right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _crop_match_score(normal: dict[str, str], mask: dict[str, str]) -> float | None:
    normal_box = _crop_box(normal)
    mask_box = _crop_box(mask)
    if normal_box is None or mask_box is None:
        return None
    width = max(abs(normal_box[2] - normal_box[0]), abs(mask_box[2] - mask_box[0]), 1.0)
    height = max(abs(normal_box[3] - normal_box[1]), abs(mask_box[3] - mask_box[1]), 1.0)
    return float(
        (
            abs(normal_box[0] - mask_box[0]) / width
            + abs(normal_box[2] - mask_box[2]) / width
            + abs(normal_box[1] - mask_box[1]) / height
            + abs(normal_box[3] - mask_box[3]) / height
        )
        / 4.0
    )


def _normal_mask_match_score(normal: dict[str, str], mask: dict[str, str]) -> tuple[float, str]:
    score = 0.0
    reasons: list[str] = []
    normal_split_key = normal.get("split_key", "")
    mask_split_key = _row_value(mask, "split_key")
    if normal_split_key and mask_split_key:
        if normal_split_key == mask_split_key:
            reasons.append("same_split_key")
        else:
            score += 0.5
            reasons.append("different_split_key")
    normal_video = normal.get("video_id", "")
    mask_video = _row_value(mask, "video_id")
    if normal_video and mask_video:
        if normal_video == mask_video:
            reasons.append("same_video")
        else:
            score += 0.25
            reasons.append("different_video")
    crop_score = _crop_match_score(normal, mask)
    if crop_score is None:
        reasons.append("no_crop_match")
    else:
        score += crop_score
        reasons.append(f"crop_l1={crop_score:.6f}")
    if mask.get("placement_clamped", "0") == "1":
        score += 10.0
        reasons.append("placement_clamped")
    return score, ";".join(reasons)


def _checkpoint_model_nonfinite_tensors(torch_module: Any, ckpt: dict[str, Any]) -> list[str]:
    bad: list[str] = []
    for key, value in ckpt.get("model", {}).items():
        if torch_module.is_tensor(value) and not torch_module.isfinite(value).all():
            bad.append(key)
    return bad


def _masked_tv(value: np.ndarray, mask: np.ndarray) -> float:
    mask = mask.astype(bool)
    if value.shape[0] < 2 or value.shape[1] < 2:
        return 0.0
    y_mask = mask[1:, :] & mask[:-1, :]
    x_mask = mask[:, 1:] & mask[:, :-1]
    parts = []
    if np.any(y_mask):
        parts.append(float(np.abs(value[1:, :] - value[:-1, :])[y_mask].mean()))
    if np.any(x_mask):
        parts.append(float(np.abs(value[:, 1:] - value[:, :-1])[x_mask].mean()))
    return float(np.mean(parts)) if parts else 0.0


def _residual_quality_metrics(
    *,
    applied_delta: np.ndarray,
    gate: np.ndarray,
    raw: np.ndarray,
    color_speckle_threshold: float,
) -> dict[str, float | int]:
    gate_mask = gate > 0.001
    if not np.any(gate_mask):
        return {
            "residual_chroma_mean": 0.0,
            "residual_chroma_p99": 0.0,
            "color_speckle_ratio": 0.0,
            "color_speckle_pixels": 0,
            "residual_tv_mean": 0.0,
            "residual_highfreq_ratio": 0.0,
        }
    chroma = applied_delta - applied_delta.mean(axis=2, keepdims=True)
    chroma_mag = np.max(np.abs(chroma), axis=2)
    residual_luma_abs = gray_from_rgb(np.abs(applied_delta))
    speckles = (chroma_mag > float(color_speckle_threshold)) & gate_mask & (residual_luma_abs > float(color_speckle_threshold) * 0.5)
    tv_mask = gate_mask | raw.astype(bool)
    tv = _masked_tv(residual_luma_abs, tv_mask)
    mean_residual = float(residual_luma_abs[tv_mask].mean()) if np.any(tv_mask) else 0.0
    return {
        "residual_chroma_mean": round(float(chroma_mag[gate_mask].mean()), 8),
        "residual_chroma_p99": round(float(np.percentile(chroma_mag[gate_mask], 99)), 8),
        "color_speckle_ratio": round(float(speckles.sum() / max(int(gate_mask.sum()), 1)), 10),
        "color_speckle_pixels": int(speckles.sum()),
        "residual_tv_mean": round(float(tv), 8),
        "residual_highfreq_ratio": round(float(tv / max(mean_residual, 1e-8)), 8),
    }


def _flow_delta(
    *,
    torch_module: Any,
    model: Any,
    image: np.ndarray,
    condition: np.ndarray,
    domain: str,
    style: np.ndarray,
    device: Any,
    steps: int,
    sampler: str,
    sigma: float,
    max_delta: float | None,
    seed: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    domain_idx = torch_module.tensor([residual_domain_index(domain)], device=device)
    style_tensor = torch_module.from_numpy(style[None, ...]).to(device=device, dtype=torch_module.float32)
    img_tensor = torch_module.from_numpy(np.transpose(image, (2, 0, 1))[None, ...]).to(device=device, dtype=torch_module.float32)
    cond_tensor = torch_module.from_numpy(condition[None, ...]).to(device=device, dtype=torch_module.float32)
    gate = cond_tensor[:, M_GATE_INDEX : M_GATE_INDEX + 1].clamp(0.0, 1.0)
    gate_support = (gate > 0.001).to(dtype=torch_module.float32)
    gate_state_weight = torch_module.sqrt(gate) * gate_support
    try:
        generator = torch_module.Generator(device=device)
    except TypeError:
        generator = torch_module.Generator()
    generator.manual_seed(int(seed))
    x_t = torch_module.randn((1, 3, height, width), generator=generator, device=device, dtype=torch_module.float32)
    x_t = x_t * float(sigma) * gate_state_weight
    step_count = max(1, int(steps))
    dt = 1.0 / float(step_count)
    for step in range(step_count):
        t0 = step / float(step_count)
        t = torch_module.full((1, 1, 1, 1), t0, device=device, dtype=torch_module.float32)
        v0 = model(x_t, t, img_tensor, cond_tensor, domain_idx, style_tensor)
        if sampler == "heun":
            x_pred = (x_t + dt * v0) * gate_support
            t1 = torch_module.full((1, 1, 1, 1), min(1.0, t0 + dt), device=device, dtype=torch_module.float32)
            v1 = model(x_pred, t1, img_tensor, cond_tensor, domain_idx, style_tensor)
            x_t = x_t + 0.5 * dt * (v0 + v1)
        else:
            x_t = x_t + dt * v0
        x_t = x_t * gate_support
    delta = x_t.squeeze(0).detach().cpu().numpy()
    delta = np.transpose(delta, (1, 2, 0))
    if max_delta is not None and max_delta > 0:
        delta = np.clip(delta, -float(max_delta), float(max_delta))
    return delta


def dry_run_generate_summary(args: Any) -> dict[str, object]:
    config = load_config(args.config)
    output_root = resolve_output_root(config, args.output_root)
    masks_manifest = output_root / "sampled_masks" / "sampled_masks_manifest.csv" if args.mask_source == "descriptor_flow" else output_root / "masks" / "masks_manifest.csv"
    latest_finite = output_root / "residual_flow_plus" / "checkpoints" / "latest_finite.pt"
    latest = output_root / "residual_flow_plus" / "checkpoints" / "latest.pt"
    default_checkpoint = latest_finite if latest_finite.exists() else latest
    mask_flow_checkpoint = args.mask_flow_checkpoint or output_root / "mask_descriptor_flow" / "checkpoints" / "latest.pt"
    return {
        "stage": "plus_generate_synthetic",
        "output_root": str(output_root),
        "residual_source": "flow",
        "checkpoint": str(args.checkpoint or default_checkpoint),
        "masks_manifest": str(args.masks_manifest or masks_manifest),
        "mask_source": args.mask_source,
        "mask_flow_checkpoint": str(mask_flow_checkpoint) if args.mask_source == "descriptor_flow" else "",
        "flow_steps": args.flow_steps,
        "flow_sampler": args.flow_sampler,
        "flow_sigma": args.flow_sigma,
        "flow_max_delta": args.flow_max_delta,
        "mask_match_top_k": args.mask_match_top_k,
        "mask_match_max_score": args.mask_match_max_score,
        "allow_clamped_masks": args.allow_clamped_masks,
        "color_speckle_threshold": args.color_speckle_threshold,
        "teacher_checkpoint": str(args.teacher_checkpoint or output_root / args.teacher_stage_name / "checkpoints" / "latest.pt"),
        "max_samples": args.max_samples,
        "dry_run": True,
    }


def generate(args: Any) -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for plus synthetic generation.") from exc

    from ucdmr_flow_residual_plus.evaluation import _metrics as segmentation_metrics
    from ucdmr_flow_residual_plus.evaluation import _predict as predict_segmentation
    from ucdmr_flow_residual_plus.models.residual_flow import ResidualFlowUNet
    from ucdmr_flow_residual_plus.teacher import load_teacher_segmenter

    config = load_config(args.config)
    dataset_root = resolve_dataset_root(config, args.dataset_root)
    output_root = resolve_output_root(config, args.output_root)
    split_manifest = Path(args.split_manifest) if args.split_manifest else output_root / "data" / "manifest_splits.csv"
    default_masks = output_root / "sampled_masks" / "sampled_masks_manifest.csv" if args.mask_source == "descriptor_flow" else output_root / "masks" / "masks_manifest.csv"
    masks_manifest = Path(args.masks_manifest) if args.masks_manifest else default_masks
    latest_finite = output_root / "residual_flow_plus" / "checkpoints" / "latest_finite.pt"
    latest = output_root / "residual_flow_plus" / "checkpoints" / "latest.pt"
    default_checkpoint = latest_finite if latest_finite.exists() else latest
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint
    mask_flow_checkpoint = Path(args.mask_flow_checkpoint) if args.mask_flow_checkpoint else output_root / "mask_descriptor_flow" / "checkpoints" / "latest.pt"
    synthetic_root = Path(args.synthetic_output) if args.synthetic_output else output_root / "synthetic" / "raw"
    ensure_dir(synthetic_root)
    normals = [row for row in read_csv_records(split_manifest) if row.get("label") == "normal" and row.get("split", args.split) == args.split]
    masks = [
        row
        for row in read_csv_records(masks_manifest)
        if row.get("split") in {args.split, "generated"} and row.get("label", "crack") != "broken"
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
    bad_tensors = _checkpoint_model_nonfinite_tensors(torch, ckpt)
    if bad_tensors:
        raise SystemExit(
            "Residual flow checkpoint contains non-finite tensors. "
            f"Refusing generation. checkpoint={checkpoint_path}, bad_tensors={bad_tensors[:20]}"
        )
    ckpt_args = ckpt.get("args", {})
    style_dim = int(args.style_dim or ckpt_args.get("style_dim", 16))
    model = ResidualFlowUNet(
        condition_channels=CONDITION_CHANNELS,
        base_channels=int(args.base_channels or ckpt_args.get("base_channels", 48)),
        style_dim=style_dim,
        time_dim=int(args.time_dim or ckpt_args.get("time_dim", 128)),
        max_velocity=float(args.max_velocity if args.max_velocity is not None else ckpt_args.get("max_velocity", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    teacher = None
    teacher_checkpoint = Path(args.teacher_checkpoint) if args.teacher_checkpoint else output_root / args.teacher_stage_name / "checkpoints" / "latest.pt"
    if teacher_checkpoint.exists():
        teacher = load_teacher_segmenter(
            torch_module=torch,
            checkpoint_path=teacher_checkpoint,
            device=device,
            source_root=args.teacher_source_root,
        )
    seed_mask = int(args.seed_mask if args.seed_mask is not None else args.seed)
    seed_residual = int(args.seed_residual if args.seed_residual is not None else args.seed + 100000)
    records: list[dict[str, object]] = []
    skipped_no_candidates = 0
    skipped_clamped_candidates = 0
    skipped_match_score = 0
    with torch.no_grad():
        for idx, normal in enumerate(normals):
            domain = effective_domain(normal)
            base_domain = base_domain_name(domain)
            image = load_rgb01(dataset_path(dataset_root, normal["dataset_relative_path"]))
            height, width = image.shape[:2]
            candidates = [row for row in masks if effective_domain(row) == domain and _same_shape(row, width, height)]
            if not candidates:
                skipped_no_candidates += 1
                continue
            if not args.allow_clamped_masks:
                candidates = [row for row in candidates if row.get("placement_clamped", "0") != "1"]
            if not candidates:
                skipped_clamped_candidates += 1
                continue
            sample_seed_residual = seed_residual + idx * 1000003
            sample_seed_mask = seed_mask + idx * 1000003
            scored_candidates = sorted(
                ((*_normal_mask_match_score(normal, row), row) for row in candidates),
                key=lambda item: item[0],
            )
            if args.mask_match_max_score is not None:
                scored_candidates = [item for item in scored_candidates if item[0] <= float(args.mask_match_max_score)]
                if not scored_candidates:
                    skipped_match_score += 1
                    continue
            top_k = max(1, int(args.mask_match_top_k))
            match_score, match_reason, mask_row = random.Random(sample_seed_mask).choice(scored_candidates[:top_k])
            condition = _condition(mask_row)
            style_rng = random.Random(sample_seed_residual)
            style = np.asarray([style_rng.gauss(0.0, 1.0) for _ in range(style_dim)], dtype=np.float32)
            delta = _flow_delta(
                torch_module=torch,
                model=model,
                image=image,
                condition=condition,
                domain=domain,
                style=style,
                device=device,
                steps=args.flow_steps,
                sampler=args.flow_sampler,
                sigma=args.flow_sigma,
                max_delta=args.flow_max_delta,
                seed=sample_seed_residual,
            )
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
                prob = predict_segmentation(torch, teacher, synthetic, device=device)
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
            residual_rgb_path = synthetic_root / "residual_rgb_abs" / f"{synthetic_id}.png"
            _save_rgb(image_path, synthetic)
            save_mask(mask_path, raw)
            ensure_dir(residual_path.parent)
            applied_delta = gate[..., None] * delta
            Image.fromarray(np.clip(gray_from_rgb(np.abs(applied_delta) * 255.0), 0, 255).astype(np.uint8), mode="L").save(residual_path)
            ensure_dir(residual_rgb_path.parent)
            Image.fromarray(np.clip(np.abs(applied_delta) * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(residual_rgb_path)
            residual_metrics = _residual_quality_metrics(
                applied_delta=applied_delta,
                gate=gate,
                raw=raw,
                color_speckle_threshold=args.color_speckle_threshold,
            )
            records.append(
                {
                    "synthetic_id": synthetic_id,
                    "domain": domain,
                    "base_domain": base_domain,
                    "source_split": normal.get("split", ""),
                    "normal_source_path": normal["dataset_relative_path"],
                    "source_normal_path": normal["dataset_relative_path"],
                    "mask_source_id": mask_row.get("sample_id", ""),
                    "native_width": width,
                    "native_height": height,
                    "mask_source": args.mask_source,
                    "same_domain_only": "1",
                    "residual_source": "flow",
                    "residual_flow_checkpoint": str(checkpoint_path),
                    "mask_flow_checkpoint": str(mask_flow_checkpoint) if args.mask_source == "descriptor_flow" else "",
                    "flow_steps": args.flow_steps,
                    "flow_sampler": args.flow_sampler,
                    "flow_sigma": args.flow_sigma,
                    "flow_max_delta": args.flow_max_delta,
                    "seed_residual": sample_seed_residual,
                    "seed_mask": sample_seed_mask,
                    "mask_match_score": round(float(match_score), 8),
                    "mask_match_reason": match_reason,
                    "mask_placement_clamped": mask_row.get("placement_clamped", ""),
                    "source_template_split_key": mask_row.get("source_template_split_key", ""),
                    "source_template_video_id": mask_row.get("source_template_video_id", ""),
                    "source_template_crop_left": mask_row.get("source_template_crop_left", ""),
                    "source_template_crop_top": mask_row.get("source_template_crop_top", ""),
                    "source_template_crop_right": mask_row.get("source_template_crop_right", ""),
                    "source_template_crop_bottom": mask_row.get("source_template_crop_bottom", ""),
                    "m_raw_ratio": round(float(raw.mean()), 10),
                    "mask_area_ratio": round(float(raw.mean()), 10),
                    "outside_change": round(outside_change, 8),
                    "inside_change": round(inside_change, 8),
                    "residual_leak": round(outside_change / max(inside_change, 1e-8), 8),
                    "residual_leakage_score": round(outside_change / max(inside_change, 1e-8), 8),
                    "mask_residual_iou": round(mask_residual_iou, 8),
                    **residual_metrics,
                    "teacher_dice": teacher_dice,
                    "teacher_recall_on_mask": teacher_recall,
                    "teacher_fp_outside_mask": teacher_fp,
                    "quality_score": "",
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "synthetic_image_path": str(image_path),
                    "synthetic_mask_path": str(mask_path),
                    "residual_path": str(residual_path),
                    "residual_rgb_path": str(residual_rgb_path),
                    "generation_formula": "I_syn = I_normal + gate(M_syn) * Delta_flow",
                }
            )
    manifest_path = synthetic_root / "synthetic_manifest.csv"
    write_csv_records(manifest_path, records)
    summary = {
        "sample_count": len(records),
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path),
        "residual_source": "flow",
        "mean_residual_leak": round(float(np.mean([row["residual_leak"] for row in records])) if records else math.nan, 8),
        "mean_residual_chroma_p99": round(float(np.mean([row["residual_chroma_p99"] for row in records])) if records else math.nan, 8),
        "mean_color_speckle_ratio": round(float(np.mean([row["color_speckle_ratio"] for row in records])) if records else math.nan, 10),
        "skipped_no_candidates": skipped_no_candidates,
        "skipped_clamped_candidates": skipped_clamped_candidates,
        "skipped_match_score": skipped_match_score,
    }
    write_json(synthetic_root / "synthetic_generation_summary.json", summary)
    print(summary)


def dry_run_filter_summary(args: Any) -> dict[str, object]:
    config = load_config(args.config)
    output_root = resolve_output_root(config, args.output_root)
    return {
        "stage": "plus_filter_synthetic",
        "synthetic_manifest": str(args.synthetic_manifest or output_root / "synthetic" / "raw" / "synthetic_manifest.csv"),
        "domain_thresholds": str(args.domain_thresholds) if args.domain_thresholds else "config.filter.per_domain",
        "dry_run": True,
    }


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except ValueError:
        return default


def _domain_thresholds(config: dict[str, Any], path: Path | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    config_thresholds = nested_get(config, ("filter", "per_domain"), {})
    if isinstance(config_thresholds, dict):
        merged.update(config_thresholds)
    if path is not None:
        file_config = load_config(path)
        file_thresholds = nested_get(file_config, ("filter", "per_domain"), file_config.get("per_domain", file_config))
        if isinstance(file_thresholds, dict):
            merged.update(file_thresholds)
    return merged


def _threshold_for_domain(
    *,
    domain_thresholds: dict[str, Any],
    domain: str,
    key: str,
    default: float | None,
) -> float | None:
    values = domain_thresholds.get(domain, {})
    if not isinstance(values, dict) or key not in values:
        values = domain_thresholds.get(base_domain_name(domain), values)
    if isinstance(values, dict) and key in values and values[key] not in {"", None}:
        return float(values[key])
    return default


def filter_synthetic(args: Any) -> None:
    config = load_config(args.config)
    output_root = resolve_output_root(config, args.output_root)
    synthetic_manifest = Path(args.synthetic_manifest) if args.synthetic_manifest else output_root / "synthetic" / "raw" / "synthetic_manifest.csv"
    output_dir = Path(args.filtered_output) if args.filtered_output else output_root / "synthetic" / "filtered"
    rows = read_csv_records(synthetic_manifest)
    domain_thresholds = _domain_thresholds(config, args.domain_thresholds)
    scored = []
    kept = []
    for row in rows:
        domain = row.get("domain", "")
        max_residual_leak = _threshold_for_domain(domain_thresholds=domain_thresholds, domain=domain, key="max_residual_leak", default=args.max_residual_leak)
        max_outside_change = _threshold_for_domain(domain_thresholds=domain_thresholds, domain=domain, key="max_outside_change", default=args.max_outside_change)
        min_mask_area = _threshold_for_domain(domain_thresholds=domain_thresholds, domain=domain, key="min_mask_area", default=args.min_mask_area)
        max_mask_area = _threshold_for_domain(domain_thresholds=domain_thresholds, domain=domain, key="max_mask_area", default=args.max_mask_area)
        min_mask_residual_iou = _threshold_for_domain(domain_thresholds=domain_thresholds, domain=domain, key="min_mask_residual_iou", default=args.min_mask_residual_iou)
        min_teacher_dice = _threshold_for_domain(domain_thresholds=domain_thresholds, domain=domain, key="min_teacher_dice", default=args.min_teacher_dice)
        min_teacher_recall = _threshold_for_domain(domain_thresholds=domain_thresholds, domain=domain, key="min_teacher_recall", default=args.min_teacher_recall)
        max_residual_chroma_p99 = _threshold_for_domain(
            domain_thresholds=domain_thresholds,
            domain=domain,
            key="max_residual_chroma_p99",
            default=args.max_residual_chroma_p99,
        )
        max_color_speckle_ratio = _threshold_for_domain(
            domain_thresholds=domain_thresholds,
            domain=domain,
            key="max_color_speckle_ratio",
            default=args.max_color_speckle_ratio,
        )
        max_color_speckle_pixels = _threshold_for_domain(
            domain_thresholds=domain_thresholds,
            domain=domain,
            key="max_color_speckle_pixels",
            default=args.max_color_speckle_pixels,
        )
        max_residual_highfreq_ratio = _threshold_for_domain(
            domain_thresholds=domain_thresholds,
            domain=domain,
            key="max_residual_highfreq_ratio",
            default=args.max_residual_highfreq_ratio,
        )
        leak = _float(row.get("residual_leak"), math.inf)
        outside = _float(row.get("outside_change"), math.inf)
        area = _float(row.get("mask_area_ratio"), 0.0)
        iou = _float(row.get("mask_residual_iou"), 0.0)
        teacher = _float(row.get("teacher_dice"), math.nan)
        teacher_recall = _float(row.get("teacher_recall_on_mask"), math.nan)
        chroma_p99 = _float(row.get("residual_chroma_p99"), 0.0)
        color_speckle_ratio = _float(row.get("color_speckle_ratio"), 0.0)
        color_speckle_pixels = _float(row.get("color_speckle_pixels"), 0.0)
        highfreq_ratio = _float(row.get("residual_highfreq_ratio"), 0.0)
        topology = _float(row.get("topology_score"), math.nan)
        keep = (
            leak <= float(max_residual_leak)
            and outside <= float(max_outside_change)
            and float(min_mask_area) <= area <= float(max_mask_area)
            and iou >= float(min_mask_residual_iou)
        )
        if min_teacher_dice is not None and not math.isnan(teacher):
            keep = keep and teacher >= float(min_teacher_dice)
        if min_teacher_recall is not None and not math.isnan(teacher_recall):
            keep = keep and teacher_recall >= float(min_teacher_recall)
        if max_residual_chroma_p99 is not None:
            keep = keep and chroma_p99 <= float(max_residual_chroma_p99)
        if max_color_speckle_ratio is not None:
            keep = keep and color_speckle_ratio <= float(max_color_speckle_ratio)
        if max_color_speckle_pixels is not None:
            keep = keep and color_speckle_pixels <= float(max_color_speckle_pixels)
        if max_residual_highfreq_ratio is not None:
            keep = keep and highfreq_ratio <= float(max_residual_highfreq_ratio)
        score = 1.0 - min(leak / max(float(max_residual_leak), 1e-8), 2.0) * 0.3 - min(outside / max(float(max_outside_change), 1e-8), 2.0) * 0.3 + min(iou, 1.0) * 0.4
        if max_residual_chroma_p99 is not None:
            score -= min(chroma_p99 / max(float(max_residual_chroma_p99), 1e-8), 2.0) * 0.1
        if max_color_speckle_ratio is not None:
            score -= min(color_speckle_ratio / max(float(max_color_speckle_ratio), 1e-8), 2.0) * 0.1
        if max_color_speckle_pixels is not None:
            score -= min(color_speckle_pixels / max(float(max_color_speckle_pixels), 1.0), 2.0) * 0.1
        if max_residual_highfreq_ratio is not None:
            score -= min(highfreq_ratio / max(float(max_residual_highfreq_ratio), 1e-8), 2.0) * 0.1
        if not math.isnan(teacher):
            score += min(max(teacher, 0.0), 1.0) * 0.1
        if not math.isnan(teacher_recall):
            score += min(max(teacher_recall, 0.0), 1.0) * 0.1
        if not math.isnan(topology):
            score += min(max(topology, 0.0), 1.0) * 0.1
        record = {
            **row,
            "filter_keep": "1" if keep else "0",
            "quality_score": round(float(score), 8),
            "filter_reason": "kept" if keep else "threshold_failed",
            "filter_threshold_domain": domain,
            "threshold_max_residual_leak": max_residual_leak,
            "threshold_max_outside_change": max_outside_change,
            "threshold_min_mask_area": min_mask_area,
            "threshold_max_mask_area": max_mask_area,
            "threshold_min_mask_residual_iou": min_mask_residual_iou,
            "threshold_min_teacher_dice": "" if min_teacher_dice is None else min_teacher_dice,
            "threshold_min_teacher_recall": "" if min_teacher_recall is None else min_teacher_recall,
            "threshold_max_residual_chroma_p99": "" if max_residual_chroma_p99 is None else max_residual_chroma_p99,
            "threshold_max_color_speckle_ratio": "" if max_color_speckle_ratio is None else max_color_speckle_ratio,
            "threshold_max_color_speckle_pixels": "" if max_color_speckle_pixels is None else max_color_speckle_pixels,
            "threshold_max_residual_highfreq_ratio": "" if max_residual_highfreq_ratio is None else max_residual_highfreq_ratio,
        }
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
