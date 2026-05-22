from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
from typing import Any
import math
import random

import numpy as np
from PIL import Image

from ucdmr_flow_residual_plus.image_utils import bool_mask, save_mask
from ucdmr_flow_residual_plus.io_utils import ensure_dir, read_csv_records, write_csv_records, write_json

from ucdmr_flow_residual_plus.config import domain_value, load_config, resolve_output_root
from ucdmr_flow_residual_plus.constants import (
    MASK_DOMAIN_TO_INDEX,
    base_domain_name,
    materialize_domain_fields,
    mask_domain_from_row,
    resolve_existing,
)
from ucdmr_flow_residual_plus.mask_representations import build_plus_regions, orientation


LEGACY_DESCRIPTOR_FIELDS = [
    "center_x",
    "center_y",
    "m_raw_ratio",
    "bbox_w",
    "bbox_h",
    "main_orientation_norm",
    "skeleton_ratio",
    "thickness_mean",
    "component_count_norm",
]
DESCRIPTOR_FIELDS = [
    "center_x",
    "center_y",
    "m_raw_ratio",
    "bbox_w",
    "bbox_h",
    "orientation_sin2_norm",
    "orientation_cos2_norm",
    "skeleton_ratio",
    "thickness_mean",
    "component_count_norm",
]
DESCRIPTOR_FIELD_WEIGHTS = {
    "center_x": 1.5,
    "center_y": 1.5,
    "m_raw_ratio": 2.0,
    "bbox_w": 1.0,
    "bbox_h": 1.0,
    "main_orientation_norm": 0.75,
    "orientation_sin2_norm": 0.4,
    "orientation_cos2_norm": 0.4,
    "skeleton_ratio": 0.5,
    "thickness_mean": 0.5,
    "component_count_norm": 0.25,
}
TEMPLATE_METADATA_FIELDS = [
    "dataset_relative_path",
    "video_id",
    "video_name",
    "split_key",
    "crop_left",
    "crop_top",
    "crop_right",
    "crop_bottom",
    "crop_width",
    "crop_height",
    "crop_match_strategy",
    "dphone_id",
    "effective_domain",
    "base_domain",
    "residual_domain",
    "center_x",
    "center_y",
    "bbox_w",
    "bbox_h",
    "main_orientation",
]


def _wrap_axial_angle(angle: float) -> float:
    while angle < -math.pi / 2:
        angle += math.pi
    while angle > math.pi / 2:
        angle -= math.pi
    return float(angle)


def _angle_to_norm(angle: float) -> float:
    return float((_wrap_axial_angle(angle) + math.pi / 2) / math.pi)


def _angle_from_norm(value: float) -> float:
    return _wrap_axial_angle(float(value) * math.pi - math.pi / 2)


def _angle_to_axial_pair(angle: float) -> tuple[float, float]:
    angle = _wrap_axial_angle(angle)
    return float((math.sin(2.0 * angle) + 1.0) * 0.5), float((math.cos(2.0 * angle) + 1.0) * 0.5)


def _angle_from_descriptor(desc: np.ndarray, fields: Sequence[str]) -> float:
    if "orientation_sin2_norm" in fields and "orientation_cos2_norm" in fields:
        sin2 = float(desc[fields.index("orientation_sin2_norm")]) * 2.0 - 1.0
        cos2 = float(desc[fields.index("orientation_cos2_norm")]) * 2.0 - 1.0
        if abs(sin2) < 1e-8 and abs(cos2) < 1e-8:
            return 0.0
        return _wrap_axial_angle(0.5 * math.atan2(sin2, cos2))
    if "main_orientation_norm" in fields:
        return _angle_from_norm(float(desc[fields.index("main_orientation_norm")]))
    return 0.0


def _normalize_descriptor_orientation(desc: np.ndarray, fields: Sequence[str]) -> np.ndarray:
    out = desc.copy()
    if "orientation_sin2_norm" not in fields or "orientation_cos2_norm" not in fields:
        return out
    sin_idx = fields.index("orientation_sin2_norm")
    cos_idx = fields.index("orientation_cos2_norm")
    sin2 = float(out[sin_idx]) * 2.0 - 1.0
    cos2 = float(out[cos_idx]) * 2.0 - 1.0
    norm = math.hypot(sin2, cos2)
    if norm < 1e-8:
        sin2, cos2 = 0.0, 1.0
    else:
        sin2, cos2 = sin2 / norm, cos2 / norm
    out[sin_idx] = (sin2 + 1.0) * 0.5
    out[cos_idx] = (cos2 + 1.0) * 0.5
    return out


def _descriptor_field_value(row: dict[str, str], field: str) -> float:
    angle = float(row.get("main_orientation", 0.0) or 0.0)
    component_count = float(row.get("component_count", 1.0) or 1.0)
    if field == "center_x":
        return float(row.get("center_x", 0.5) or 0.5)
    if field == "center_y":
        return float(row.get("center_y", 0.5) or 0.5)
    if field == "m_raw_ratio":
        return float(row.get("m_raw_ratio", row.get("mask_area_ratio", 0.0)) or 0.0)
    if field == "bbox_w":
        return float(row.get("bbox_w", 0.01) or 0.01)
    if field == "bbox_h":
        return float(row.get("bbox_h", 0.01) or 0.01)
    if field == "main_orientation_norm":
        return _angle_to_norm(angle)
    if field == "orientation_sin2_norm":
        return _angle_to_axial_pair(angle)[0]
    if field == "orientation_cos2_norm":
        return _angle_to_axial_pair(angle)[1]
    if field == "skeleton_ratio":
        return float(row.get("skeleton_ratio", 0.0) or 0.0)
    if field == "thickness_mean":
        return float(row.get("thickness_mean", 0.0) or 0.0)
    if field == "component_count_norm":
        return min(component_count, 10.0) / 10.0
    raise KeyError(f"Unknown descriptor field: {field}")


def descriptor_from_row(row: dict[str, str], fields: Sequence[str] | None = None) -> np.ndarray:
    active_fields = list(fields or DESCRIPTOR_FIELDS)
    return np.asarray([_descriptor_field_value(row, field) for field in active_fields], dtype=np.float32)


def dry_run_train_summary(args: Any) -> dict[str, object]:
    config = load_config(args.config)
    output_root = resolve_output_root(config, args.output_root)
    return {
        "stage": "plus_mask_descriptor_flow",
        "masks_manifest": str(args.masks_manifest or output_root / "masks" / "masks_manifest.csv"),
        "output_root": str(output_root),
        "epochs": args.epochs,
        "dry_run": True,
    }


def train(args: Any) -> None:
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for mask descriptor flow training.") from exc

    from ucdmr_flow_residual_plus.models.descriptor_flow import DescriptorFlowMLP

    config = load_config(args.config)
    output_root = resolve_output_root(config, args.output_root)
    masks_manifest = Path(args.masks_manifest) if args.masks_manifest else output_root / "masks" / "masks_manifest.csv"
    rows = []
    for source_row in read_csv_records(masks_manifest):
        if source_row.get("split", args.split) != args.split:
            continue
        row = materialize_domain_fields(source_row)
        domain = str(row["effective_domain"])
        rows.append(row)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        raise SystemExit(f"No mask rows for split={args.split}: {masks_manifest}")
    descriptor_fields = list(DESCRIPTOR_FIELDS)
    x1 = np.stack([descriptor_from_row(row, descriptor_fields) for row in rows], axis=0)
    domain_to_index = dict(MASK_DOMAIN_TO_INDEX)
    condition_dim = len(domain_to_index)
    domain_idx = np.asarray([domain_to_index[row["domain"]] for row in rows], dtype=np.int64)
    dataset = TensorDataset(torch.from_numpy(x1), torch.from_numpy(domain_idx))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DescriptorFlowMLP(
        descriptor_dim=len(descriptor_fields),
        condition_dim=condition_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    for epoch in range(args.epochs):
        total = 0.0
        count = 0
        for desc, domain in loader:
            desc = desc.to(device=device, dtype=torch.float32)
            domain = domain.to(device=device)
            cond = torch.nn.functional.one_hot(domain, num_classes=condition_dim).float()
            x0 = torch.rand_like(desc)
            t = torch.rand((desc.shape[0], 1), device=device)
            x_t = (1.0 - t) * x0 + t * desc
            target_v = desc - x0
            pred_v = model(x_t, t, cond)
            loss = torch.nn.functional.mse_loss(pred_v, target_v)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
            count += 1
        history.append({"epoch": epoch, "loss": round(total / max(count, 1), 8)})
    train_root = Path(args.train_output) if args.train_output else output_root / "mask_descriptor_flow"
    ckpt_root = train_root / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    write_csv_records(train_root / "template_manifest.csv", rows)
    ckpt = {
        "model": model.state_dict(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "descriptor_fields": descriptor_fields,
        "domain_to_index": domain_to_index,
        "condition_dim": condition_dim,
        "template_manifest": str(train_root / "template_manifest.csv"),
        "history": history,
    }
    torch.save(ckpt, ckpt_root / "latest.pt")
    write_json(output_root / "reports" / "plus_mask_descriptor_flow_training.json", ckpt | {"model": "<state_dict>"})
    print({"checkpoint": str(ckpt_root / "latest.pt"), "rows": len(rows), "last_loss": history[-1]["loss"]})


def dry_run_sample_summary(args: Any) -> dict[str, object]:
    config = load_config(args.config)
    output_root = resolve_output_root(config, args.output_root)
    return {
        "stage": "plus_sample_masks",
        "checkpoint": str(args.checkpoint or output_root / "mask_descriptor_flow" / "checkpoints" / "latest.pt"),
        "sample_count": args.sample_count,
        "samples_per_domain": args.samples_per_domain,
        "workers": args.workers,
        "max_render_center_error": args.max_render_center_error,
        "max_render_angle_error": args.max_render_angle_error,
        "reject_clamped": args.reject_clamped,
        "keep_rejected": args.keep_rejected,
        "mask_region_radii": "config_per_domain" if args.inpaint_radius is None or args.band_radius is None or args.gate_radius is None or args.gate_blur is None else "cli_override",
        "dry_run": True,
    }


def _sample_descriptor(
    torch_module: Any,
    model: Any,
    *,
    descriptor_fields: Sequence[str],
    domain_idx: int,
    condition_dim: int,
    steps: int,
    device: Any,
) -> np.ndarray:
    x = torch_module.rand((1, len(descriptor_fields)), device=device)
    condition = torch_module.nn.functional.one_hot(
        torch_module.tensor([domain_idx], device=device),
        num_classes=condition_dim,
    ).float()
    dt = 1.0 / max(steps, 1)
    with torch_module.no_grad():
        for step in range(steps):
            t = torch_module.full((1, 1), step / max(steps, 1), device=device)
            x = (x + model(x, t, condition) * dt).clamp(0.0, 1.0)
    desc = x.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return _normalize_descriptor_orientation(desc, descriptor_fields)


def _descriptor_match_score(row: dict[str, str], desc: np.ndarray, fields: Sequence[str]) -> float:
    ref = descriptor_from_row(row, fields)
    diff = np.abs(ref - desc)
    if "main_orientation_norm" in fields:
        idx = fields.index("main_orientation_norm")
        diff[idx] = min(float(diff[idx]), 1.0 - float(diff[idx]))
    return float(sum(float(value) * DESCRIPTOR_FIELD_WEIGHTS.get(field, 1.0) for field, value in zip(fields, diff)))


def _template(rows: list[dict[str, str]], desc: np.ndarray, rng: random.Random, *, fields: Sequence[str]) -> dict[str, str]:
    scored = []
    for row in rows:
        score = _descriptor_match_score(row, desc, fields)
        scored.append((score, row))
    scored.sort(key=lambda item: item[0])
    return rng.choice(scored[: max(1, min(5, len(scored)))])[1]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _render(
    template_row: dict[str, str],
    desc: np.ndarray,
    *,
    width: int,
    height: int,
    fields: Sequence[str] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    active_fields = list(fields or DESCRIPTOR_FIELDS)
    src = bool_mask(Image.open(resolve_existing(template_row["m_raw_path"])))
    box = _bbox(src)
    if box is None:
        return np.zeros((height, width), dtype=bool), {"placement_clamped": "0", "rendered_center_x": "", "rendered_center_y": ""}
    x0, y0, x1, y1 = box
    crop = Image.fromarray(src[y0:y1, x0:x1].astype(np.uint8) * 255, mode="L")
    target_w = max(2, min(width, int(round(float(desc[active_fields.index("bbox_w")]) * width))))
    target_h = max(2, min(height, int(round(float(desc[active_fields.index("bbox_h")]) * height))))
    crop = crop.resize((target_w, target_h), resample=Image.Resampling.NEAREST)
    template_angle = orientation(np.asarray(crop, dtype=np.uint8) > 0)
    target_angle = _angle_from_descriptor(desc, active_fields)
    pil_angle = math.degrees(_wrap_axial_angle(template_angle - target_angle))
    crop = crop.rotate(pil_angle, resample=Image.Resampling.NEAREST, expand=True)
    canvas = Image.new("L", (width, height), 0)
    cx = int(round(float(desc[active_fields.index("center_x")]) * width))
    cy = int(round(float(desc[active_fields.index("center_y")]) * height))
    requested_left = cx - crop.width // 2
    requested_top = cy - crop.height // 2
    left = max(0, min(width - crop.width, requested_left))
    top = max(0, min(height - crop.height, requested_top))
    canvas.paste(crop, (left, top), crop)
    raw = bool_mask(canvas)
    rendered_box = _bbox(raw)
    rendered_center_x: str | float = ""
    rendered_center_y: str | float = ""
    rendered_orientation: str | float = ""
    orientation_error_degrees: str | float = ""
    if rendered_box is not None:
        ys, xs = np.where(raw)
        rendered_center_x = round(float((xs.mean() + 0.5) / width), 8)
        rendered_center_y = round(float((ys.mean() + 0.5) / height), 8)
        rendered_orientation_value = orientation(raw)
        rendered_orientation = round(rendered_orientation_value, 8)
        orientation_error_degrees = round(math.degrees(_wrap_axial_angle(rendered_orientation_value - target_angle)), 6)
    placement = {
        "requested_center_x": round(float(desc[active_fields.index("center_x")]), 8),
        "requested_center_y": round(float(desc[active_fields.index("center_y")]), 8),
        "rendered_center_x": rendered_center_x,
        "rendered_center_y": rendered_center_y,
        "target_main_orientation": round(target_angle, 8),
        "template_main_orientation": round(template_angle, 8),
        "rendered_main_orientation": rendered_orientation,
        "render_rotation_degrees": round(pil_angle, 6),
        "render_orientation_error_degrees": orientation_error_degrees,
        "placement_clamped": "1" if left != requested_left or top != requested_top else "0",
        "render_crop_width": crop.width,
        "render_crop_height": crop.height,
        "requested_left": requested_left,
        "requested_top": requested_top,
        "render_left": left,
        "render_top": top,
    }
    return raw, placement


def _region_value(config: dict[str, Any], args_value: int | float | None, key: str, domain: str, default: int | float) -> int | float:
    if args_value is not None:
        return args_value
    return domain_value(config, "masks", key, base_domain_name(domain), default)


def _render_sample_task(task: dict[str, Any]) -> dict[str, object]:
    idx = int(task["idx"])
    domain = str(task["domain"])
    desc = np.asarray(task["desc"], dtype=np.float32)
    descriptor_fields = list(task.get("descriptor_fields") or DESCRIPTOR_FIELDS)
    template = dict(task["template"])
    sample_root = Path(task["sample_root"])
    width = int(template.get("mask_width", template.get("image_width", 0)))
    height = int(template.get("mask_height", template.get("image_height", 0)))
    raw, placement = _render(template, desc, width=width, height=height, fields=descriptor_fields)
    raw_ratio = float(raw.mean())
    center_error: str | float = ""
    if placement.get("rendered_center_x") != "" and placement.get("rendered_center_y") != "":
        dx = float(placement["rendered_center_x"]) - float(placement["requested_center_x"])
        dy = float(placement["rendered_center_y"]) - float(placement["requested_center_y"])
        center_error = round(float((dx * dx + dy * dy) ** 0.5), 8)
    angle_error_abs: str | float = ""
    if placement.get("render_orientation_error_degrees") != "":
        angle_error_abs = round(abs(float(placement["render_orientation_error_degrees"])), 6)
    reject_reasons: list[str] = []
    if raw_ratio <= 0.0:
        reject_reasons.append("empty_mask")
    if raw_ratio < float(task["min_area_ratio"]):
        reject_reasons.append("area_too_small")
    if raw_ratio > float(task["max_area_ratio"]):
        reject_reasons.append("area_too_large")
    if bool(task["reject_clamped"]) and placement.get("placement_clamped") == "1":
        reject_reasons.append("placement_clamped")
    if center_error != "" and float(center_error) > float(task["max_render_center_error"]):
        reject_reasons.append("center_error")
    if angle_error_abs != "" and float(angle_error_abs) > float(task["max_render_angle_error"]):
        reject_reasons.append("orientation_error")
    render_accept = "0" if reject_reasons else "1"
    sample_id = f"plus_mask_{idx:06d}_{domain}_{template.get('sample_id', Path(template['m_raw_path']).stem)}"
    paths: dict[str, str] = {}
    if render_accept == "1" or bool(task["keep_rejected"]):
        regions = build_plus_regions(
            raw,
            inpaint_radius=int(task["inpaint_radius"]),
            band_radius=int(task["band_radius"]),
            gate_radius=int(task["gate_radius"]),
            gate_blur=float(task["gate_blur"]),
        )
        for name, mask in regions.items():
            path = sample_root / name / f"{sample_id}.png"
            ensure_dir(path.parent)
            if mask.dtype == bool:
                save_mask(path, mask)
            else:
                Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)
            paths[f"{name}_path"] = str(path)
    target_angle = _angle_from_descriptor(desc, descriptor_fields)
    descriptor_record = {field: round(float(value), 8) for field, value in zip(descriptor_fields, desc)}
    descriptor_record["main_orientation_norm"] = round(_angle_to_norm(target_angle), 8)
    return {
        "idx": idx,
        "sample_id": sample_id,
        "domain": domain,
        "dphone_id": template.get("dphone_id", ""),
        "base_domain": base_domain_name(domain),
        "residual_domain": base_domain_name(domain),
        "effective_domain": domain,
        "split": "generated",
        "label": "crack",
        "mask_width": width,
        "mask_height": height,
        "inpaint_radius": int(task["inpaint_radius"]),
        "band_radius": int(task["band_radius"]),
        "gate_radius": int(task["gate_radius"]),
        "gate_blur": float(task["gate_blur"]),
        "m_raw_ratio": round(raw_ratio, 10),
        "source_template_id": template.get("sample_id", ""),
        "source_template_domain": template.get("domain", ""),
        **{f"source_template_{key}": template.get(key, "") for key in TEMPLATE_METADATA_FIELDS},
        **descriptor_record,
        "main_orientation": round(orientation(raw), 8),
        "component_count": "",
        "placement_score": "",
        "render_center_error": center_error,
        "render_orientation_error_abs_degrees": angle_error_abs,
        "render_accept": render_accept,
        "render_reject_reason": ";".join(reject_reasons) if reject_reasons else "accepted",
        **placement,
        **paths,
    }


def sample(args: Any) -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for mask descriptor flow sampling.") from exc

    from ucdmr_flow_residual_plus.models.descriptor_flow import DescriptorFlowMLP

    config = load_config(args.config)
    output_root = resolve_output_root(config, args.output_root)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_root / "mask_descriptor_flow" / "checkpoints" / "latest.pt"
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    descriptor_fields = list(ckpt.get("descriptor_fields") or DESCRIPTOR_FIELDS)
    domain_to_index = ckpt.get("domain_to_index") or dict(MASK_DOMAIN_TO_INDEX)
    condition_dim = int(ckpt.get("condition_dim", len(domain_to_index)))
    model = DescriptorFlowMLP(
        descriptor_dim=len(descriptor_fields),
        condition_dim=condition_dim,
        hidden_dim=int(args.hidden_dim or ckpt_args.get("hidden_dim", 128)),
        depth=int(args.depth or ckpt_args.get("depth", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    template_manifest = Path(args.template_manifest) if args.template_manifest else Path(ckpt.get("template_manifest", output_root / "mask_descriptor_flow" / "template_manifest.csv"))
    templates = read_csv_records(template_manifest)
    by_domain: dict[str, list[dict[str, str]]] = {}
    for row in templates:
        domain = mask_domain_from_row(row)
        if domain in domain_to_index:
            by_domain.setdefault(domain, []).append(row)
    rng = random.Random(args.seed)
    sample_root = Path(args.sample_output) if args.sample_output else output_root / "sampled_masks"
    ensure_dir(sample_root)
    records = []
    domains = sorted(by_domain)
    if args.domain is not None and args.domain not in by_domain:
        raise SystemExit(f"No template masks available for domain={args.domain!r}. Available domains: {domains}")
    plan: list[str] = []
    if args.samples_per_domain is not None:
        active_domains = [args.domain] if args.domain is not None else domains
        for domain in active_domains:
            plan.extend([domain] * int(args.samples_per_domain))
    else:
        for _ in range(args.sample_count):
            plan.append(args.domain or rng.choice(domains))
    tasks: list[dict[str, Any]] = []
    for idx, domain in enumerate(plan):
        rows = by_domain[domain]
        domain_idx = domain_to_index[domain]
        desc = _sample_descriptor(
            torch,
            model,
            descriptor_fields=descriptor_fields,
            domain_idx=domain_idx,
            condition_dim=condition_dim,
            steps=args.steps,
            device=device,
        )
        desc[2] = float(np.clip(desc[2], args.min_area_ratio, args.max_area_ratio))
        template = _template(rows, desc, rng, fields=descriptor_fields)
        tasks.append(
            {
                "idx": idx,
                "domain": domain,
                "desc": desc,
                "descriptor_fields": descriptor_fields,
                "template": template,
                "sample_root": str(sample_root),
                "inpaint_radius": int(_region_value(config, args.inpaint_radius, "inpaint_radius", domain, 9)),
                "band_radius": int(_region_value(config, args.band_radius, "band_radius", domain, 5)),
                "gate_radius": int(_region_value(config, args.gate_radius, "gate_radius", domain, 7)),
                "gate_blur": float(_region_value(config, args.gate_blur, "gate_blur", domain, 3.0)),
                "min_area_ratio": float(args.min_area_ratio),
                "max_area_ratio": float(args.max_area_ratio),
                "max_render_center_error": float(args.max_render_center_error),
                "max_render_angle_error": float(args.max_render_angle_error),
                "reject_clamped": bool(args.reject_clamped),
                "keep_rejected": bool(args.keep_rejected),
            }
        )
    workers = max(1, int(args.workers))
    progress_every = max(0, int(args.progress_every))
    if workers == 1:
        for task in tasks:
            records.append(_render_sample_task(task))
            if progress_every and len(records) % progress_every == 0:
                print({"stage": "render_masks", "done": len(records), "total": len(tasks)}, flush=True)
    else:
        context = mp.get_context(args.worker_start_method)
        completed = 0
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            futures = [executor.submit(_render_sample_task, task) for task in tasks]
            for future in as_completed(futures):
                records.append(future.result())
                completed += 1
                if progress_every and completed % progress_every == 0:
                    print({"stage": "render_masks", "done": completed, "total": len(tasks), "workers": workers}, flush=True)
    records.sort(key=lambda record: int(record.get("idx", 0)))
    for record in records:
        record.pop("idx", None)
    all_records = list(records)
    records = [record for record in all_records if record.get("render_accept") == "1"]
    manifest_path = sample_root / "sampled_masks_manifest.csv"
    write_csv_records(manifest_path, records)
    all_manifest_path = sample_root / "sampled_masks_manifest_all.csv"
    if args.keep_rejected:
        write_csv_records(all_manifest_path, all_records)
    summary = {
        "sample_count": len(records),
        "attempted_sample_count": len(all_records),
        "rejected_sample_count": len(all_records) - len(records),
        "clamped_placement_count": sum(1 for record in records if record.get("placement_clamped") == "1"),
        "all_clamped_placement_count": sum(1 for record in all_records if record.get("placement_clamped") == "1"),
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "all_manifest": str(all_manifest_path) if args.keep_rejected else "",
        "workers": workers,
    }
    write_json(sample_root / "sampled_masks_summary.json", summary)
    print(summary)
