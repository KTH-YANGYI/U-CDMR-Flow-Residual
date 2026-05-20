from __future__ import annotations

from pathlib import Path
from typing import Any
import random

import numpy as np
from PIL import Image

from ucdmr_flow_residual_plus.image_utils import bool_mask, save_mask
from ucdmr_flow_residual_plus.io_utils import ensure_dir, read_csv_records, write_csv_records, write_json

from ucdmr_flow_residual_plus.config import load_config, resolve_output_root
from ucdmr_flow_residual_plus.constants import DOMAIN_TO_INDEX, resolve_existing
from ucdmr_flow_residual_plus.mask_representations import build_plus_regions, orientation


DESCRIPTOR_FIELDS = [
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


def descriptor_from_row(row: dict[str, str]) -> np.ndarray:
    angle = float(row.get("main_orientation", 0.0) or 0.0)
    component_count = float(row.get("component_count", 1.0) or 1.0)
    return np.asarray(
        [
            float(row.get("center_x", 0.5) or 0.5),
            float(row.get("center_y", 0.5) or 0.5),
            float(row.get("m_raw_ratio", row.get("mask_area_ratio", 0.0)) or 0.0),
            float(row.get("bbox_w", 0.01) or 0.01),
            float(row.get("bbox_h", 0.01) or 0.01),
            float((angle + np.pi / 2) / np.pi),
            float(row.get("skeleton_ratio", 0.0) or 0.0),
            float(row.get("thickness_mean", 0.0) or 0.0),
            min(component_count, 10.0) / 10.0,
        ],
        dtype=np.float32,
    )


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
    rows = [row for row in read_csv_records(masks_manifest) if row.get("split", args.split) == args.split]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        raise SystemExit(f"No mask rows for split={args.split}: {masks_manifest}")
    x1 = np.stack([descriptor_from_row(row) for row in rows], axis=0)
    domain_idx = np.asarray([DOMAIN_TO_INDEX.get(row.get("domain", row.get("dataset_group", "")), 0) for row in rows], dtype=np.int64)
    dataset = TensorDataset(torch.from_numpy(x1), torch.from_numpy(domain_idx))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DescriptorFlowMLP(descriptor_dim=len(DESCRIPTOR_FIELDS), condition_dim=3, hidden_dim=args.hidden_dim, depth=args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    for epoch in range(args.epochs):
        total = 0.0
        count = 0
        for desc, domain in loader:
            desc = desc.to(device=device, dtype=torch.float32)
            domain = domain.to(device=device)
            cond = torch.nn.functional.one_hot(domain, num_classes=3).float()
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
        "descriptor_fields": DESCRIPTOR_FIELDS,
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
        "dry_run": True,
    }


def _sample_descriptor(torch_module: Any, model: Any, *, domain_idx: int, steps: int, device: Any) -> np.ndarray:
    x = torch_module.rand((1, len(DESCRIPTOR_FIELDS)), device=device)
    condition = torch_module.nn.functional.one_hot(torch_module.tensor([domain_idx], device=device), num_classes=3).float()
    dt = 1.0 / max(steps, 1)
    with torch_module.no_grad():
        for step in range(steps):
            t = torch_module.full((1, 1), step / max(steps, 1), device=device)
            x = (x + model(x, t, condition) * dt).clamp(0.0, 1.0)
    return x.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _template(rows: list[dict[str, str]], desc: np.ndarray, rng: random.Random) -> dict[str, str]:
    scored = []
    for row in rows:
        ref = descriptor_from_row(row)
        score = float(np.abs(ref[2:8] - desc[2:8]).sum())
        scored.append((score, row))
    scored.sort(key=lambda item: item[0])
    return rng.choice(scored[: max(1, min(5, len(scored)))])[1]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _render(template_row: dict[str, str], desc: np.ndarray, *, width: int, height: int) -> np.ndarray:
    src = bool_mask(Image.open(resolve_existing(template_row["m_raw_path"])))
    box = _bbox(src)
    if box is None:
        return np.zeros((height, width), dtype=bool)
    x0, y0, x1, y1 = box
    crop = Image.fromarray(src[y0:y1, x0:x1].astype(np.uint8) * 255, mode="L")
    target_w = max(2, min(width, int(round(float(desc[3]) * width))))
    target_h = max(2, min(height, int(round(float(desc[4]) * height))))
    crop = crop.resize((target_w, target_h), resample=Image.Resampling.NEAREST)
    angle = float(desc[5]) * 180.0 - 90.0
    crop = crop.rotate(angle, resample=Image.Resampling.NEAREST, expand=True)
    canvas = Image.new("L", (width, height), 0)
    cx = int(round(float(desc[0]) * width))
    cy = int(round(float(desc[1]) * height))
    left = max(0, min(width - crop.width, cx - crop.width // 2))
    top = max(0, min(height - crop.height, cy - crop.height // 2))
    canvas.paste(crop, (left, top), crop)
    return bool_mask(canvas)


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
    model = DescriptorFlowMLP(
        descriptor_dim=len(DESCRIPTOR_FIELDS),
        condition_dim=3,
        hidden_dim=int(args.hidden_dim or ckpt_args.get("hidden_dim", 128)),
        depth=int(args.depth or ckpt_args.get("depth", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    template_manifest = Path(args.template_manifest) if args.template_manifest else Path(ckpt.get("template_manifest", output_root / "mask_descriptor_flow" / "template_manifest.csv"))
    templates = read_csv_records(template_manifest)
    by_domain: dict[str, list[dict[str, str]]] = {}
    for row in templates:
        by_domain.setdefault(row.get("domain", row.get("dataset_group", "")), []).append(row)
    rng = random.Random(args.seed)
    sample_root = Path(args.sample_output) if args.sample_output else output_root / "sampled_masks"
    ensure_dir(sample_root)
    records = []
    domains = sorted(by_domain)
    for idx in range(args.sample_count):
        domain = args.domain or rng.choice(domains)
        rows = by_domain[domain]
        domain_idx = DOMAIN_TO_INDEX.get(domain, 0)
        desc = _sample_descriptor(torch, model, domain_idx=domain_idx, steps=args.steps, device=device)
        desc[2] = float(np.clip(desc[2], args.min_area_ratio, args.max_area_ratio))
        template = _template(rows, desc, rng)
        width = int(template.get("mask_width", template.get("image_width", 0)))
        height = int(template.get("mask_height", template.get("image_height", 0)))
        raw = _render(template, desc, width=width, height=height)
        regions = build_plus_regions(raw, inpaint_radius=args.inpaint_radius, band_radius=args.band_radius, gate_radius=args.gate_radius, gate_blur=args.gate_blur)
        sample_id = f"plus_mask_{idx:06d}_{domain}_{template.get('sample_id', Path(template['m_raw_path']).stem)}"
        paths: dict[str, str] = {}
        for name, mask in regions.items():
            path = sample_root / name / f"{sample_id}.png"
            ensure_dir(path.parent)
            if mask.dtype == bool:
                save_mask(path, mask)
            else:
                Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)
            paths[f"{name}_path"] = str(path)
        records.append(
            {
                "sample_id": sample_id,
                "domain": domain,
                "split": "generated",
                "label": "crack",
                "mask_width": width,
                "mask_height": height,
                "m_raw_ratio": round(float(raw.mean()), 10),
                "source_template_id": template.get("sample_id", ""),
                **{field: round(float(value), 8) for field, value in zip(DESCRIPTOR_FIELDS, desc)},
                "main_orientation": round(orientation(raw), 8),
                "component_count": "",
                "placement_score": "",
                **paths,
            }
        )
    manifest_path = sample_root / "sampled_masks_manifest.csv"
    write_csv_records(manifest_path, records)
    summary = {"sample_count": len(records), "checkpoint": str(checkpoint_path), "manifest": str(manifest_path)}
    write_json(sample_root / "sampled_masks_summary.json", summary)
    print(summary)
