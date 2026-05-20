from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from ucdmr_flow_residual_plus.io_utils import read_csv_records, write_json

from ucdmr_flow_residual_plus.config import load_config, resolve_dataset_root, resolve_output_root
from ucdmr_flow_residual_plus.training.utils import autocast_context, jsonable_args, make_grad_scaler


def dry_run_summary(args: Any, *, stage_name: str) -> dict[str, object]:
    config = load_config(args.config)
    dataset_root = resolve_dataset_root(config, args.dataset_root)
    output_root = resolve_output_root(config, args.output_root)
    return {
        "stage": stage_name,
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "encoder": args.encoder,
        "pretrained": args.pretrained,
        "include_synthetic": args.include_synthetic,
        "split": args.split,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "tile_size": args.tile_size,
        "distributed_launcher": "torchrun/slurm_env",
        "dry_run": True,
    }


def _rows(args: Any, output_root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    split_manifest = Path(args.split_manifest) if args.split_manifest else output_root / "data" / "manifest_splits.csv"
    real_rows = [row for row in read_csv_records(split_manifest) if row.get("split", args.split) == args.split and row.get("label") in {"crack", "normal"}]
    synthetic_rows: list[dict[str, str]] = []
    if args.include_synthetic:
        synthetic_manifest = Path(args.synthetic_manifest) if args.synthetic_manifest else output_root / "synthetic" / "filtered" / "synthetic_filtered.csv"
        if synthetic_manifest.exists():
            synthetic_rows = [row for row in read_csv_records(synthetic_manifest) if row.get("filter_keep", "1") == "1"]
        elif args.require_synthetic:
            raise SystemExit(f"Synthetic manifest not found: {synthetic_manifest}")
    if args.max_samples is not None:
        real_rows = real_rows[: args.max_samples]
        synthetic_rows = synthetic_rows[: args.max_samples]
    if not real_rows and not synthetic_rows:
        raise SystemExit("No segmentation rows found")
    return real_rows, synthetic_rows


def train(args: Any, *, stage_name: str) -> None:
    try:
        import torch
        from torch.nn.parallel import DistributedDataParallel
        from torch.utils.data import DataLoader, DistributedSampler
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for plus segmentation training.") from exc

    from ucdmr_flow_residual_plus.models.segmentation import SegmenterPlus
    from ucdmr_flow_residual_plus.training.datasets import PlusSegmentationDataset
    from ucdmr_flow_residual_plus.training.distributed import barrier, cleanup, init_distributed
    from ucdmr_flow_residual_plus.training.losses import segmentation_loss

    config = load_config(args.config)
    dataset_root = resolve_dataset_root(config, args.dataset_root)
    output_root = resolve_output_root(config, args.output_root)
    train_root = Path(args.train_output) if args.train_output else output_root / stage_name
    ckpt_root = train_root / "checkpoints"
    report_root = output_root / "reports"
    real_rows, synthetic_rows = _rows(args, output_root)
    state = init_distributed(torch)
    if state.is_main:
        ckpt_root.mkdir(parents=True, exist_ok=True)
        report_root.mkdir(parents=True, exist_ok=True)
        print({"stage": stage_name, "real": len(real_rows), "synthetic": len(synthetic_rows), "device": str(state.device)}, flush=True)
    dataset = PlusSegmentationDataset(
        real_rows=real_rows,
        synthetic_rows=synthetic_rows,
        dataset_root=dataset_root,
        tile_size=args.tile_size,
        samples_per_epoch=args.samples_per_epoch,
        seed=args.seed,
        synthetic_weight=args.synthetic_weight,
    )
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True) if state.distributed else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None, num_workers=args.workers, pin_memory=torch.cuda.is_available(), drop_last=True)
    model = SegmenterPlus(encoder_name=args.encoder, pretrained=args.pretrained, base_channels=args.base_channels).to(state.device)
    if state.distributed:
        model = DistributedDataParallel(model, device_ids=[state.local_rank] if torch.cuda.is_available() else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_grad_scaler(torch, enabled=args.amp and torch.cuda.is_available())
    history: list[dict[str, float | int]] = []
    global_step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        start = time.time()
        rolling: dict[str, float] = {}
        for step, batch in enumerate(loader):
            image = batch["image"].to(state.device, non_blocking=True)
            mask = batch["mask"].to(state.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(torch, enabled=args.amp and torch.cuda.is_available()):
                logits = model(image)
                loss, parts = segmentation_loss(torch, logits, mask, bce_weight=args.bce_weight, dice_weight=args.dice_weight, focal_weight=args.focal_weight, pos_weight=args.pos_weight)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            for key, value in parts.items():
                rolling[key] = rolling.get(key, 0.0) + float(value)
            if state.is_main and args.log_every > 0 and global_step % args.log_every == 0:
                denom = step + 1
                print({"epoch": epoch, "step": step, "global_step": global_step, **{k: v / denom for k, v in rolling.items()}}, flush=True)
        if state.is_main:
            denom = max(len(loader), 1)
            metrics = {key: value / denom for key, value in rolling.items()}
            metrics.update({"epoch": epoch, "global_step": global_step, "seconds": round(time.time() - start, 3)})
            history.append(metrics)
            if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
                target_model = model.module if hasattr(model, "module") else model
                ckpt = {"model": target_model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "global_step": global_step, "args": jsonable_args(args), "metrics": metrics, "stage": stage_name}
                torch.save(ckpt, ckpt_root / f"epoch_{epoch:04d}.pt")
                torch.save(ckpt, ckpt_root / "latest.pt")
            write_json(report_root / f"{stage_name}_training.json", {"history": history, "args": jsonable_args(args)})
        barrier(torch, state)
    cleanup(torch)
