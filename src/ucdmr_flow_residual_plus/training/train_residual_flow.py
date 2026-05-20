from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from ucdmr_flow_residual_plus.io_utils import read_csv_records, write_json

from ucdmr_flow_residual_plus.config import load_config, resolve_dataset_root, resolve_output_root
from ucdmr_flow_residual_plus.training.utils import autocast_context, jsonable_args, make_grad_scaler


def dry_run_summary(args: Any) -> dict[str, object]:
    config = load_config(args.config)
    dataset_root = resolve_dataset_root(config, args.dataset_root)
    output_root = resolve_output_root(config, args.output_root)
    pseudo_manifest = Path(args.pseudo_manifest) if args.pseudo_manifest else output_root / "pseudo_normal" / "pseudo_manifest.csv"
    return {
        "stage": "plus_residual_flow",
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "pseudo_manifest": str(pseudo_manifest),
        "split": args.split,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_mode": "full_native",
        "flow_sigma": args.flow_sigma,
        "distributed_launcher": "torchrun/slurm_env",
        "dry_run": True,
    }


def train(args: Any) -> None:
    try:
        import torch
        from torch.nn.parallel import DistributedDataParallel
        from torch.utils.data import DataLoader, DistributedSampler
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for plus residual flow training.") from exc

    from ucdmr_flow_residual_plus.models.residual_flow import ResidualFlowUNet
    from ucdmr_flow_residual_plus.training.datasets import CONDITION_CHANNELS, PlusResidualDataset, native_collate
    from ucdmr_flow_residual_plus.training.distributed import barrier, cleanup, init_distributed
    from ucdmr_flow_residual_plus.training.losses import residual_flow_loss

    config = load_config(args.config)
    dataset_root = resolve_dataset_root(config, args.dataset_root)
    output_root = resolve_output_root(config, args.output_root)
    pseudo_manifest = Path(args.pseudo_manifest) if args.pseudo_manifest else output_root / "pseudo_normal" / "pseudo_manifest.csv"
    train_root = Path(args.train_output) if args.train_output else output_root / "residual_flow_plus"
    ckpt_root = train_root / "checkpoints"
    report_root = output_root / "reports"
    rows = [
        row
        for row in read_csv_records(pseudo_manifest)
        if row.get("split", args.split) == args.split and row.get("pseudo_quality_accepted", "1") == "1"
    ]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        raise SystemExit(f"No accepted pseudo-normal rows found for split={args.split}: {pseudo_manifest}")

    state = init_distributed(torch)
    if state.is_main:
        ckpt_root.mkdir(parents=True, exist_ok=True)
        report_root.mkdir(parents=True, exist_ok=True)
        print({"stage": "plus_residual_flow", "rows": len(rows), "device": str(state.device), "distributed": state.distributed}, flush=True)

    dataset = PlusResidualDataset(
        pseudo_rows=rows,
        dataset_root=dataset_root,
        samples_per_epoch=args.samples_per_epoch,
        seed=args.seed,
        style_dim=args.style_dim,
        style_dropout=args.style_dropout,
    )
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True) if state.distributed else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None, num_workers=args.workers, pin_memory=torch.cuda.is_available(), drop_last=True, collate_fn=native_collate)
    model = ResidualFlowUNet(
        condition_channels=CONDITION_CHANNELS,
        base_channels=args.base_channels,
        style_dim=args.style_dim,
        time_dim=args.time_dim,
        max_velocity=args.max_velocity,
    ).to(state.device)
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
            context = batch["context"].to(state.device, non_blocking=True)
            target = batch["target"].to(state.device, non_blocking=True)
            condition = batch["condition"].to(state.device, non_blocking=True)
            m_band = batch["m_band"].to(state.device, non_blocking=True)
            m_gate = batch["m_gate"].to(state.device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(state.device, non_blocking=True)
            style = batch["style"].to(state.device, non_blocking=True)
            domain_idx = batch["domain_idx"].to(state.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(torch, enabled=args.amp and torch.cuda.is_available()):
                gate_support = ((m_gate > 0.0) & (valid_mask > 0.5)).to(dtype=target.dtype)
                x1 = (target - context) * gate_support
                x0 = torch.randn_like(x1) * float(args.flow_sigma) * gate_support
                t = torch.rand((x1.shape[0], 1, 1, 1), device=state.device, dtype=x1.dtype)
                x_t = (1.0 - t) * x0 + t * x1
                target_v = x1 - x0
                pred_v = model(x_t, t, context, condition, domain_idx, style)
                loss, parts = residual_flow_loss(
                    torch_module=torch,
                    pred_v=pred_v,
                    target_v=target_v,
                    m_band=m_band,
                    m_gate=m_gate,
                    valid_mask=valid_mask,
                    lambda_flow=args.lambda_flow,
                    lambda_outside=args.lambda_outside,
                    lambda_leak=args.lambda_leak,
                )
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
                ckpt = {"model": target_model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "global_step": global_step, "args": jsonable_args(args), "metrics": metrics, "stage": "residual_flow_plus"}
                torch.save(ckpt, ckpt_root / f"epoch_{epoch:04d}.pt")
                torch.save(ckpt, ckpt_root / "latest.pt")
            write_json(report_root / "plus_residual_flow_training.json", {"history": history, "args": jsonable_args(args)})
        barrier(torch, state)
    cleanup(torch)
