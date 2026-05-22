from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from ucdmr_flow_residual_plus.io_utils import read_csv_records, write_json

from ucdmr_flow_residual_plus.config import load_config, resolve_dataset_root, resolve_output_root
from ucdmr_flow_residual_plus.training.utils import autocast_context, jsonable_args, make_grad_scaler


def _tensor_stats(tensor: Any) -> dict[str, object]:
    t = tensor.detach()
    finite = t.isfinite()
    if not bool(finite.any().item()):
        return {
            "shape": tuple(t.shape),
            "finite": False,
            "min": None,
            "max": None,
            "mean": None,
            "nan_count": int(t.isnan().sum().item()),
            "inf_count": int(t.isinf().sum().item()),
        }
    tf = t[finite].float()
    return {
        "shape": tuple(t.shape),
        "finite": bool(finite.all().item()),
        "min": float(tf.min().item()),
        "max": float(tf.max().item()),
        "mean": float(tf.mean().item()),
        "nan_count": int(t.isnan().sum().item()),
        "inf_count": int(t.isinf().sum().item()),
    }


def _assert_finite_tensor(name: str, tensor: Any, *, meta: dict[str, object] | None = None) -> None:
    if not bool(tensor.isfinite().all().item()):
        raise FloatingPointError(
            {
                "error": "non_finite_tensor",
                "name": name,
                "stats": _tensor_stats(tensor),
                "meta": meta or {},
            }
        )


def _find_nonfinite_state_dict_tensors(state_dict: dict[str, Any]) -> list[str]:
    bad: list[str] = []
    for key, value in state_dict.items():
        if hasattr(value, "isfinite") and not bool(value.isfinite().all().item()):
            bad.append(key)
    return bad


def _find_nonfinite_optimizer_tensors(optimizer: Any) -> list[str]:
    bad: list[str] = []
    for state_index, state in enumerate(optimizer.state.values()):
        for key, value in state.items():
            if hasattr(value, "isfinite") and not bool(value.isfinite().all().item()):
                bad.append(f"state_{state_index}:{key}")
    return bad


def _jsonable_value(value: Any) -> object:
    if hasattr(value, "detach"):
        flat = value.detach().cpu().reshape(-1)
        if flat.numel() == 1:
            item = flat.item()
            return item if isinstance(item, (str, int, float, bool)) else str(item)
        return flat.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _batch_meta(batch: dict[str, Any], *, epoch: int, step: int, global_step: int, state: Any) -> dict[str, object]:
    keys = [
        "sample_id",
        "domain",
        "dataset_relative_path",
        "pseudo_image_path",
        "m_raw_ratio",
        "m_band_ratio",
        "native_height",
        "native_width",
    ]
    meta: dict[str, object] = {
        "epoch": epoch,
        "step": step,
        "global_step": global_step,
        "rank": state.rank,
        "local_rank": state.local_rank,
        "world_size": state.world_size,
    }
    for key in keys:
        if key in batch:
            meta[key] = _jsonable_value(batch[key])
    return meta


def _add_runtime_meta(meta: dict[str, object], **values: Any) -> None:
    for key, value in values.items():
        meta[key] = _jsonable_value(value)


def _any_rank_bad(torch_module: Any, state: Any, local_bad: bool) -> bool:
    if not state.distributed:
        return local_bad
    flag = torch_module.tensor([1 if local_bad else 0], device=state.device, dtype=torch_module.int32)
    torch_module.distributed.all_reduce(flag, op=torch_module.distributed.ReduceOp.MAX)
    return bool(flag.item())


def _save_nonfinite_debug(
    *,
    torch_module: Any,
    ckpt_root: Path,
    model: Any,
    optimizer: Any,
    epoch: int,
    step: int,
    global_step: int,
    args: Any,
    batch_meta: dict[str, object],
    state: Any,
    error: object,
    bad_params: list[str] | None = None,
    bad_optimizer_tensors: list[str] | None = None,
) -> Path:
    target_model = model.module if hasattr(model, "module") else model
    path = ckpt_root / f"nonfinite_rank{state.rank}_global_step_{global_step:08d}.pt"
    debug_ckpt = {
        "model": target_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "global_step": global_step,
        "args": jsonable_args(args),
        "batch_meta": batch_meta,
        "bad_params": (bad_params or [])[:100],
        "bad_optimizer_tensors": (bad_optimizer_tensors or [])[:100],
        "error": repr(error),
        "stage": "residual_flow_plus_nonfinite_debug",
        "finite": False,
    }
    torch_module.save(debug_ckpt, path)
    return path


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
        "model_type": args.model_type,
        "dit_patch_size": args.dit_patch_size,
        "dit_hidden_size": args.dit_hidden_size,
        "dit_depth": args.dit_depth,
        "dit_num_heads": args.dit_num_heads,
        "split": args.split,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "batching": "native_size_bucketed_no_padding",
        "image_mode": "full_native",
        "flow_sigma": args.flow_sigma,
        "max_velocity": args.max_velocity,
        "lambda_tv": args.lambda_tv,
        "lambda_chroma": args.lambda_chroma,
        "lr": args.lr,
        "amp": args.amp,
        "amp_dtype": args.amp_dtype,
        "distributed_launcher": "torchrun/slurm_env",
        "dry_run": True,
    }


def train(args: Any) -> None:
    try:
        import torch
        from torch.nn.parallel import DistributedDataParallel
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for plus residual flow training.") from exc

    from ucdmr_flow_residual_plus.models.residual_flow import build_residual_flow_model, normalize_residual_flow_model_type
    from ucdmr_flow_residual_plus.training.datasets import CONDITION_CHANNELS, PlusResidualDataset, SizeBucketBatchSampler, strict_native_collate
    from ucdmr_flow_residual_plus.training.distributed import barrier, cleanup, init_distributed
    from ucdmr_flow_residual_plus.training.losses import residual_flow_loss

    args.model_type = normalize_residual_flow_model_type(args.model_type)
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
        print(
            {
                "stage": "plus_residual_flow",
                "rows": len(rows),
                "device": str(state.device),
                "distributed": state.distributed,
                "amp": args.amp,
                "amp_dtype": args.amp_dtype,
                "model_type": args.model_type,
                "max_velocity": args.max_velocity,
            },
            flush=True,
        )
    barrier(torch, state)

    dataset = PlusResidualDataset(
        pseudo_rows=rows,
        dataset_root=dataset_root,
        samples_per_epoch=args.samples_per_epoch,
        seed=args.seed,
        style_dim=args.style_dim,
        style_dropout=args.style_dropout,
    )
    batch_sampler = SizeBucketBatchSampler(
        rows=rows,
        dataset_root=dataset_root,
        samples_per_epoch=args.samples_per_epoch,
        batch_size=args.batch_size,
        seed=args.seed,
        rank=state.rank,
        world_size=state.world_size,
        drop_last=True,
    )
    if len(batch_sampler) == 0:
        raise SystemExit(
            "No native-size batches were produced. "
            f"rows={len(rows)}, samples_per_epoch={args.samples_per_epoch}, batch_size={args.batch_size}, world_size={state.world_size}"
        )
    if state.is_main:
        print(
            {
                "stage": "plus_residual_flow_batching",
                "strategy": "native_size_bucketed_no_padding",
                "batch_size": args.batch_size,
                "batches_per_rank": len(batch_sampler),
                "size_buckets": batch_sampler.bucket_counts(),
            },
            flush=True,
        )
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=strict_native_collate,
    )
    model = build_residual_flow_model(
        model_type=args.model_type,
        condition_channels=CONDITION_CHANNELS,
        base_channels=args.base_channels,
        style_dim=args.style_dim,
        time_dim=args.time_dim,
        max_velocity=args.max_velocity,
        dit_patch_size=args.dit_patch_size,
        dit_hidden_size=args.dit_hidden_size,
        dit_depth=args.dit_depth,
        dit_num_heads=args.dit_num_heads,
        dit_mlp_ratio=args.dit_mlp_ratio,
    ).to(state.device)
    if state.distributed:
        model = DistributedDataParallel(model, device_ids=[state.local_rank] if torch.cuda.is_available() else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = bool(args.amp and torch.cuda.is_available())
    use_scaler = bool(use_amp and args.amp_dtype == "fp16")
    scaler = make_grad_scaler(torch, enabled=use_scaler)
    history: list[dict[str, float | int]] = []
    global_step = 0
    for epoch in range(args.epochs):
        batch_sampler.set_epoch(epoch)
        start = time.time()
        rolling: dict[str, float] = {}
        for step, batch in enumerate(loader):
            step_global = global_step + 1
            batch_meta = _batch_meta(batch, epoch=epoch, step=step, global_step=step_global, state=state)
            debug_saved = False
            try:
                context = batch["context"].to(state.device, non_blocking=True)
                target = batch["target"].to(state.device, non_blocking=True)
                condition = batch["condition"].to(state.device, non_blocking=True)
                m_band = batch["m_band"].to(state.device, non_blocking=True)
                m_gate = batch["m_gate"].to(state.device, non_blocking=True)
                valid_mask = batch["valid_mask"].to(state.device, non_blocking=True)
                style = batch["style"].to(state.device, non_blocking=True)
                domain_idx = batch["domain_idx"].to(state.device, non_blocking=True)

                _assert_finite_tensor("context", context, meta=batch_meta)
                _assert_finite_tensor("target", target, meta=batch_meta)
                _assert_finite_tensor("condition", condition, meta=batch_meta)
                _assert_finite_tensor("m_gate", m_gate, meta=batch_meta)
                _assert_finite_tensor("m_band", m_band, meta=batch_meta)
                _assert_finite_tensor("valid_mask", valid_mask, meta=batch_meta)
                _add_runtime_meta(
                    batch_meta,
                    m_gate_sum=m_gate.detach().float().sum(dim=(1, 2, 3)),
                    m_band_sum=m_band.detach().float().sum(dim=(1, 2, 3)),
                    context_min=context.detach().float().amin(dim=(1, 2, 3)),
                    context_max=context.detach().float().amax(dim=(1, 2, 3)),
                    target_min=target.detach().float().amin(dim=(1, 2, 3)),
                    target_max=target.detach().float().amax(dim=(1, 2, 3)),
                )

                optimizer.zero_grad(set_to_none=True)
                gate_support = ((m_gate > 0.001) & (valid_mask > 0.5)).to(dtype=target.dtype)
                gate_state_weight = torch.sqrt(m_gate.clamp(0.0, 1.0)) * gate_support
                x1 = (target - context) * gate_support
                x0 = torch.randn_like(x1) * float(args.flow_sigma) * gate_state_weight
                t = torch.rand((x1.shape[0], 1, 1, 1), device=state.device, dtype=x1.dtype)
                x_t = (1.0 - t) * x0 + t * x1
                target_v = x1 - x0
                _add_runtime_meta(
                    batch_meta,
                    t=t,
                    x1_abs_max=x1.detach().abs().float().amax(dim=(1, 2, 3)),
                    x0_abs_max=x0.detach().abs().float().amax(dim=(1, 2, 3)),
                    x_t_abs_max=x_t.detach().abs().float().amax(dim=(1, 2, 3)),
                    target_v_abs_max=target_v.detach().abs().float().amax(dim=(1, 2, 3)),
                    gate_state_sum=gate_state_weight.detach().float().sum(dim=(1, 2, 3)),
                )

                _assert_finite_tensor("x1", x1, meta=batch_meta)
                _assert_finite_tensor("x0", x0, meta=batch_meta)
                _assert_finite_tensor("x_t", x_t, meta=batch_meta)
                _assert_finite_tensor("target_v", target_v, meta=batch_meta)
                with autocast_context(torch, enabled=use_amp, dtype=args.amp_dtype):
                    pred_v = model(x_t, t, context, condition, domain_idx, style)
                _assert_finite_tensor("pred_v", pred_v, meta=batch_meta)
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
                    lambda_tv=args.lambda_tv,
                    lambda_chroma=args.lambda_chroma,
                )
                _assert_finite_tensor("loss", loss, meta=batch_meta)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip if args.grad_clip > 0 else float("inf"),
                    error_if_nonfinite=True,
                )
                _add_runtime_meta(batch_meta, grad_norm=grad_norm)
                parts["grad_norm"] = float(grad_norm.detach().float().cpu().item() if hasattr(grad_norm, "detach") else grad_norm)
                scaler.step(optimizer)
                scaler.update()
                global_step = step_global

                target_model = model.module if hasattr(model, "module") else model
                bad_params = _find_nonfinite_state_dict_tensors(target_model.state_dict())
                has_bad_params = _any_rank_bad(torch, state, bool(bad_params))
                if has_bad_params:
                    if bad_params or state.is_main:
                        _save_nonfinite_debug(
                            torch_module=torch,
                            ckpt_root=ckpt_root,
                            model=model,
                            optimizer=optimizer,
                            epoch=epoch,
                            step=step,
                            global_step=global_step,
                            args=args,
                            batch_meta=batch_meta,
                            state=state,
                            error=f"Non-finite model parameters at global_step={global_step}",
                            bad_params=bad_params,
                        )
                        debug_saved = True
                    raise FloatingPointError(f"Non-finite model parameters at global_step={global_step}")
            except (FloatingPointError, RuntimeError) as exc:
                message = str(exc).lower()
                should_debug = isinstance(exc, FloatingPointError) or "non-finite" in message or "nan" in message or "inf" in message
                if should_debug and not debug_saved:
                    _save_nonfinite_debug(
                        torch_module=torch,
                        ckpt_root=ckpt_root,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        step=step,
                        global_step=step_global,
                        args=args,
                        batch_meta=batch_meta,
                        state=state,
                        error=exc,
                    )
                raise
            for key, value in parts.items():
                rolling[key] = rolling.get(key, 0.0) + float(value)
            if state.is_main and args.log_every > 0 and global_step % args.log_every == 0:
                denom = step + 1
                print({"epoch": epoch, "step": step, "global_step": global_step, **{k: v / denom for k, v in rolling.items()}}, flush=True)
        denom = max(len(loader), 1)
        metrics = {key: value / denom for key, value in rolling.items()}
        metrics.update({"epoch": epoch, "global_step": global_step, "seconds": round(time.time() - start, 3)})
        target_model = model.module if hasattr(model, "module") else model
        model_bad = _find_nonfinite_state_dict_tensors(target_model.state_dict())
        optim_bad = _find_nonfinite_optimizer_tensors(optimizer)
        has_bad_checkpoint_state = _any_rank_bad(torch, state, bool(model_bad or optim_bad))
        if has_bad_checkpoint_state:
            if state.is_main:
                debug_ckpt = {
                    "model": target_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "args": jsonable_args(args),
                    "metrics": metrics,
                    "bad_model_tensors": model_bad[:100],
                    "bad_optimizer_tensors": optim_bad[:100],
                    "stage": "residual_flow_plus_nonfinite_debug",
                    "finite": False,
                }
                torch.save(debug_ckpt, ckpt_root / f"nonfinite_epoch_{epoch:04d}.pt")
            raise FloatingPointError(f"Refusing to save non-finite checkpoint at epoch={epoch}, global_step={global_step}")
        if state.is_main:
            history.append(metrics)
            if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
                ckpt = {
                    "model": target_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "args": jsonable_args(args),
                    "metrics": metrics,
                    "stage": "residual_flow_plus",
                    "finite": True,
                }
                torch.save(ckpt, ckpt_root / f"epoch_{epoch:04d}.pt")
                torch.save(ckpt, ckpt_root / "latest_finite.pt")
                torch.save(ckpt, ckpt_root / "latest.pt")
            write_json(report_root / "plus_residual_flow_training.json", {"history": history, "args": jsonable_args(args)})
        barrier(torch, state)
    cleanup(torch)
