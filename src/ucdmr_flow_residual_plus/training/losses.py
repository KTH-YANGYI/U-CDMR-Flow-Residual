from __future__ import annotations


def residual_flow_loss(
    *,
    torch_module,
    pred_v,
    target_v,
    m_band,
    m_gate,
    valid_mask,
    lambda_flow: float,
    lambda_outside: float,
    lambda_leak: float,
    lambda_tv: float = 0.0,
    lambda_chroma: float = 0.0,
):
    pred_v = pred_v.float()
    target_v = target_v.float()
    m_band = m_band.float()
    m_gate = m_gate.float()
    valid_mask = valid_mask.float()
    valid_w = valid_mask.to(dtype=pred_v.dtype).clamp(0.0, 1.0).expand_as(pred_v)
    inside_w = (m_band.clamp(0.0, 1.0) + 0.25 * m_gate.clamp(0.0, 1.0)).clamp(0.0, 1.0).expand_as(pred_v) * valid_w
    outside_w = (m_gate <= 0.001).to(dtype=pred_v.dtype).expand_as(pred_v) * valid_w
    flow = ((pred_v - target_v).pow(2) * inside_w).sum() / inside_w.sum().clamp_min(1.0)
    outside = (pred_v.pow(2) * outside_w).sum() / outside_w.sum().clamp_min(1.0)
    leak = (torch_module.abs(pred_v) * outside_w).sum() / outside_w.sum().clamp_min(1.0)
    tv_y_w = torch_module.minimum(inside_w[:, :, 1:, :], inside_w[:, :, :-1, :])
    tv_x_w = torch_module.minimum(inside_w[:, :, :, 1:], inside_w[:, :, :, :-1])
    tv_y = (torch_module.abs(pred_v[:, :, 1:, :] - pred_v[:, :, :-1, :]) * tv_y_w).sum() / tv_y_w.sum().clamp_min(1.0)
    tv_x = (torch_module.abs(pred_v[:, :, :, 1:] - pred_v[:, :, :, :-1]) * tv_x_w).sum() / tv_x_w.sum().clamp_min(1.0)
    tv = 0.5 * (tv_y + tv_x)
    luma_v = pred_v.mean(dim=1, keepdim=True)
    chroma = ((pred_v - luma_v).pow(2) * inside_w).sum() / inside_w.sum().clamp_min(1.0)
    loss = lambda_flow * flow + lambda_outside * outside + lambda_leak * leak + lambda_tv * tv + lambda_chroma * chroma
    return loss, {
        "loss": float(loss.detach().cpu()),
        "flow_mse": float(flow.detach().cpu()),
        "outside_mse": float(outside.detach().cpu()),
        "velocity_leak": float(leak.detach().cpu()),
        "velocity_tv": float(tv.detach().cpu()),
        "velocity_chroma": float(chroma.detach().cpu()),
    }


def segmentation_loss(torch_module, logits, target, *, valid_mask, bce_weight: float, dice_weight: float, focal_weight: float, pos_weight: float):
    pos = torch_module.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    valid = valid_mask.to(device=logits.device, dtype=logits.dtype).clamp(0.0, 1.0)
    bce_map = torch_module.nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos, reduction="none")
    bce = (bce_map * valid).sum() / valid.sum().clamp_min(1.0)
    probs = torch_module.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    inter = (probs * target * valid).sum(dim=dims)
    denom = ((probs + target) * valid).sum(dim=dims)
    dice = 1.0 - ((2.0 * inter + 1e-6) / (denom + 1e-6)).mean()
    pt = probs * target + (1.0 - probs) * (1.0 - target)
    focal_map = -(1.0 - pt).pow(2.0) * torch_module.log(pt.clamp_min(1e-6))
    focal = (focal_map * valid).sum() / valid.sum().clamp_min(1.0)
    loss = bce_weight * bce + dice_weight * dice + focal_weight * focal
    return loss, {
        "loss": float(loss.detach().cpu()),
        "bce": float(bce.detach().cpu()),
        "dice_loss": float(dice.detach().cpu()),
        "focal": float(focal.detach().cpu()),
    }
