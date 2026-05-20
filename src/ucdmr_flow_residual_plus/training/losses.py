from __future__ import annotations


def residual_flow_loss(
    *,
    torch_module,
    pred_v,
    target_v,
    m_band,
    m_gate,
    lambda_flow: float,
    lambda_outside: float,
    lambda_leak: float,
):
    inside_w = (m_band.clamp(0.0, 1.0) + 0.25 * m_gate.clamp(0.0, 1.0)).clamp(0.0, 1.0).expand_as(pred_v)
    outside_w = (m_gate <= 0.001).to(dtype=pred_v.dtype).expand_as(pred_v)
    flow = ((pred_v - target_v).pow(2) * inside_w).sum() / inside_w.sum().clamp_min(1.0)
    outside = (pred_v.pow(2) * outside_w).sum() / outside_w.sum().clamp_min(1.0)
    leak = (torch_module.abs(pred_v) * outside_w).sum() / outside_w.sum().clamp_min(1.0)
    loss = lambda_flow * flow + lambda_outside * outside + lambda_leak * leak
    return loss, {
        "loss": float(loss.detach().cpu()),
        "flow_mse": float(flow.detach().cpu()),
        "outside_mse": float(outside.detach().cpu()),
        "velocity_leak": float(leak.detach().cpu()),
    }


def segmentation_loss(torch_module, logits, target, *, bce_weight: float, dice_weight: float, focal_weight: float, pos_weight: float):
    pos = torch_module.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    bce = torch_module.nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos)
    probs = torch_module.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    inter = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = 1.0 - ((2.0 * inter + 1e-6) / (denom + 1e-6)).mean()
    pt = probs * target + (1.0 - probs) * (1.0 - target)
    focal = (-(1.0 - pt).pow(2.0) * torch_module.log(pt.clamp_min(1e-6))).mean()
    loss = bce_weight * bce + dice_weight * dice + focal_weight * focal
    return loss, {
        "loss": float(loss.detach().cpu()),
        "bce": float(bce.detach().cpu()),
        "dice_loss": float(dice.detach().cpu()),
        "focal": float(focal.detach().cpu()),
    }
