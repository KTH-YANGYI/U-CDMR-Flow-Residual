from __future__ import annotations


def residual_loss(
    *,
    torch_module,
    context,
    target,
    delta,
    m_raw,
    m_band,
    m_gate,
    lambda_inside: float,
    lambda_outside: float,
    lambda_leak: float,
    lambda_edge: float,
):
    recon = (context + m_gate * delta).clamp(0.0, 1.0)
    inside_w = m_band.clamp(0.0, 1.0).expand_as(recon)
    outside_w = (1.0 - m_gate).expand_as(recon)
    inside = (torch_module.abs(recon - target) * inside_w).sum() / inside_w.sum().clamp_min(1.0)
    outside = (torch_module.abs(recon - context) * outside_w).sum() / outside_w.sum().clamp_min(1.0)
    leak = (torch_module.abs(delta) * outside_w).sum() / outside_w.sum().clamp_min(1.0)
    if lambda_edge > 0:
        gx_r = torch_module.abs(recon[..., :, 1:] - recon[..., :, :-1])
        gx_t = torch_module.abs(target[..., :, 1:] - target[..., :, :-1])
        gy_r = torch_module.abs(recon[..., 1:, :] - recon[..., :-1, :])
        gy_t = torch_module.abs(target[..., 1:, :] - target[..., :-1, :])
        wx = m_raw.expand_as(recon)[..., :, 1:]
        wy = m_raw.expand_as(recon)[..., 1:, :]
        edge = (torch_module.abs(gx_r - gx_t) * wx).sum() / wx.sum().clamp_min(1.0)
        edge = edge + (torch_module.abs(gy_r - gy_t) * wy).sum() / wy.sum().clamp_min(1.0)
    else:
        edge = inside.new_tensor(0.0)
    loss = lambda_inside * inside + lambda_outside * outside + lambda_leak * leak + lambda_edge * edge
    return loss, {
        "loss": float(loss.detach().cpu()),
        "inside_l1": float(inside.detach().cpu()),
        "outside_identity": float(outside.detach().cpu()),
        "residual_leak": float(leak.detach().cpu()),
        "edge": float(edge.detach().cpu()),
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
