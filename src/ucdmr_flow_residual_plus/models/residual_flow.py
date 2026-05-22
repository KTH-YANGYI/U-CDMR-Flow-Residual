from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from ucdmr_flow_residual_plus.models.blocks import ConvBlock, resize_like


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    if t.ndim == 1:
        t = t[:, None]
    t = t.reshape(t.shape[0], 1)
    half = dim // 2
    if half == 0:
        return t
    freqs = torch.exp(
        torch.arange(half, device=t.device, dtype=t.dtype)
        * -(math.log(10000.0) / max(half - 1, 1))
    )
    args = t * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    return emb


class ResidualFlowUNet(nn.Module):
    """Rectified-flow velocity model for mask-gated crack residuals."""

    def __init__(
        self,
        *,
        residual_channels: int = 3,
        context_channels: int = 3,
        condition_channels: int = 7,
        base_channels: int = 48,
        domain_count: int = 3,
        style_dim: int = 16,
        time_dim: int = 128,
        max_velocity: float = 0.0,
        local_refine: bool = False,
        local_refine_channels: int = 64,
    ) -> None:
        super().__init__()
        c = int(base_channels)
        self.style_dim = int(style_dim)
        self.time_dim = int(time_dim)
        self.max_velocity = float(max_velocity)
        in_channels = residual_channels + context_channels + condition_channels
        self.in_block = ConvBlock(in_channels, c)
        self.down1 = ConvBlock(c, c * 2)
        self.down2 = ConvBlock(c * 2, c * 4)
        self.down3 = ConvBlock(c * 4, c * 8)
        bottleneck = c * 8
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, bottleneck),
            nn.SiLU(inplace=True),
            nn.Linear(bottleneck, bottleneck),
        )
        self.domain_embed = nn.Embedding(domain_count, bottleneck)
        self.style_proj = nn.Linear(style_dim, bottleneck)
        self.mid = ConvBlock(bottleneck, bottleneck)
        self.up2 = ConvBlock(bottleneck + c * 4, c * 4)
        self.up1 = ConvBlock(c * 4 + c * 2, c * 2)
        self.up0 = ConvBlock(c * 2 + c, c)
        self.out = nn.Conv2d(c, residual_channels, kernel_size=1)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        condition: torch.Tensor,
        domain_idx: torch.Tensor,
        style: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if style is None:
            style = x_t.new_zeros((x_t.shape[0], self.style_dim))
        x = torch.cat([x_t, context, condition], dim=1)
        f0 = self.in_block(x)
        f1 = self.down1(F.avg_pool2d(f0, 2))
        f2 = self.down2(F.avg_pool2d(f1, 2))
        f3 = self.down3(F.avg_pool2d(f2, 2))
        emb = self.time_proj(sinusoidal_embedding(t.to(dtype=f3.dtype), self.time_dim))
        emb = emb + self.domain_embed(domain_idx).to(dtype=f3.dtype)
        emb = emb + self.style_proj(style.to(dtype=f3.dtype))
        h = self.mid(f3 + emb[:, :, None, None])
        h = resize_like(h, f2)
        h = self.up2(torch.cat([h, f2], dim=1))
        h = resize_like(h, f1)
        h = self.up1(torch.cat([h, f1], dim=1))
        h = resize_like(h, f0)
        h = self.up0(torch.cat([h, f0], dim=1))
        out = self.out(h)
        if self.max_velocity > 0:
            out = torch.tanh(out) * self.max_velocity
        return out


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _patchify_position_embedding(
    *,
    hidden_size: int,
    grid_h: int,
    grid_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if hidden_size % 4 != 0:
        raise ValueError(f"DiT hidden_size must be divisible by 4 for 2D sin/cos positions, got {hidden_size}")
    y = torch.arange(grid_h, device=device, dtype=torch.float32)
    x = torch.arange(grid_w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.cat(
        [
            _sincos_from_positions(xx.reshape(-1), hidden_size // 2),
            _sincos_from_positions(yy.reshape(-1), hidden_size // 2),
        ],
        dim=1,
    ).to(dtype=dtype)[None, :, :]


def _sincos_from_positions(pos: torch.Tensor, dim: int) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError(f"sin/cos embedding dim must be even, got {dim}")
    omega = torch.arange(dim // 2, device=pos.device, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (omega / max(dim / 2.0, 1.0)))
    values = pos.reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat([values.sin(), values.cos()], dim=1)


class _PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class _TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int) -> None:
        super().__init__()
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = sinusoidal_embedding(t.reshape(t.shape[0]).float(), self.frequency_embedding_size)
        return self.mlp(t_freq)


class _SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if hasattr(F, "scaled_dot_product_attention"):
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            x = attn.softmax(dim=-1) @ v
        x = x.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(x)


class _MLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float) -> None:
        super().__init__()
        mlp_hidden = int(hidden_size * float(mlp_ratio))
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DiTBlock(nn.Module):
    """DiT-style transformer block with adaLN-Zero conditioning."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = _SelfAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = _MLP(hidden_size, mlp_ratio)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(_modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class _DiTFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(_modulate(self.norm_final(x), shift, scale))


class ResidualFlowDiT(nn.Module):
    """DiT backbone for the same mask-gated rectified-flow residual target."""

    def __init__(
        self,
        *,
        residual_channels: int = 3,
        context_channels: int = 3,
        condition_channels: int = 7,
        patch_size: int = 32,
        hidden_size: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        domain_count: int = 3,
        style_dim: int = 16,
        time_dim: int = 128,
        max_velocity: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % 4 != 0:
            raise ValueError(f"DiT hidden_size must be divisible by 4, got {hidden_size}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        in_channels = residual_channels + context_channels + condition_channels
        self.out_channels = int(residual_channels)
        self.patch_size = int(patch_size)
        self.hidden_size = int(hidden_size)
        self.style_dim = int(style_dim)
        self.time_dim = int(time_dim)
        self.max_velocity = float(max_velocity)
        self.local_refine_enabled = bool(local_refine)
        self.x_embedder = _PatchEmbed(in_channels, self.hidden_size, self.patch_size)
        self.t_embedder = _TimestepEmbedder(self.hidden_size, self.time_dim)
        self.domain_embed = nn.Embedding(domain_count, self.hidden_size)
        self.style_proj = nn.Linear(self.style_dim, self.hidden_size)
        self.blocks = nn.ModuleList(
            [_DiTBlock(self.hidden_size, int(num_heads), float(mlp_ratio)) for _ in range(int(depth))]
        )
        self.final_layer = _DiTFinalLayer(self.hidden_size, self.patch_size, self.out_channels)
        if self.local_refine_enabled:
            refine_channels = int(local_refine_channels)
            self.local_refine = nn.Sequential(
                nn.Conv2d(self.out_channels + condition_channels, refine_channels, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(refine_channels, refine_channels, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(refine_channels, self.out_channels, kernel_size=3, padding=1),
            )
        else:
            self.local_refine = None
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _init_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_init_linear)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.domain_embed.weight, std=0.02)
        nn.init.normal_(self.style_proj.weight, std=0.02)
        nn.init.constant_(self.style_proj.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        if self.local_refine is not None:
            nn.init.constant_(self.local_refine[-1].weight, 0)
            nn.init.constant_(self.local_refine[-1].bias, 0)

    def unpatchify(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        p = self.patch_size
        c = self.out_channels
        x = x.reshape(x.shape[0], grid_h, grid_w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, grid_h * p, grid_w * p)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        condition: torch.Tensor,
        domain_idx: torch.Tensor,
        style: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if style is None:
            style = x_t.new_zeros((x_t.shape[0], self.style_dim))
        height, width = x_t.shape[-2:]
        x = torch.cat([x_t, context, condition], dim=1)
        pad_h = (-height) % self.patch_size
        pad_w = (-width) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        grid_h = x.shape[-2] // self.patch_size
        grid_w = x.shape[-1] // self.patch_size
        x = self.x_embedder(x)
        x = x + _patchify_position_embedding(
            hidden_size=self.hidden_size,
            grid_h=grid_h,
            grid_w=grid_w,
            device=x.device,
            dtype=x.dtype,
        )
        c = self.t_embedder(t)
        c = c + self.domain_embed(domain_idx)
        c = c + self.style_proj(style.to(dtype=c.dtype))
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        out = self.unpatchify(x, grid_h, grid_w)[:, :, :height, :width]
        if self.local_refine is not None:
            out = out + self.local_refine(torch.cat([out, condition], dim=1))
        if self.max_velocity > 0:
            out = torch.tanh(out) * self.max_velocity
        return out


def normalize_residual_flow_model_type(model_type: str | None) -> str:
    name = (model_type or "residual_flow_unet").strip().lower().replace("-", "_")
    if name in {"unet", "residual_flow_unet"}:
        return "residual_flow_unet"
    if name in {"dit", "residual_flow_dit"}:
        return "residual_flow_dit"
    raise ValueError(f"Unknown residual flow model_type={model_type!r}")


def build_residual_flow_model(
    *,
    model_type: str | None = "residual_flow_unet",
    residual_channels: int = 3,
    context_channels: int = 3,
    condition_channels: int = 7,
    base_channels: int = 48,
    domain_count: int = 3,
    style_dim: int = 16,
    time_dim: int = 128,
    max_velocity: float = 0.0,
    dit_patch_size: int = 32,
    dit_hidden_size: int = 384,
    dit_depth: int = 8,
    dit_num_heads: int = 6,
    dit_mlp_ratio: float = 4.0,
    dit_local_refine: bool = False,
    dit_local_refine_channels: int = 64,
) -> nn.Module:
    resolved_type = normalize_residual_flow_model_type(model_type)
    if resolved_type == "residual_flow_unet":
        return ResidualFlowUNet(
            residual_channels=residual_channels,
            context_channels=context_channels,
            condition_channels=condition_channels,
            base_channels=base_channels,
            domain_count=domain_count,
            style_dim=style_dim,
            time_dim=time_dim,
            max_velocity=max_velocity,
        )
    return ResidualFlowDiT(
        residual_channels=residual_channels,
        context_channels=context_channels,
        condition_channels=condition_channels,
        patch_size=dit_patch_size,
        hidden_size=dit_hidden_size,
        depth=dit_depth,
        num_heads=dit_num_heads,
        mlp_ratio=dit_mlp_ratio,
        domain_count=domain_count,
        style_dim=style_dim,
        time_dim=time_dim,
        max_velocity=max_velocity,
        local_refine=dit_local_refine,
        local_refine_channels=dit_local_refine_channels,
    )
