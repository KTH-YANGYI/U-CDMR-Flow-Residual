from __future__ import annotations

from pathlib import Path
from typing import Any
import importlib
import sys

import torch
from torch import nn
import torch.nn.functional as F

from ucdmr_flow_residual_plus.models.segmentation import SegmenterPlus


class TeacherWrapper(nn.Module):
    def __init__(self, model: nn.Module, *, normalize_imagenet: bool = False) -> None:
        super().__init__()
        self.model = model
        self.normalize_imagenet = bool(normalize_imagenet)
        self.register_buffer("mean", self._make_buffer([0.485, 0.456, 0.406]), persistent=False)
        self.register_buffer("std", self._make_buffer([0.229, 0.224, 0.225]), persistent=False)

    @staticmethod
    def _make_buffer(values: list[float]) -> Any:
        return torch.tensor(values, dtype=torch.float32).view(1, 3, 1, 1)

    def forward(self, image: Any) -> Any:
        x = image
        if self.normalize_imagenet:
            x = (x - self.mean.to(device=x.device, dtype=x.dtype)) / self.std.to(device=x.device, dtype=x.dtype)
        logits = self.model(x)
        if isinstance(logits, dict):
            for key in ("logits", "out", "pred", "prediction"):
                if key in logits:
                    logits = logits[key]
                    break
            else:
                logits = next(iter(logits.values()))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return logits


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def _infer_legacy_root(checkpoint_path: Path) -> Path | None:
    for parent in checkpoint_path.parents:
        if parent.name == "UNET_two_stage" and (parent / "src" / "models" / "registry.py").exists():
            return parent
    return None


class _LegacyBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Any) -> Any:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class _LegacyConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Any) -> Any:
        return self.block(x)


class _LegacyDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_block = _LegacyConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: Any, skip: Any) -> Any:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv_block(torch.cat([x, skip], dim=1))


class LegacyResNet34UNetBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.encoder_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.encoder_layer1 = self._make_layer(64, 64, blocks=3, stride=1)
        self.encoder_layer2 = self._make_layer(64, 128, blocks=4, stride=2)
        self.encoder_layer3 = self._make_layer(128, 256, blocks=6, stride=2)
        self.encoder_layer4 = self._make_layer(256, 512, blocks=3, stride=2)
        self.center = _LegacyConvBlock(512, 512)
        self.decoder4 = _LegacyDecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.decoder3 = _LegacyDecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.decoder2 = _LegacyDecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.decoder1 = _LegacyDecoderBlock(in_channels=64, skip_channels=64, out_channels=64)
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, *, blocks: int, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = [_LegacyBasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(_LegacyBasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, image: Any) -> Any:
        input_size = image.shape[-2:]
        x0 = self.encoder_stem(image)
        x1 = self.encoder_layer1(self.encoder_pool(x0))
        x2 = self.encoder_layer2(x1)
        x3 = self.encoder_layer3(x2)
        x4 = self.center(self.encoder_layer4(x3))
        d4 = self.decoder4(x4, x3)
        d3 = self.decoder3(d4, x2)
        d2 = self.decoder2(d3, x1)
        d1 = self.decoder1(d2, x0)
        d1 = F.interpolate(d1, size=input_size, mode="bilinear", align_corners=False)
        return self.segmentation_head(d1)


class _LegacySkipAttentionGate(nn.Module):
    def __init__(
        self,
        *,
        skip_channels: int,
        gate_channels: int,
        inter_channels: int | None = None,
        gamma_init: float = 0.0,
    ) -> None:
        super().__init__()
        if inter_channels is None:
            inter_channels = max(16, min(skip_channels, gate_channels) // 2)
        self.skip_proj = nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False)
        self.gate_proj = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, skip: Any, gate: Any) -> Any:
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        attention = self.psi(self.skip_proj(skip) + self.gate_proj(gate))
        return skip * (1.0 + self.gamma * (attention - 1.0))


class LegacySkipGateUNet(LegacyResNet34UNetBaseline):
    def __init__(self, *, skip_attention_levels: list[str] | None = None, skip_attention_gamma_init: float = 0.0) -> None:
        super().__init__()
        levels = set(skip_attention_levels or ["d4", "d3"])
        self.skip_gate_d4 = (
            _LegacySkipAttentionGate(skip_channels=256, gate_channels=512, gamma_init=skip_attention_gamma_init)
            if "d4" in levels
            else None
        )
        self.skip_gate_d3 = (
            _LegacySkipAttentionGate(skip_channels=128, gate_channels=256, gamma_init=skip_attention_gamma_init)
            if "d3" in levels
            else None
        )

    def forward(self, image: Any) -> Any:
        input_size = image.shape[-2:]
        x0 = self.encoder_stem(image)
        x1 = self.encoder_layer1(self.encoder_pool(x0))
        x2 = self.encoder_layer2(x1)
        x3 = self.encoder_layer3(x2)
        x4 = self.center(self.encoder_layer4(x3))
        skip3 = self.skip_gate_d4(x3, x4) if self.skip_gate_d4 is not None else x3
        d4 = self.decoder4(x4, skip3)
        skip2 = self.skip_gate_d3(x2, d4) if self.skip_gate_d3 is not None else x2
        d3 = self.decoder3(d4, skip2)
        d2 = self.decoder2(d3, x1)
        d1 = self.decoder1(d2, x0)
        d1 = F.interpolate(d1, size=input_size, mode="bilinear", align_corners=False)
        return self.segmentation_head(d1)


def _legacy_variant(config: dict[str, Any]) -> str:
    return str(config.get("model_variant") or config.get("model_name") or "resnet34_unet_baseline")


def _load_legacy_native_model(
    *,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    device: Any,
) -> TeacherWrapper:
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise SystemExit(f"Legacy teacher checkpoint has no model_state_dict: {checkpoint_path}")
    config = dict(checkpoint.get("config", {}))
    variant = _legacy_variant(config)
    if variant in {"resnet34_unet_baseline", "811_m0_baseline", "baseline"}:
        model = LegacyResNet34UNetBaseline().to(device)
    elif variant in {"skipgate_d4d3", "811_m2_skip_d4", "811_m3_skip_d3", "811_m4_skip_d4d3"}:
        model = LegacySkipGateUNet(
            skip_attention_levels=list(config.get("skip_attention_levels", ["d4", "d3"])),
            skip_attention_gamma_init=float(config.get("skip_attention_gamma_init", 0.0)),
        ).to(device)
    else:
        raise ValueError(f"Unsupported native legacy teacher variant: {variant}")
    model.load_state_dict(_strip_module_prefix(state_dict))
    model.eval()
    return TeacherWrapper(model, normalize_imagenet=bool(config.get("use_imagenet_normalize", False))).to(device)


def _load_legacy_unet(
    *,
    torch_module: Any,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    source_root: Path | None,
    device: Any,
) -> TeacherWrapper:
    config = dict(checkpoint.get("config", {}))
    if _legacy_variant(config) in {
        "resnet34_unet_baseline",
        "811_m0_baseline",
        "baseline",
        "skipgate_d4d3",
        "811_m2_skip_d4",
        "811_m3_skip_d3",
        "811_m4_skip_d4d3",
    }:
        return _load_legacy_native_model(checkpoint=checkpoint, checkpoint_path=checkpoint_path, device=device)
    root = source_root or _infer_legacy_root(checkpoint_path)
    if root is None:
        raise SystemExit(
            "Legacy UNET_two_stage checkpoint detected, but its source root was not found. "
            "Pass --teacher-source-root /path/to/UNET_two_stage."
        )
    root = root.resolve()
    if not (root / "src" / "models" / "registry.py").exists():
        raise SystemExit(f"Legacy teacher source root does not look like UNET_two_stage: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    registry = importlib.import_module("src.models.registry")
    model = registry.build_model_from_config(config).to(device)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise SystemExit(f"Legacy teacher checkpoint has no model_state_dict: {checkpoint_path}")
    model.load_state_dict(_strip_module_prefix(state_dict))
    model.eval()
    return TeacherWrapper(model, normalize_imagenet=bool(config.get("use_imagenet_normalize", False))).to(device)


def load_teacher_segmenter(
    *,
    torch_module: Any,
    checkpoint_path: Path,
    device: Any,
    encoder: str | None = None,
    base_channels: int | None = None,
    source_root: Path | None = None,
) -> TeacherWrapper:
    try:
        checkpoint = torch_module.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch_module.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"Unsupported teacher checkpoint format: {checkpoint_path}")
    if "model" in checkpoint:
        args = checkpoint.get("args", {})
        model = SegmenterPlus(
            encoder_name=str(encoder or args.get("encoder", "resnet34")),
            pretrained=False,
            base_channels=int(base_channels or args.get("base_channels", 48)),
        ).to(device)
        model.load_state_dict(_strip_module_prefix(checkpoint["model"]))
        model.eval()
        return TeacherWrapper(model, normalize_imagenet=False).to(device)
    if "model_state_dict" in checkpoint and "config" in checkpoint:
        return _load_legacy_unet(
            torch_module=torch_module,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            source_root=source_root,
            device=device,
        )
    raise SystemExit(
        f"Unsupported teacher checkpoint keys in {checkpoint_path}. "
        "Expected current {'model', 'args'} or legacy {'model_state_dict', 'config'}."
    )
