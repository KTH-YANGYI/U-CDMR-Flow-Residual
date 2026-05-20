from __future__ import annotations

from pathlib import Path
from typing import Any


def jsonable_args(args: Any) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value if isinstance(value, (str, int, float, bool)) or value is None else str(value)
    return out


def make_grad_scaler(torch_module: Any, enabled: bool) -> Any:
    if hasattr(torch_module, "amp"):
        return torch_module.amp.GradScaler("cuda", enabled=enabled)
    return torch_module.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(torch_module: Any, enabled: bool) -> Any:
    if hasattr(torch_module, "amp"):
        return torch_module.amp.autocast("cuda", enabled=enabled)
    return torch_module.cuda.amp.autocast(enabled=enabled)
