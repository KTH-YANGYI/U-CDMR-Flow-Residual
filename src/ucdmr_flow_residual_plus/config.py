from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

from ucdmr_flow_residual_plus.paths import DEFAULT_CONFIG, DEFAULT_DATASET_ROOT, DEFAULT_METHOD_ROOT
from ucdmr_flow_residual_plus.constants import base_domain_name


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    if not config_path.exists() or yaml is None:
        return {}
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def nested_get(config: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    node: Any = config
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def resolve_dataset_root(config: dict[str, Any], override: str | Path | None = None) -> Path:
    if override is not None:
        return Path(override).expanduser()
    return Path(nested_get(config, ("dataset", "root"), DEFAULT_DATASET_ROOT)).expanduser()


def resolve_output_root(config: dict[str, Any], override: str | Path | None = None) -> Path:
    if override is not None:
        return Path(override).expanduser()
    return Path(nested_get(config, ("artifacts", "root"), DEFAULT_METHOD_ROOT)).expanduser()


def domain_value(config: dict[str, Any], section: str, key: str, domain: str, default: int | float) -> int | float:
    value = nested_get(config, (section, key), default)
    if isinstance(value, dict):
        if domain in value:
            return value[domain]
        return value.get(base_domain_name(domain), default)
    return value
