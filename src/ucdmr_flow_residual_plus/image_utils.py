from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def load_labelme_mask(path: str | Path, size: tuple[int, int], *, label: str = "crack") -> Image.Image:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for shape in data.get("shapes", []):
        if str(shape.get("label", "")).lower() != label:
            continue
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        xy = [(float(x), float(y)) for x, y in points]
        if str(shape.get("shape_type", "polygon")).lower() == "line":
            draw.line(xy, fill=255, width=3)
        else:
            draw.polygon(xy, fill=255)
    return mask


def bool_mask(mask: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(mask, Image.Image):
        arr = np.asarray(mask.convert("L"))
    else:
        arr = mask
    return arr > 0


def save_mask(path: str | Path, mask: np.ndarray | Image.Image) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(mask, Image.Image):
        mask.convert("L").save(path)
        return
    arr = np.asarray(mask)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return (np.asarray(image.filter(ImageFilter.MaxFilter(size=radius * 2 + 1))) > 0).astype(bool)


def gray_from_rgb(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
