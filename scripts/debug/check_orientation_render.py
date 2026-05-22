from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from ucdmr_flow_residual_plus.mask_representations import orientation


def line_mask(theta: float, size: int = 256) -> np.ndarray:
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    cx = cy = size // 2
    radius = size // 3
    dx = math.cos(theta) * radius
    dy = math.sin(theta) * radius
    draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=255, width=5)
    return np.asarray(img) > 0


def wrap_axial(angle: float) -> float:
    while angle < -math.pi / 2:
        angle += math.pi
    while angle > math.pi / 2:
        angle -= math.pi
    return angle


def main() -> None:
    for deg in [-60, -30, 0, 30, 60]:
        base = line_mask(math.radians(deg))
        base_theta = orientation(base)
        print(f"base target={deg:+.1f}, measured={math.degrees(base_theta):+.2f}")
        for pil_angle in [-45, -30, 30, 45]:
            rot = Image.fromarray(base.astype("uint8") * 255).rotate(
                pil_angle,
                resample=Image.Resampling.NEAREST,
                expand=False,
            )
            actual = orientation(np.asarray(rot) > 0)
            delta = math.degrees(wrap_axial(actual - base_theta))
            print(f"  PIL rotate={pil_angle:+.1f}, measured={math.degrees(actual):+.2f}, delta={delta:+.2f}")


if __name__ == "__main__":
    main()
