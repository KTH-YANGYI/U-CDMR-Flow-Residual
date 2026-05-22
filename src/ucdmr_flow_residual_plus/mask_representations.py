from __future__ import annotations

from collections import deque
from pathlib import Path
import math

import numpy as np
from PIL import Image, ImageFilter

from ucdmr_flow_residual_plus.image_utils import bool_mask, dilate, load_labelme_mask, save_mask
from ucdmr_flow_residual_plus.io_utils import ensure_dir

from ucdmr_flow_residual_plus.constants import mask_domain_from_row, materialize_domain_fields
from ucdmr_flow_residual_plus.paths import dataset_path


def blur_float_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    if radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(image, dtype=np.float32) / 255.0


def save_float_mask(path: str | Path, mask: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)


def chamfer_distance(mask: np.ndarray) -> np.ndarray:
    """Approximate distance to nearest True pixel with a two-pass chamfer transform."""
    try:
        import cv2

        inv = (~mask.astype(bool)).astype(np.uint8)
        return cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3).astype(np.float32)
    except ModuleNotFoundError:
        pass
    try:
        from scipy import ndimage

        return ndimage.distance_transform_edt(~mask.astype(bool)).astype(np.float32)
    except ModuleNotFoundError:
        pass
    mask = mask.astype(bool)
    inf = 1e6
    dist = np.where(mask, 0.0, inf).astype(np.float32)
    h, w = dist.shape
    diag = math.sqrt(2.0)
    for y in range(h):
        for x in range(w):
            best = dist[y, x]
            if y > 0:
                best = min(best, dist[y - 1, x] + 1.0)
                if x > 0:
                    best = min(best, dist[y - 1, x - 1] + diag)
                if x + 1 < w:
                    best = min(best, dist[y - 1, x + 1] + diag)
            if x > 0:
                best = min(best, dist[y, x - 1] + 1.0)
            dist[y, x] = best
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            best = dist[y, x]
            if y + 1 < h:
                best = min(best, dist[y + 1, x] + 1.0)
                if x > 0:
                    best = min(best, dist[y + 1, x - 1] + diag)
                if x + 1 < w:
                    best = min(best, dist[y + 1, x + 1] + diag)
            if x + 1 < w:
                best = min(best, dist[y, x + 1] + 1.0)
            dist[y, x] = best
    return dist


def _local_window(mask: np.ndarray, pad: int) -> tuple[slice, slice]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + pad + 1)
    return slice(y0, y1), slice(x0, x1)


def signed_distance(mask: np.ndarray, *, clip: float = 64.0) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=np.float32)
    if not np.any(mask):
        return out
    ys, xs = _local_window(mask, int(clip) + 4)
    crop = mask[ys, xs]
    inside = chamfer_distance(~crop)
    outside = chamfer_distance(crop)
    sdf = np.where(crop, inside, -outside)
    out[ys, xs] = np.clip((sdf + clip) / (2.0 * clip), 0.0, 1.0).astype(np.float32)
    return out


def thinning_skeleton(mask: np.ndarray, *, max_iter: int = 128) -> np.ndarray:
    """Zhang-Suen thinning for binary masks."""
    img = mask.astype(np.uint8).copy()
    if img.sum() == 0:
        return img.astype(bool)
    for _ in range(max_iter):
        changed = False
        for phase in (0, 1):
            padded = np.pad(img, 1, mode="constant")
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
            count = sum(neighbors)
            transitions = sum((neighbors[i] == 0) & (neighbors[(i + 1) % 8] == 1) for i in range(8))
            if phase == 0:
                keep = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                keep = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (img == 1) & (count >= 2) & (count <= 6) & (transitions == 1) & keep
            if np.any(remove):
                img[remove] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)


def connected_components(mask: np.ndarray) -> int:
    mask = mask.astype(bool)
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    count = 0
    for y, x in zip(*np.where(mask & ~seen)):
        if seen[y, x]:
            continue
        count += 1
        queue: deque[tuple[int, int]] = deque([(int(y), int(x))])
        seen[y, x] = True
        while queue:
            cy, cx = queue.popleft()
            for ny in range(max(0, cy - 1), min(h, cy + 2)):
                for nx in range(max(0, cx - 1), min(w, cx + 2)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
    return count


def orientation(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if len(xs) < 2:
        return 0.0
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    centered = coords - coords.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(coords) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, int(np.argmax(eigvals))]
    angle = float(np.arctan2(direction[1], direction[0]))
    while angle < -np.pi / 2:
        angle += np.pi
    while angle > np.pi / 2:
        angle -= np.pi
    return angle


def build_plus_regions(
    raw: np.ndarray,
    *,
    inpaint_radius: int,
    band_radius: int,
    gate_radius: int,
    gate_blur: float,
    sdf_clip: float = 64.0,
) -> dict[str, np.ndarray]:
    raw = raw.astype(bool)
    m_inpaint = dilate(raw, inpaint_radius)
    m_band = dilate(raw, band_radius)
    m_gate_binary = dilate(raw, gate_radius)
    m_gate = blur_float_mask(m_gate_binary, gate_blur)
    skeleton = thinning_skeleton(raw)
    sdf = signed_distance(raw, clip=sdf_clip)
    thickness = np.zeros(raw.shape, dtype=np.float32)
    if np.any(raw):
        ys, xs = _local_window(raw, int(sdf_clip) + 4)
        crop = raw[ys, xs]
        thickness[ys, xs] = np.clip((chamfer_distance(~crop) * 2.0) / max(sdf_clip, 1.0), 0.0, 1.0).astype(np.float32)
    return {
        "m_raw": raw,
        "m_inpaint": m_inpaint,
        "m_band": m_band,
        "m_gate": m_gate,
        "m_skeleton": skeleton,
        "m_sdf": sdf,
        "m_thickness": thickness,
    }


def prepare_plus_masks(
    row: dict[str, str],
    *,
    dataset_root: Path,
    output_root: Path,
    inpaint_radius: int,
    band_radius: int,
    gate_radius: int,
    gate_blur: float,
    sdf_clip: float,
    skip_existing: bool = False,
) -> dict[str, object]:
    if row.get("label") != "crack":
        raise ValueError("prepare_plus_masks expects a crack row")
    image_path = dataset_path(dataset_root, row["dataset_relative_path"])
    ann_path = dataset_path(dataset_root, row["annotation_relative_path"])
    with Image.open(image_path) as image:
        width, height = image.size
    raw = bool_mask(load_labelme_mask(ann_path, (width, height)))
    regions = build_plus_regions(
        raw,
        inpaint_radius=inpaint_radius,
        band_radius=band_radius,
        gate_radius=gate_radius,
        gate_blur=gate_blur,
        sdf_clip=sdf_clip,
    )
    row = materialize_domain_fields(row)
    domain = mask_domain_from_row(row)
    sample_id = f"{domain}__{Path(row['dataset_relative_path']).stem}"

    paths: dict[str, Path] = {}
    for name, mask in regions.items():
        path = output_root / name / f"{sample_id}.png"
        if not (skip_existing and path.exists()):
            if mask.dtype == bool:
                save_mask(path, mask)
            else:
                save_float_mask(path, mask)
        paths[f"{name}_path"] = path

    ys, xs = np.where(raw)
    bbox_w = float((xs.max() - xs.min() + 1) / width) if len(xs) else 0.0
    bbox_h = float((ys.max() - ys.min() + 1) / height) if len(ys) else 0.0
    skel = regions["m_skeleton"].astype(bool)
    thickness_values = regions["m_thickness"][raw] if np.any(raw) else np.asarray([0.0], dtype=np.float32)
    total = width * height
    record: dict[str, object] = {
        **row,
        "sample_id": sample_id,
        "domain": domain,
        "effective_domain": domain,
        "base_domain": row.get("base_domain", ""),
        "residual_domain": row.get("residual_domain", ""),
        "dphone_id": row.get("dphone_id", ""),
        "mask_width": width,
        "mask_height": height,
        "inpaint_radius": inpaint_radius,
        "band_radius": band_radius,
        "gate_radius": gate_radius,
        "gate_blur": gate_blur,
        "m_raw_pixels": int(raw.sum()),
        "m_raw_ratio": round(float(raw.sum() / total), 10),
        "bbox_w": round(bbox_w, 8),
        "bbox_h": round(bbox_h, 8),
        "center_x": round(float((xs.mean() + 0.5) / width), 8) if len(xs) else "",
        "center_y": round(float((ys.mean() + 0.5) / height), 8) if len(ys) else "",
        "skeleton_length": int(skel.sum()),
        "skeleton_ratio": round(float(skel.sum() / total), 10),
        "component_count": connected_components(raw),
        "main_orientation": round(orientation(raw), 8),
        "thickness_mean": round(float(thickness_values.mean()), 8),
        "thickness_p90": round(float(np.percentile(thickness_values, 90)), 8),
    }
    for name, mask in regions.items():
        if mask.dtype == bool:
            record[f"{name}_pixels"] = int(mask.sum())
            record[f"{name}_ratio"] = round(float(mask.sum() / total), 10)
    record.update({key: str(value) for key, value in paths.items()})
    return record
