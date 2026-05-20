from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from ucdmr_flow_residual_plus.image_utils import bool_mask, gray_from_rgb, save_mask
from ucdmr_flow_residual_plus.io_utils import ensure_dir

from ucdmr_flow_residual_plus.paths import dataset_path


def inpaint_pseudo_normal(rgb: np.ndarray, mask: np.ndarray, *, method: str, blur_radius: float = 9.0) -> np.ndarray:
    mask = mask.astype(bool)
    if not np.any(mask):
        return rgb.copy()
    method = method.lower()
    if method in {"opencv_telea", "telea", "opencv_ns", "navier_stokes"}:
        try:
            import cv2

            flag = cv2.INPAINT_TELEA if method in {"opencv_telea", "telea"} else cv2.INPAINT_NS
            return cv2.inpaint(rgb, (mask.astype(np.uint8) * 255), 3, flag)
        except ModuleNotFoundError:
            method = "local_blur"
    out = rgb.copy()
    if method == "local_blur":
        blurred = np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(radius=blur_radius)))
        out[mask] = blurred[mask]
        return out
    if method in {"local_median", "texture_copy"}:
        band = bool_mask(Image.fromarray(mask.astype(np.uint8) * 255, mode="L").filter(ImageFilter.MaxFilter(size=49))) & ~mask
        fill = np.median(rgb[band], axis=0).astype(np.uint8) if np.any(band) else np.median(rgb.reshape(-1, 3), axis=0).astype(np.uint8)
        out[mask] = fill
        return out
    raise ValueError(f"Unsupported pseudo-normal method: {method}")


def pseudo_quality(crack: np.ndarray, pseudo: np.ndarray, *, inpaint_mask: np.ndarray, gate_mask: np.ndarray) -> dict[str, float]:
    diff = gray_from_rgb(np.abs(crack.astype(np.float32) - pseudo.astype(np.float32)))
    outside = ~gate_mask.astype(bool)
    inside = inpaint_mask.astype(bool)
    outside_l1 = float(diff[outside].mean()) if np.any(outside) else 0.0
    inside_l1 = float(diff[inside].mean()) if np.any(inside) else 0.0
    band = gate_mask.astype(bool) & ~inpaint_mask.astype(bool)
    artifact_band_energy = float(diff[band].mean()) if np.any(band) else 0.0
    return {
        "outside_l1": round(outside_l1, 6),
        "inside_l1": round(inside_l1, 6),
        "artifact_band_energy": round(artifact_band_energy, 6),
        "pseudo_quality_score": round(max(0.0, inside_l1 - outside_l1 - 0.5 * artifact_band_energy), 6),
        "teacher_crack_score_on_pseudo": "",
    }


def prepare_pseudo_normal_plus(
    row: dict[str, str],
    *,
    dataset_root: Path,
    output_root: Path,
    method: str,
    max_outside_l1: float,
    min_quality_score: float,
) -> dict[str, object]:
    sample_id = row["sample_id"]
    crack_path = dataset_path(dataset_root, row["dataset_relative_path"])
    crack = np.asarray(Image.open(crack_path).convert("RGB"))
    inpaint_mask = bool_mask(Image.open(row["m_inpaint_path"]))
    gate_mask = np.asarray(Image.open(row["m_gate_path"]).convert("L"), dtype=np.float32) > 0
    pseudo = inpaint_pseudo_normal(crack, inpaint_mask, method=method)
    metrics = pseudo_quality(crack, pseudo, inpaint_mask=inpaint_mask, gate_mask=gate_mask)

    pseudo_path = output_root / "pseudo_normal" / "images" / f"{sample_id}.png"
    abs_diff_path = output_root / "pseudo_normal" / "abs_diff" / f"{sample_id}.png"
    mask_path = output_root / "pseudo_normal" / "m_inpaint" / f"{sample_id}.png"
    ensure_dir(pseudo_path.parent)
    ensure_dir(abs_diff_path.parent)
    ensure_dir(mask_path.parent)
    Image.fromarray(pseudo, mode="RGB").save(pseudo_path)
    diff = gray_from_rgb(np.abs(crack.astype(np.float32) - pseudo.astype(np.float32)))
    Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8), mode="L").save(abs_diff_path)
    save_mask(mask_path, inpaint_mask)
    accepted = float(metrics["outside_l1"]) <= max_outside_l1 and float(metrics["pseudo_quality_score"]) >= min_quality_score
    return {
        **row,
        "pseudo_image_path": str(pseudo_path),
        "pseudo_abs_diff_path": str(abs_diff_path),
        "pseudo_inpaint_mask_path": str(mask_path),
        "inpaint_method": method,
        "pseudo_quality_accepted": "1" if accepted else "0",
        **metrics,
    }
