"""
Crop stage - PCA-aligned cropping.

Reads:
    data/{split}/{label}/*.jpg
    artifacts/masks/{split}/{label}/{instance_id}.png
Writes:
    artifacts/crops/images/{split}/{label}/{instance_id}.png
    artifacts/crops/masks/{split}/{label}/{instance_id}.png
    artifacts/crops/boxes/{split}/{label}/{instance_id}.png
    artifacts/crops/metadata/{split}/{label}/{stem}.json
Updates registry: crop_iou, crop_size, alignment_angle, crop_transform
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from core.paths import ProjectPaths

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def process(paths: ProjectPaths, cfg: dict, registry: pd.DataFrame) -> pd.DataFrame:
    """
    Crop all instances using PCA-aligned bounding boxes.

    Returns updated registry with crop columns populated.
    """
    crop_cfg = cfg.get("crop", {})

    # If any crops exist, assume done (same behavior as your current version)
    if paths.crops_dir.exists() and any(paths.crops_dir.rglob("*.png")):
        print("✓ Crops already exist, skipping")
        return registry

    if not paths.masks_dir.exists():
        print("No masks found. Run segment first.")
        return registry

    # Compute output size from transformer config
    map_res = cfg.get("transformer", {}).get("map_resolution", [48, 48])
    patch_size = cfg.get("transformer", {}).get("patch_size", 14)
    out_h = int(map_res[0]) * int(patch_size)
    out_w = int(map_res[1]) * int(patch_size)
    out_size = (out_w, out_h)
    crop_size_json = json.dumps([out_w, out_h])

    # Crop config
    align_pca = bool(crop_cfg.get("align_PCA", True))
    angle_candidates = crop_cfg.get("angle_candidates", [0, 180])
    offset = float(crop_cfg.get("offset_rotation", 0.0))
    scale = float(crop_cfg.get("scale", 1.0))
    reference_id = crop_cfg.get("reference_id", None)
    max_workers = int(crop_cfg.get("max_workers", 8))

    print(f"  Output size: {out_w}x{out_h}")
    print(f"  PCA alignment: {align_pca}")
    print(f"  Angle candidates: {angle_candidates}")
    print(f"  Scale: {scale}")
    print(f"  Workers: {max_workers}")

    # Get reference mask for IoU-based orientation selection
    ref_mask = None
    if align_pca and len(angle_candidates) > 1:
        ref_mask = _get_reference_mask(paths, registry, reference_id, out_size, scale, offset)
        if ref_mask is not None:
            print("  Reference mask loaded for orientation alignment")

    # Prepare output columns (always reset these like your current version)
    registry["_stem"] = registry["instance_id"].apply(_get_stem)
    registry["crop_iou"] = np.nan
    registry["crop_size"] = ""
    registry["alignment_angle"] = np.nan
    registry["crop_transform"] = ""

    grouped = registry.groupby(["split", "label", "_stem"])

    pbar = tqdm(total=len(registry), desc="Cropping", unit="inst")
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for (split, label, stem), group in grouped:
                img_path = _find_source_image(paths.data_dir, split, label, stem)
                if img_path is None:
                    pbar.update(len(group))
                    continue

                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    pbar.update(len(group))
                    continue

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_h, img_w = img_bgr.shape[:2]

                futures = []
                for idx, row in group.iterrows():
                    instance_id = row["instance_id"]
                    mask_path = paths.mask(split, label, instance_id)

                    futures.append(
                        ex.submit(
                            _process_one_instance,
                            paths=paths,
                            split=split,
                            label=label,
                            instance_id=instance_id,
                            idx=idx,
                            img_rgb=img_rgb,
                            orig_size=(img_w, img_h),
                            mask_path=mask_path,
                            out_size=out_size,
                            scale=scale,
                            offset=offset,
                            align_pca=align_pca,
                            angle_candidates=angle_candidates,
                            ref_mask=ref_mask,
                            crop_size_json=crop_size_json,
                        )
                    )

                idxs, ious, sizes, angles, transforms, instance_metadata = [], [], [], [], [], []

                for fut in as_completed(futures):
                    pbar.update(1)

                    res = fut.result()
                    if res is None:
                        continue

                    idxs.append(res["idx"])
                    ious.append(res["iou"])
                    sizes.append(res["crop_size_json"])
                    angles.append(res["angle"])
                    transforms.append(res["transform_json"])
                    instance_metadata.append(res["meta"])

                if idxs:
                    registry.loc[idxs, "crop_iou"] = ious
                    registry.loc[idxs, "crop_size"] = sizes
                    registry.loc[idxs, "alignment_angle"] = angles
                    registry.loc[idxs, "crop_transform"] = transforms

                if instance_metadata:
                    meta_path = paths.crop_metadata(split, label, stem)
                    meta_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(meta_path, "w") as f:
                        json.dump(
                            {
                                "image": f"{split}/{label}/{stem}",
                                "original_size": [img_w, img_h],
                                "instances": sorted(instance_metadata, key=lambda d: d["instance_id"]),
                            },
                            f,
                            indent=2,
                        )
    finally:
        pbar.close()

    registry = registry.drop(columns=["_stem"])

    valid_crops = int(registry["crop_iou"].notna().sum())
    print(f"  ✓ Cropped {valid_crops} instances")
    return registry


def _process_one_instance(
    paths: ProjectPaths,
    split: str,
    label: str,
    instance_id: str,
    idx,
    img_rgb: np.ndarray,
    orig_size: tuple,  # (W, H)
    mask_path: Path,
    out_size: tuple,   # (out_w, out_h)
    scale: float,
    offset: float,
    align_pca: bool,
    angle_candidates: list,
    ref_mask: Optional[np.ndarray],
    crop_size_json: str,
) -> Optional[dict]:
    if not mask_path.exists():
        return None

    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        return None
    mask01 = (mask_raw > 127).astype(np.uint8)

    result = _compute_crop(
        img_rgb=img_rgb,
        mask01=mask01,
        out_size=out_size,
        scale=scale,
        offset=offset,
        align_pca=align_pca,
        angle_candidates=angle_candidates,
        ref_mask=ref_mask,
    )

    # Save crop image and mask
    crop_img_path = paths.crop_image(split, label, instance_id)
    crop_mask_path = paths.crop_mask(split, label, instance_id)
    crop_img_path.parent.mkdir(parents=True, exist_ok=True)
    crop_mask_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(crop_img_path), cv2.cvtColor(result["crop_img"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(crop_mask_path), result["crop_mask"] * 255)

    # Save crop box artifact on a blank canvas in original image coords
    out_w, out_h = out_size
    orig_w, orig_h = orig_size
    M = result["M"].astype(np.float64)
    Minv = cv2.invertAffineTransform(M)

    # Crop corners in crop coords -> original coords
    corners = np.array([[[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]]], dtype=np.float32)
    corners_xy = cv2.transform(corners, Minv)[0]
    poly = np.round(corners_xy).astype(np.int32).reshape(-1, 1, 2)

    # Center in original coords
    center = np.array([[[out_w * 0.5, out_h * 0.5]]], dtype=np.float32)
    center_xy = cv2.transform(center, Minv)[0, 0]
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))

    # Long-axis line endpoints (in crop coords), mapped to original
    if out_w >= out_h:
        a0 = np.array([[[0, out_h * 0.5], [out_w - 1, out_h * 0.5]]], dtype=np.float32)
    else:
        a0 = np.array([[[out_w * 0.5, 0], [out_w * 0.5, out_h - 1]]], dtype=np.float32)
    axis_xy = cv2.transform(a0, Minv)[0]
    p0 = (int(round(axis_xy[0, 0])), int(round(axis_xy[0, 1])))
    p1 = (int(round(axis_xy[1, 0])), int(round(axis_xy[1, 1])))

    canvas = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    red = (255, 0, 255)

    cv2.polylines(canvas, [poly], isClosed=True, color=red, thickness=2, lineType=cv2.LINE_AA)
    cv2.line(canvas, p0, p1, red, thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), radius=4, color=red, thickness=-1, lineType=cv2.LINE_AA)

    box_path = paths.crops_dir / "boxes" / split / label / f"{instance_id}.png"
    box_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(box_path), canvas)

    transform_json = json.dumps(
        [
            float(M[0, 0]),
            float(M[0, 1]),
            float(M[0, 2]),
            float(M[1, 0]),
            float(M[1, 1]),
            float(M[1, 2]),
        ]
    )

    meta = {
        "instance_id": instance_id,
        "transform": [
            float(M[0, 0]),
            float(M[0, 1]),
            float(M[0, 2]),
            float(M[1, 0]),
            float(M[1, 1]),
            float(M[1, 2]),
        ],
        "crop_size": json.loads(crop_size_json),
    }

    return {
        "idx": idx,
        "iou": float(result["iou"]),
        "angle": float(result["angle"]),
        "crop_size_json": crop_size_json,
        "transform_json": transform_json,
        "meta": meta,
    }


def _get_stem(instance_id: str) -> str:
    m = re.match(r"^(.*)_\d{2}$", instance_id)
    return m.group(1) if m else instance_id


def _find_source_image(data_dir: Path, split: str, label: str, stem: str) -> Optional[Path]:
    search_dir = data_dir / split / label
    if not search_dir.exists():
        return None
    for ext in IMG_EXTS:
        candidate = search_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _get_reference_mask(
    paths: ProjectPaths,
    registry: pd.DataFrame,
    reference_id: Optional[str],
    out_size: tuple,
    scale: float,
    offset: float,
) -> Optional[np.ndarray]:
    if reference_id:
        ref_row = registry[registry["instance_id"] == reference_id]
        if ref_row.empty:
            return None
        ref_row = ref_row.iloc[0]
    else:
        train_good = registry[(registry["split"] == "train") & (registry["label"] == "good")]
        train_good = train_good[train_good["mask_area_px"] > 100]
        if train_good.empty:
            return None
        ref_row = train_good.iloc[0]

    mask_path = paths.mask(ref_row["split"], ref_row["label"], ref_row["instance_id"])
    if not mask_path.exists():
        return None

    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        return None
    mask01 = (mask_raw > 127).astype(np.uint8)

    if mask01.sum() < 10:
        return None

    mean, pca_angle, std_major, _ = _compute_pca(mask01)
    angle = pca_angle + offset

    _, ref_crop_mask, _, _ = _render_crop(None, mask01, mean, angle, std_major, out_size, scale, warp_img=False)
    return ref_crop_mask


def _aspect_fit_center_crop(img_rgb: np.ndarray, out_size: tuple) -> tuple[np.ndarray, np.ndarray]:
    out_w, out_h = out_size
    h, w = img_rgb.shape[:2]

    if h <= 0 or w <= 0:
        crop = np.zeros((out_h, out_w, 3), dtype=img_rgb.dtype)
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        return crop, M

    in_ar = w / float(h)
    out_ar = out_w / float(out_h)

    if in_ar < out_ar:
        s = out_w / float(w)
        new_h = h * s
        crop_y = (new_h - out_h) * 0.5
        tx = 0.0
        ty = -crop_y
    else:
        s = out_h / float(h)
        new_w = w * s
        crop_x = (new_w - out_w) * 0.5
        tx = -crop_x
        ty = 0.0

    M = np.array([[s, 0.0, tx], [0.0, s, ty]], dtype=np.float64)

    crop_img = cv2.warpAffine(
        img_rgb,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return crop_img, M


def _compute_crop(
    img_rgb: np.ndarray,
    mask01: np.ndarray,
    out_size: tuple,
    scale: float,
    offset: float,
    align_pca: bool,
    angle_candidates: list,
    ref_mask: Optional[np.ndarray],
) -> dict:
    out_w, out_h = out_size

    if mask01.sum() < 10:
        crop_img, M = _aspect_fit_center_crop(img_rgb, out_size)
        crop_mask = np.zeros((out_h, out_w), dtype=np.uint8)
        return {"crop_img": crop_img, "crop_mask": crop_mask, "M": M, "angle": 0.0, "iou": 0.0}

    mean, pca_angle, std_major, _ = _compute_pca(mask01)

    if not align_pca:
        angle = 0.0
    elif ref_mask is not None and len(angle_candidates) > 1:
        best_angle = pca_angle + offset
        best_iou = -1.0

        for cand in angle_candidates:
            test_angle = pca_angle + offset + float(cand)
            _, test_mask, _, _ = _render_crop(None, mask01, mean, test_angle, std_major, out_size, scale, warp_img=False)
            iou = _compute_iou(test_mask, ref_mask)
            if iou > best_iou:
                best_iou = iou
                best_angle = test_angle

        angle = best_angle
    else:
        angle = pca_angle + offset

    crop_img, crop_mask, M, _ = _render_crop(img_rgb, mask01, mean, angle, std_major, out_size, scale, warp_img=True)

    if ref_mask is not None:
        iou = _compute_iou(crop_mask, ref_mask)
    else:
        original_area = int(mask01.sum())
        crop_area = int(crop_mask.sum())
        iou = (crop_area / original_area) if original_area > 0 else 0.0

    return {"crop_img": crop_img, "crop_mask": crop_mask, "M": M, "angle": float(angle), "iou": float(iou)}


def _compute_pca(mask01: np.ndarray) -> tuple:
    y, x = np.where(mask01 > 0)

    if len(x) < 5:
        h, w = mask01.shape[:2]
        return np.array([w / 2, h / 2]), 0.0, max(w, h) / 4, min(w, h) / 4

    pts = np.stack([x, y], axis=1).astype(np.float32)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)

    angle = float(np.degrees(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])))
    std = np.sqrt(eigenvalues[:, 0])
    std_major = float(std[0])
    std_minor = float(std[1]) if len(std) > 1 else std_major * 0.5

    return mean[0], angle, std_major, std_minor


def _render_crop(
    img_rgb: Optional[np.ndarray],
    mask01: np.ndarray,
    mean: np.ndarray,
    angle: float,
    std_major: float,
    out_size: tuple,
    scale: float,
    warp_img: bool,
) -> tuple:
    out_w, out_h = out_size

    s = (out_w / max(1.0, std_major * 4.0)) * scale

    M = cv2.getRotationMatrix2D((float(mean[0]), float(mean[1])), float(angle), float(s))
    M[0, 2] += (out_w * 0.5 - float(mean[0]))
    M[1, 2] += (out_h * 0.5 - float(mean[1]))

    crop_img = None
    if warp_img:
        crop_img = cv2.warpAffine(
            img_rgb,
            M,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    crop_mask = cv2.warpAffine(
        mask01 * 255,
        M,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    crop_mask = (crop_mask > 127).astype(np.uint8)

    return crop_img, crop_mask, M, s


def _compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = a > 0
    b_bool = b > 0
    inter = np.logical_and(a_bool, b_bool).sum()
    union = np.logical_or(a_bool, b_bool).sum()
    return float(inter) / float(union) if union > 0 else 0.0
