import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict

from core.project import Project
from core.database import Instance, Image
from workers.base import BaseWorker


class CropWorker(BaseWorker):
    def __init__(self):
        super().__init__()
        self._ref_cache: Dict[str, dict] = {}

    def load_model(self):
        pass

    def process(self, project: Project) -> int:
        crop_cfg = project.cfg.get("crop", {})
        trans_cfg = project.cfg.get("transformer", {})

        if not crop_cfg.get("enabled", True):
            return 0

        patch_size = int(trans_cfg.get("patch_size", 14))
        grid = trans_cfg.get("output_resolution", [64, 64])
        out_wh = (int(grid[0]) * patch_size, int(grid[1]) * patch_size)

        jobs = project.get_pending_crops(limit=32)
        if not jobs:
            return 0

        ref_data = self._get_reference_data(project, crop_cfg, out_wh)

        count = 0
        with project.get_session() as session:
            for inst in jobs:
                try:
                    if inst.image:
                        img_obj = inst.image
                    else:
                        img_obj = session.get(Image, inst.image_id)

                    raw_path = project.get_raw_path(img_obj)
                    mask_path = project.root / "artifacts" / "masks" / f"{inst.id}.png"

                    if not raw_path.exists() or not mask_path.exists():
                        continue

                    img_bgr = cv2.imread(str(raw_path))
                    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

                    if img_bgr is None or mask_raw is None:
                        continue

                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    mask01 = (mask_raw > 127).astype(np.uint8)

                    # A. Calculate PCA
                    mean, angle, std = self._calculate_pca(mask01)

                    # B. Determine Best Rotation
                    base_angle = angle
                    best_angle = base_angle

                    final_iou = None

                    if ref_data:
                        best_iou = -1.0
                        candidates = crop_cfg.get("angle_candidates", [0, 180])
                        offset = crop_cfg.get("offset_rotation", 0.0)

                        for cand in candidates:
                            test_angle = base_angle + offset + float(cand)
                            _, crop_m = self._render_crop(
                                img_rgb, mask01, mean, test_angle, std[0], out_wh,
                                scale=crop_cfg.get("scale", 1.0)
                            )

                            iou = self._calculate_iou(crop_m, ref_data['mask'])
                            if iou > best_iou:
                                best_iou = iou
                                best_angle = test_angle

                        # Only set metric if we actually calculated it
                        if best_iou >= 0:
                            final_iou = float(best_iou)
                    else:
                        best_angle = base_angle + crop_cfg.get("offset_rotation", 0.0)

                    # C. Render
                    final_img, final_mask = self._render_crop(
                        img_rgb, mask01, mean, best_angle, std[0], out_wh,
                        scale=crop_cfg.get("scale", 1.0)
                    )

                    # D. Save
                    out_img_path = project.root / "artifacts" / "crop_images" / f"{inst.id}.png"
                    out_msk_path = project.root / "artifacts" / "crop_masks" / f"{inst.id}.png"

                    self._save_atomic(final_img, out_img_path, is_mask=False)
                    self._save_atomic(final_mask, out_msk_path, is_mask=True)

                    # E. Update DB
                    obb_data = {
                        "cx": float(mean[0]), "cy": float(mean[1]),
                        "angle": float(angle),
                        "w": float(std[0] * 4), "h": float(std[1] * 4)
                    }
                    inst.obb_json = json.dumps(obb_data)
                    inst.cropped_at = datetime.utcnow()
                    inst.crop_iou = final_iou  # <--- Metric Saved Here

                    session.add(inst)
                    count += 1

                except Exception as e:
                    print(f"[Crop] Error processing {inst.id}: {e}")

            session.commit()
        return count

    # --- HELPERS ---

    def _get_reference_data(self, project: Project, crop_cfg: dict, out_wh: Tuple[int, int]) -> Optional[dict]:
        ref_id = crop_cfg.get("reference_id", "").strip()
        proj_name = project.root.name

        cached = self._ref_cache.get(proj_name)
        if cached and cached['id'] == ref_id:
            return cached

        if not ref_id: return None

        mask_path = project.root / "artifacts" / "masks" / f"{ref_id}.png"
        if not mask_path.exists(): return None

        try:
            mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask01 = (mask_raw > 127).astype(np.uint8)
            mean, angle, std = self._calculate_pca(mask01)

            # Positive angle + offset (Corrected Logic)
            base_angle = angle + crop_cfg.get("offset_rotation", 0.0)

            dummy_img = np.zeros((mask01.shape[0], mask01.shape[1], 3), dtype=np.uint8)

            _, ref_crop_mask = self._render_crop(
                dummy_img, mask01, mean, base_angle, std[0], out_wh,
                scale=crop_cfg.get("scale", 1.0)
            )

            data = {'id': ref_id, 'mask': ref_crop_mask}
            self._ref_cache[proj_name] = data
            return data

        except Exception as e:
            print(f"[Crop] Failed to load reference {ref_id}: {e}")
            return None

    def _calculate_pca(self, mask: np.ndarray):
        y, x = np.where(mask > 0)
        if len(x) < 5:
            h, w = mask.shape[:2]
            return np.array([w / 2, h / 2]), 0.0, np.array([w / 4, h / 4])

        pts = np.stack([x, y], axis=1).astype(np.float32)
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)

        center = mean[0]
        std_major = np.sqrt(eigenvalues[0, 0])
        std_minor = np.sqrt(eigenvalues[1, 0])

        vx, vy = eigenvectors[0, 0], eigenvectors[0, 1]
        angle = np.degrees(np.arctan2(vy, vx))

        return center, angle, np.array([std_major, std_minor])

    def _render_crop(self, img: np.ndarray, mask: np.ndarray,
                     mean: np.ndarray, angle: float, std_major: float,
                     out_wh: Tuple[int, int], scale: float = 1.0):

        out_w, out_h = out_wh
        out_cx, out_cy = out_w * 0.5, out_h * 0.5

        obj_width = max(1.0, std_major * 4.0)
        s = (out_w / obj_width) * scale

        M = cv2.getRotationMatrix2D((mean[0], mean[1]), angle, s)
        M[0, 2] += (out_cx - mean[0])
        M[1, 2] += (out_cy - mean[1])

        img_warp = cv2.warpAffine(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

        mask_warp_raw = cv2.warpAffine((mask * 255).astype(np.uint8), M, (out_w, out_h),
                                       flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask_warp = (mask_warp_raw > 127).astype(np.uint8)

        return img_warp, mask_warp

    def _calculate_iou(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        inter = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        if union == 0: return 0.0
        return float(inter) / float(union)

    def _save_atomic(self, img: np.ndarray, path: Path, is_mask: bool):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp.png")
        if is_mask:
            cv2.imwrite(str(tmp_path), img * 255)
        else:
            cv2.imwrite(str(tmp_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        try:
            tmp_path.replace(path)
        except OSError:
            if path.exists(): path.unlink()
            tmp_path.rename(path)
