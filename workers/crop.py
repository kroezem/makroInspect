import json
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.project import Project
from core.database import Instance, Image
from core.scheduler import Job
from core.telemetry import WorkerStats
from workers.base import BaseWorker


class CropWorker(BaseWorker):
    """
    Crop worker - performs PCA-aligned cropping of segmented instances.

    Handles blank/full masks by defaulting to center crop.
    Purely functional: returns WorkerStats, no print statements.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.executor = ThreadPoolExecutor(max_workers=8)
        self._ref_cache = {}

    def process(self, projects: List[Project], job: Job, models=None) -> WorkerStats:
        """
        Execute a single atomic cropping batch.
        """
        start_time = time.perf_counter()
        stats = WorkerStats()

        active_projects = self._aggregate_work(projects, job)
        if not active_projects:
            return stats

        # Set project name from first active project
        first_proj = next(iter(active_projects.keys()), None)
        if first_proj:
            stats.project_name = first_proj.root.name

        count = 0
        errors = 0

        for proj, images_dict in active_projects.items():
            trans_cfg = proj.cfg.get("transformer", {})
            crop_cfg = proj.cfg.get("crop", {})

            grid = trans_cfg.get("map_resolution", [48, 48])
            patch_size = trans_cfg.get("patch_size", 14)
            out_wh = (int(grid[0]) * patch_size, int(grid[1]) * patch_size)

            ref_data = self._get_ref_data(proj, crop_cfg, out_wh)

            with proj.get_session() as session:
                for img_id, inst_list in images_dict.items():
                    img_obj = session.get(Image, img_id)
                    if not img_obj:
                        continue

                    raw_path = proj.get_raw_path(img_obj)
                    if not raw_path.exists():
                        continue

                    img_bgr = cv2.imread(str(raw_path))
                    if img_bgr is None:
                        continue
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    futures = {}
                    for inst in inst_list:
                        mask_path = proj.root / "artifacts" / "masks" / f"{inst.id}.png"
                        futures[self.executor.submit(
                            self._worker_thread,
                            inst_id=inst.id,
                            img_rgb=img_rgb,
                            mask_path=mask_path,
                            proj_root=proj.root,
                            crop_cfg=crop_cfg,
                            out_wh=out_wh,
                            ref_data=ref_data
                        )] = inst

                    for future in as_completed(futures):
                        inst = futures[future]
                        try:
                            res = future.result()
                            if res:
                                inst.obb_json = res['obb']
                                inst.crop_iou = res['iou']
                                inst.cropped_at = datetime.utcnow()
                                session.merge(inst)
                                count += 1
                        except Exception:
                            errors += 1

                session.commit()

        stats.items_processed = count
        stats.errors = errors
        stats.duration_ms = (time.perf_counter() - start_time) * 1000
        return stats

    def _aggregate_work(self, projects: List[Project], job: Job) -> dict:
        """Aggregate pending crop work grouped by project and parent image."""
        all_jobs_by_project = defaultdict(list)

        for proj in projects:
            jobs = proj.get_pending_crops(min_priority=job.min_priority, limit=job.batch_limit * 20)
            if jobs:
                all_jobs_by_project[proj].extend(jobs)

        if not all_jobs_by_project:
            return {}

        flat_jobs = []
        for proj, jobs in all_jobs_by_project.items():
            for j in jobs:
                if j.priority <= job.max_priority:
                    flat_jobs.append((proj, j))

        flat_jobs.sort(key=lambda x: (-x[1].priority, x[1].created_at))

        active_projects = defaultdict(lambda: defaultdict(list))
        unique_images_processed = 0
        seen_images = set()

        for proj, inst in flat_jobs:
            img_key = (proj.root.name, inst.image_id)

            if img_key not in seen_images:
                if unique_images_processed >= job.batch_limit:
                    continue
                unique_images_processed += 1
                seen_images.add(img_key)

            if img_key in seen_images:
                active_projects[proj][inst.image_id].append(inst)

        return active_projects

    def _worker_thread(self, inst_id, img_rgb, mask_path, proj_root, crop_cfg, out_wh, ref_data):
        """Logic for a single crop. Handles blank/full masks with center crop fallback."""
        if not mask_path.exists():
            return None

        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            return None
        mask01 = (mask_raw > 127).astype(np.uint8)

        h, w = mask01.shape[:2]

        non_zeros = np.count_nonzero(mask01)
        is_empty = non_zeros == 0
        is_full = (non_zeros / (w * h)) > 0.99

        if is_empty or is_full:
            mean = np.array([w / 2, h / 2], dtype=np.float32)
            angle = 0.0
            std = np.array([w / 4, h / 4], dtype=np.float32)
        else:
            mean, angle, std = self._calculate_pca(mask01)

        base_angle = angle
        best_angle = base_angle
        final_iou = None
        scale = crop_cfg.get("scale", 1.0)

        if ref_data and not (is_empty or is_full):
            best_iou = -1.0
            candidates = crop_cfg.get("angle_candidates", [0, 180])
            offset = crop_cfg.get("offset_rotation", 0.0)

            for cand in candidates:
                test_angle = base_angle + offset + float(cand)
                _, test_m = self._render_crop(img_rgb, mask01, mean, test_angle, std[0], out_wh, scale)
                iou = self._calc_iou(test_m, ref_data['mask'])
                if iou > best_iou:
                    best_iou = iou
                    best_angle = test_angle

            if best_iou >= 0:
                final_iou = float(best_iou)
        else:
            best_angle = base_angle + crop_cfg.get("offset_rotation", 0.0)

        final_img, final_mask = self._render_crop(img_rgb, mask01, mean, best_angle, std[0], out_wh, scale)

        out_img_path = proj_root / "artifacts" / "crop_images" / f"{inst_id}.png"
        out_msk_path = proj_root / "artifacts" / "crop_masks" / f"{inst_id}.png"

        cv2.imwrite(str(out_img_path), cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_msk_path), final_mask * 255)

        return {
            "obb": json.dumps({
                "cx": float(mean[0]), "cy": float(mean[1]),
                "angle": float(angle),
                "w": float(std[0] * 4), "h": float(std[1] * 4)
            }),
            "iou": final_iou
        }

    def _get_ref_data(self, project, crop_cfg, out_wh):
        """Load reference mask for alignment."""
        ref_id = crop_cfg.get("reference_id", "").strip()
        proj_name = project.root.name

        if proj_name in self._ref_cache and self._ref_cache[proj_name]['id'] == ref_id:
            return self._ref_cache[proj_name]

        if not ref_id:
            return None
        mask_path = project.root / "artifacts" / "masks" / f"{ref_id}.png"
        if not mask_path.exists():
            return None

        try:
            mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask01 = (mask_raw > 127).astype(np.uint8)

            non_zeros = np.count_nonzero(mask01)
            h, w = mask01.shape[:2]
            is_full_or_empty = non_zeros == 0 or (non_zeros / (w * h)) > 0.99

            if is_full_or_empty:
                return None

            mean, angle, std = self._calculate_pca(mask01)
            base_angle = angle + crop_cfg.get("offset_rotation", 0.0)
            dummy = np.zeros((mask01.shape[0], mask01.shape[1], 3), dtype=np.uint8)
            _, ref_m = self._render_crop(dummy, mask01, mean, base_angle, std[0], out_wh, crop_cfg.get("scale", 1.0))

            data = {'id': ref_id, 'mask': ref_m}
            self._ref_cache[proj_name] = data
            return data
        except:
            return None

    def _calculate_pca(self, mask):
        """Calculate PCA for mask alignment."""
        y, x = np.where(mask > 0)
        if len(x) < 5:
            h, w = mask.shape[:2]
            return np.array([w / 2, h / 2]), 0.0, np.array([w / 4, h / 4])

        pts = np.stack([x, y], axis=1).astype(np.float32)
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)

        angle = np.degrees(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]))
        std = np.sqrt(eigenvalues[:, 0])

        return mean[0], angle, std

    def _render_crop(self, img, mask, mean, angle, std_major, out_wh, scale=1.0):
        """Render aligned crop."""
        out_w, out_h = out_wh
        s = (out_w / max(1.0, std_major * 4.0)) * scale

        M = cv2.getRotationMatrix2D((mean[0], mean[1]), angle, s)
        M[0, 2] += (out_w * 0.5 - mean[0])
        M[1, 2] += (out_h * 0.5 - mean[1])

        img_w = cv2.warpAffine(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        mask_w = cv2.warpAffine((mask * 255).astype(np.uint8), M, (out_w, out_h),
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        return img_w, (mask_w > 127).astype(np.uint8)

    def _calc_iou(self, a, b):
        """Calculate IoU between two masks."""
        i = np.logical_and(a, b).sum()
        u = np.logical_or(a, b).sum()
        return float(i) / u if u > 0 else 0.0
