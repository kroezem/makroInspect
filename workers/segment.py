import time
import torch
import numpy as np
import cv2
from PIL import Image as PILImage
from typing import List
from datetime import datetime

from core.project import Project
from core.database import Instance
from core.scheduler import Job
from core.telemetry import WorkerStats
from workers.base import BaseWorker


class SegmentWorker(BaseWorker):
    """
    Segment worker - runs SAM3 segmentation on images.

    Purely functional: returns WorkerStats, no print statements.
    """

    def process(self, projects: List[Project], job: Job, models=None) -> WorkerStats:
        """
        Execute a single atomic segmentation batch.
        """
        start_time = time.perf_counter()
        stats = WorkerStats()

        all_jobs = self._aggregate_work(projects, job)
        if not all_jobs:
            return stats

        batch = all_jobs[:job.batch_limit]

        # Set project name from first item in batch
        if batch:
            stats.project_name = batch[0][0].root.name

        valid_batch, pil_images, prompts = self._prepare_batch(batch)
        if not valid_batch:
            return stats

        results_map = self._run_inference(valid_batch, pil_images, prompts, models)

        stats.items_processed = self._persist_results(valid_batch, results_map)
        stats.duration_ms = (time.perf_counter() - start_time) * 1000

        return stats

    def _aggregate_work(self, projects: List[Project], job: Job) -> List:
        """Aggregate pending segmentation work across projects."""
        all_jobs = []
        for proj in projects:
            pending = proj.get_pending_images(min_priority=job.min_priority, limit=job.batch_limit * 2)
            for img in pending:
                if img.priority <= job.max_priority:
                    all_jobs.append((proj, img))

        all_jobs.sort(key=lambda x: (-x[1].priority, x[1].ingested_at))
        return all_jobs

    def _prepare_batch(self, batch: List) -> tuple:
        """Prepare images for inference."""
        pil_images = []
        prompts = []
        valid_batch = []

        for proj, img in batch:
            raw_path = proj.get_raw_path(img)
            if not raw_path.exists():
                continue

            try:
                seg_cfg = proj.cfg.get("segmentation", {})
                is_enabled = seg_cfg.get("enabled", True)
                prompt_text = seg_cfg.get("description", "object")

                p_img = PILImage.open(raw_path).convert("RGB")

                valid_batch.append({
                    "proj": proj,
                    "img": img,
                    "pil": p_img,
                    "enabled": is_enabled,
                    "prompt": prompt_text
                })

                if is_enabled:
                    pil_images.append(p_img)
                    prompts.append(prompt_text)

            except Exception:
                pass

        return valid_batch, pil_images, prompts

    def _run_inference(self, valid_batch: List, pil_images: List, prompts: List, models) -> dict:
        """Run SAM3 inference on enabled images."""
        results_map = {}

        if not pil_images:
            return results_map

        model_results = models.segment_batch(pil_images, prompt=prompts)
        if "cuda" in self.device:
            torch.cuda.synchronize()

        curr_model_idx = 0
        for i, item in enumerate(valid_batch):
            if item["enabled"]:
                results_map[i] = model_results[curr_model_idx]
                curr_model_idx += 1

        return results_map

    def _persist_results(self, valid_batch: List, results_map: dict) -> int:
        """Save segmentation results to DB and disk."""
        count = 0

        for i, item in enumerate(valid_batch):
            proj, img = item["proj"], item["img"]

            masks = self._extract_masks(item, results_map.get(i))

            multi_allowed = proj.cfg.get("segmentation", {}).get("multi_instance", True)
            if not multi_allowed and len(masks) > 1:
                masks = masks[:1]

            with proj.get_session() as session:
                db_img = session.merge(img)

                for m_idx, mask in enumerate(masks):
                    inst_id = f"{db_img.id:09d}_{m_idx:02d}"
                    mask_path = proj.root / "artifacts" / "masks" / f"{inst_id}.png"

                    if mask.max() <= 1.0:
                        mask = mask * 255
                    cv2.imwrite(str(mask_path), mask.astype(np.uint8))

                    # Determine label from station_id (special inbox folders)
                    label = "unlabeled"
                    station = db_img.station_id
                    SPECIAL_STATIONS = {'_normal': 'normal', '_anomaly': 'anomaly', '_exclude': 'exclude'}
                    if station in SPECIAL_STATIONS:
                        label = SPECIAL_STATIONS[station]

                    instance = Instance(
                        id=inst_id,
                        image_id=db_img.id,
                        station_id=db_img.station_id,
                        priority=db_img.priority,
                        label=label
                    )
                    session.merge(instance)

                db_img.segmented_at = datetime.utcnow()
                session.add(db_img)
                session.commit()
                count += 1

        return count

    def _extract_masks(self, item: dict, result) -> List[np.ndarray]:
        """Extract masks from inference result or create fallback."""
        masks = []

        if item["enabled"] and result:
            masks = result.get('masks', [])
            if isinstance(masks, torch.Tensor):
                masks = masks.detach().cpu().numpy()

        if not item["enabled"] or len(masks) == 0:
            w, h = item["pil"].size
            masks = [np.ones((h, w), dtype=np.uint8)]

        return masks
