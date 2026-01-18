import time
import cv2
import torch
import numpy as np
from datetime import datetime
from typing import List

from core.project import Project
from core.scheduler import Job
from core.telemetry import WorkerStats
from workers.base import BaseWorker


class EvaluateWorker(BaseWorker):
    """
    Evaluate worker - computes DINOv2 features and anomaly scores.

    Purely functional: returns WorkerStats, no print statements.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.bank_B = None
        self.bank_B2 = None
        self.bank_handle = None

    def process(self, projects: List[Project], job: Job, models=None) -> WorkerStats:
        """Execute a single atomic evaluation batch."""
        start_time = time.perf_counter()
        stats = WorkerStats()

        best_project = self._select_project(projects, job)
        if not best_project:
            return stats

        stats.project_name = best_project.root.name

        batch_limit = self._get_budgeted_batch_limit(best_project)
        final_limit = min(batch_limit, job.batch_limit)

        jobs = best_project.get_pending_evaluation(min_priority=job.min_priority, limit=final_limit)
        if not jobs:
            return stats

        valid_jobs, paths = self._prepare_batch(best_project, jobs)
        if not valid_jobs:
            return stats

        self._sync_bank_to_gpu(best_project)

        feats = models.compute_patches(paths)
        if feats is None:
            return stats

        if "cuda" in self.device:
            torch.cuda.synchronize()

        score_cfg = best_project.cfg.get("anomaly_score", {})

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            scores_np, amaps_np = self._batch_score_gpu(feats, score_cfg)

        pooled_np = feats.mean(dim=(1, 2)).cpu().numpy().astype(np.float16)

        if "cuda" in self.device:
            torch.cuda.synchronize()

        stats.items_processed = self._persist_results(
            best_project, valid_jobs, pooled_np, scores_np, amaps_np
        )
        stats.duration_ms = (time.perf_counter() - start_time) * 1000

        return stats

    def _select_project(self, projects: List[Project], job: Job):
        """Select project with highest priority pending work."""
        best_project = None
        best_priority = -1

        for proj in projects:
            pending = proj.get_pending_evaluation(min_priority=job.min_priority, limit=1)
            if pending:
                prio = pending[0].priority
                if prio > best_priority and prio <= job.max_priority:
                    best_priority = prio
                    best_project = proj

        return best_project

    def _prepare_batch(self, project: Project, jobs: List):
        """Prepare batch of crop images for evaluation."""
        crops_dir = project.root / "artifacts" / "crop_images"
        valid_jobs, paths = [], []

        for inst in jobs:
            p = crops_dir / f"{inst.id}.png"
            if p.exists():
                valid_jobs.append(inst)
                paths.append(p)

        return valid_jobs, paths

    def _get_budgeted_batch_limit(self, project: Project) -> int:
        """Calculate batch limit based on token budget."""
        trans_cfg = project.cfg.get("transformer", {})
        budget = int(trans_cfg.get("token_budget", 150000))
        grid = trans_cfg.get("map_resolution", [48, 32])
        tokens_per_img = grid[0] * grid[1]
        if tokens_per_img == 0:
            return 1
        return max(1, min(int(budget // tokens_per_img), 256))

    def _sync_bank_to_gpu(self, project: Project):
        """Sync bank matrix to GPU."""
        matrix = project.bank_matrix
        if matrix is None:
            self.bank_B, self.bank_B2, self.bank_handle = None, None, None
            return

        current_handle = id(matrix)
        if current_handle == self.bank_handle:
            return

        raw = matrix.astype(np.float16)
        b_tensor = torch.from_numpy(raw).to(self.device).contiguous()
        self.bank_B = b_tensor
        self.bank_B2 = (self.bank_B.float() * self.bank_B.float()).sum(dim=-1).half()
        self.bank_handle = current_handle

    def _batch_score_gpu(self, feats, score_cfg: dict = None):
        """
        Compute anomaly scores using GPU BMM.

        Uses config for scoring method:
        - method: "mean", "max"
        - min_percentile: lower bound (e.g., 99.0)
        - max_percentile: upper bound (e.g., 99.9)
        """
        if self.bank_B is None:
            N, H, W, _ = feats.shape
            return np.zeros(N), np.zeros((N, H, W))

        N, H, W, C = feats.shape
        K = self.bank_B.shape[0]

        x = feats.half()

        f_flat = x.permute(1, 2, 0, 3).reshape(H * W, N, C)
        b_flat = self.bank_B.permute(1, 2, 3, 0).reshape(H * W, C, K)

        xb_flat = torch.bmm(f_flat, b_flat)
        xb = xb_flat.reshape(H, W, N, K).permute(2, 0, 1, 3)

        x2 = (x.float() * x.float()).sum(dim=-1, keepdim=True).half()
        y2 = self.bank_B2.permute(1, 2, 0).unsqueeze(0)

        d2 = x2 + y2 - 2.0 * xb

        amaps = torch.sqrt(torch.clamp(d2.min(dim=-1).values, min=0.0))

        scores = self._compute_scores_from_amaps(amaps, score_cfg)

        return scores.float().cpu().numpy(), amaps.float().cpu().numpy()

    def _compute_scores_from_amaps(self, amaps: torch.Tensor, score_cfg: dict = None) -> torch.Tensor:
        """
        Compute anomaly scores from anomaly maps using config settings.

        Config options:
        - method: "mean" (default), "max"
        - min_percentile: 99.0 (use pixels from this percentile)
        - max_percentile: 99.9 (up to this percentile)

        Example: min=99.0, max=99.9 means take pixels between 99th and 99.9th percentile.
        """
        if score_cfg is None:
            score_cfg = {}

        method = score_cfg.get("method", "mean")
        min_pct = score_cfg.get("min_percentile", 99.0)
        max_pct = score_cfg.get("max_percentile", 99.9)

        N = amaps.shape[0]
        flat_amaps = amaps.view(N, -1)
        num_pixels = flat_amaps.shape[1]

        if method == "max":
            return flat_amaps.max(dim=1).values

        # Percentile-bounded scoring
        sorted_vals, _ = torch.sort(flat_amaps, dim=1)

        min_idx = int(num_pixels * (min_pct / 100.0))
        max_idx = int(num_pixels * (max_pct / 100.0))

        min_idx = max(0, min(min_idx, num_pixels - 1))
        max_idx = max(min_idx + 1, min(max_idx, num_pixels))

        percentile_slice = sorted_vals[:, min_idx:max_idx]

        if method == "mean":
            return percentile_slice.mean(dim=1)
        else:
            return percentile_slice.mean(dim=1)

    def _persist_results(self, project: Project, valid_jobs, pooled_np, scores_np, amaps_np) -> int:
        """Save evaluation results."""
        dir_embeds = project.root / "artifacts" / "embeddings"
        dir_heatmaps = project.root / "artifacts" / "heatmaps"
        score_cfg = project.cfg.get("anomaly_score", {})
        curr_bank_uuid = project.current_bank_id

        count = 0

        with project.get_session() as session:
            for i, inst in enumerate(valid_jobs):
                db_inst = session.merge(inst)

                self._save_atomic(pooled_np[i], dir_embeds / f"{db_inst.id}.npy")

                heatmap = self._render_heatmap(amaps_np[i], score_cfg)
                if heatmap is not None:
                    cv2.imwrite(str(dir_heatmaps / f"{db_inst.id}.png"), heatmap)

                db_inst.score = float(scores_np[i])
                db_inst.evaluated_at = datetime.utcnow()
                db_inst.bank_id = curr_bank_uuid
                db_inst.priority = 1  # Move to maintenance lane after evaluation

                session.add(db_inst)
                count += 1

            session.commit()

        return count

    def _render_heatmap(self, amap_np, score_cfg):
        """
        Render anomaly heatmap.

        Uses per-image mean subtraction for visualization.
        Score is calculated separately using percentile config.
        """
        cmap_name = str(score_cfg.get("cmap", "MAGMA")).upper()
        target_cmap = getattr(cv2, f"COLORMAP_{cmap_name}", cv2.COLORMAP_MAGMA)

        # Use per-image mean for heatmap visualization
        arr = np.maximum(amap_np - np.mean(amap_np), 0)
        norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)

        return cv2.applyColorMap(norm.astype(np.uint8), target_cmap)

    def _save_atomic(self, arr, path):
        """Atomic save with temp file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, arr)
        try:
            tmp.replace(path)
        except:
            if path.exists():
                path.unlink()
            tmp.rename(path)
