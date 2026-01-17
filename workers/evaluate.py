import cv2
import torch
import numpy as np
import time
from pathlib import Path
from datetime import datetime
from PIL import Image as PILImage
from torchvision import transforms

from core.project import Project
from workers.base import BaseWorker


class EvaluateWorker(BaseWorker):
    def load_model(self):
        repo = "facebookresearch/dinov2"
        model_name = "dinov2_vitb14"

        print(f"[Evaluate] Loading {model_name}...")
        self.model = torch.hub.load(repo, model_name)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # Your GaussianBlur setting
            transforms.GaussianBlur(5, sigma=2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Cache State
        self.bank_B = None  # [K, H, W, C]
        self.bank_B2 = None  # [H, W, K]
        self.bank_handle = None
        self.last_bank_mtime = 0
        self.batch_counter = 0

    def process(self, project: Project) -> int:
        tfm_cfg = project.cfg.get("transformer", {})
        grid = tfm_cfg.get("output_resolution", [64, 64])
        Hp, Wp = grid[1], grid[0]

        token_budget = int(tfm_cfg.get("token_budget", 40000))

        # --- 1. SMART BANK LOADING ---
        bank_dir = project.root / "artifacts" / "bank"
        current_mtime = bank_dir.stat().st_mtime if bank_dir.exists() else 0

        if current_mtime > self.last_bank_mtime:
            project.load_bank()
            self.last_bank_mtime = current_mtime

        # Sync to GPU
        self._sync_bank_to_gpu(project, Hp, Wp)

        # --- 2. GET WORK ---
        tokens_per_img = Hp * Wp
        batch_size = max(1, token_budget // tokens_per_img)

        jobs = project.get_pending_evaluation(limit=batch_size)
        if not jobs: return 0

        crops_dir = project.root / "artifacts" / "crop_images"
        dir_embeds = project.root / "artifacts" / "embeddings"
        dir_bank = project.root / "artifacts" / "bank"
        dir_heatmaps = project.root / "artifacts" / "heatmaps"
        for d in [dir_embeds, dir_bank, dir_heatmaps]: d.mkdir(parents=True, exist_ok=True)

        valid_jobs = []
        images_pt = []

        for inst in jobs:
            img_path = crops_dir / f"{inst.id}.png"
            if not img_path.exists(): continue
            try:
                with PILImage.open(img_path) as im:
                    im = im.convert("RGB")
                    images_pt.append(self.transform(im))
                valid_jobs.append(inst)
            except:
                pass

        if not valid_jobs: return 0

        t0 = time.perf_counter()

        # --- 3. INFERENCE ---
        x = torch.stack(images_pt).to(self.device)
        N = x.shape[0]

        with torch.inference_mode():
            with torch.autocast(device_type=self.device, enabled=(self.device == "cuda"), dtype=torch.float16):
                out = self.model.forward_features(x)
                tokens = out["x_norm_patchtokens"] if isinstance(out, dict) else out[0]

                # Reshape to [N, Hp, Wp, C]
                C = tokens.shape[-1]
                feats = tokens.reshape(N, Hp, Wp, C).contiguous().float()

        t1 = time.perf_counter()

        # --- 4. SCORING ---
        pooled_np = feats.mean(dim=(1, 2)).cpu().numpy().astype(np.float16)

        t_math_start = time.perf_counter()

        scores = []
        heatmaps = []

        if self.bank_B is not None:
            # 4D Math Optimized
            x2 = (feats * feats).sum(-1, keepdim=True)
            xb = torch.einsum('nhwc,khwc->nhwk', feats, self.bank_B)

            # Distance Squared: [N, Hp, Wp, K]
            d2 = x2 + self.bank_B2.unsqueeze(0) - 2.0 * xb

            # Min across K (Bank Images) -> [N, Hp, Wp]
            min_dists_sq = d2.min(dim=3).values
            min_dists_sq = torch.clamp(min_dists_sq, min=0.0)
            amaps = torch.sqrt(min_dists_sq)

            for i in range(N):
                amap = amaps[i]

                # Score Calculation
                flat_map = amap.view(-1)
                sorted_dists, _ = torch.sort(flat_map)
                n_pixels = flat_map.shape[0]
                k = int(n_pixels * 0.99)
                subset = sorted_dists[k:]
                score_val = torch.mean(subset).item() if subset.numel() > 0 else sorted_dists[-1].item()
                scores.append(score_val)

                # Heatmap Generation
                dist_map_np = amap.cpu().numpy()
                mean_val = np.mean(dist_map_np)
                dist_map_np = np.maximum(dist_map_np - mean_val, 0)
                norm_map = cv2.normalize(dist_map_np, None, 0, 255, cv2.NORM_MINMAX)
                heatmap_img = cv2.applyColorMap(norm_map.astype(np.uint8), cv2.COLORMAP_MAGMA)
                heatmaps.append(heatmap_img)
        else:
            scores = [0.0] * N
            heatmaps = [None] * N

        t_math_end = time.perf_counter()

        # --- 5. IO ---
        feats_flat = feats.reshape(N, -1, C).cpu().numpy().astype(np.float16)

        count = 0
        with project.get_session() as session:
            for i, inst in enumerate(valid_jobs):
                try:
                    self._save_atomic(pooled_np[i], dir_embeds / f"{inst.id}.npy")

                    if inst.in_bank:
                        self._save_atomic(feats_flat[i], dir_bank / f"{inst.id}.npy")

                    if heatmaps[i] is not None:
                        cv2.imwrite(str(dir_heatmaps / f"{inst.id}.png"), heatmaps[i])

                    inst.score = float(scores[i])
                    inst.evaluated_at = datetime.utcnow()
                    session.merge(inst)
                    count += 1
                except Exception as e:
                    print(f"IO Error {inst.id}: {e}")
            session.commit()

        # --- 6. PROFILE ---
        self.batch_counter += 1
        if self.batch_counter % 5 == 0:
            ms_infer = (t1 - t0) * 1000
            ms_math = (t_math_end - t_math_start) * 1000

            # Stats
            bank_sz = self.bank_B.shape[0] if self.bank_B is not None else 0

            if self.device == "cuda":
                free, total = torch.cuda.mem_get_info()
                alloc = torch.cuda.memory_allocated()
                print(
                    f"[Profile] Batch: {N} | Infer: {ms_infer:.0f}ms | Math: {ms_math:.0f}ms | Bank: {bank_sz} | Mem: {alloc / 1e9:.1f}G / {free / 1e9:.1f}G")

        return count

    def _sync_bank_to_gpu(self, project, Hp, Wp):
        if project.bank_matrix is None:
            self.bank_B = None;
            self.bank_B2 = None;
            self.bank_handle = None
            return

        if id(project.bank_matrix) == self.bank_handle: return

        # Load & Clean
        raw_bank = project.bank_matrix.astype(np.float32)
        patches_per_image = Hp * Wp

        if raw_bank.shape[0] % patches_per_image != 0:
            valid = (raw_bank.shape[0] // patches_per_image) * patches_per_image
            raw_bank = raw_bank[:valid]

        n_images = raw_bank.shape[0] // patches_per_image
        if n_images == 0: return

        # Reshape to [K, Hp*Wp, C] -> [K, Hp, Wp, C]
        reshaped = raw_bank.reshape(n_images, patches_per_image, -1)
        B_numpy = reshaped.reshape(n_images, Hp, Wp, -1)

        # To GPU
        self.bank_B = torch.from_numpy(B_numpy).to(self.device).contiguous()

        # Pre-calculate Norms [K, Hp, Wp] -> [Hp, Wp, K]
        B2 = (self.bank_B * self.bank_B).sum(dim=-1)
        self.bank_B2 = B2.permute(1, 2, 0).contiguous()

        self.bank_handle = id(project.bank_matrix)
        print(f"[Evaluate] Bank Synced [K,H,W,C]: {self.bank_B.shape}")

    def _save_atomic(self, arr, path):
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, arr)
        try:
            tmp.replace(path)
        except:
            if path.exists(): path.unlink()
            tmp.rename(path)