import torch
import numpy as np
from pathlib import Path
from PIL import Image as PILImage
from datetime import datetime
from transformers import Sam3Processor, Sam3Model

from core.project import Project
from core.database import Instance
from workers.base import BaseWorker


class SegmentWorker(BaseWorker):
    def load_model(self):
        model_id = "facebook/sam3"
        print(f"[Segment] Loading SAM3 ({model_id}) to {self.device}...")

        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def process(self, project: Project) -> int:
        seg_cfg = project.cfg.get("segmentation", {})
        if not seg_cfg.get("enabled", True):
            return 0

        prompt = seg_cfg.get("description", "object")
        batch_size = int(seg_cfg.get("batch_size", 4))
        multi_instance = bool(seg_cfg.get("multi_instance", True))

        jobs = project.get_pending_segmentation(limit=batch_size)
        if not jobs:
            return 0

        images_pil = []
        valid_jobs = []

        for img_row in jobs:
            path = project.get_raw_path(img_row)
            try:
                pil = PILImage.open(path).convert("RGB")
                images_pil.append(pil)
                valid_jobs.append(img_row)
            except Exception as e:
                print(f"[Segment] Read failed {path}: {e}")

        if not valid_jobs:
            return 0

        # Inference
        inputs = self.processor(
            images=images_pil,
            text=[prompt] * len(images_pil),
            return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

            target_sizes = [img.size[::-1] for img in images_pil]
            results = self.processor.post_process_instance_segmentation(
                outputs,
                target_sizes=target_sizes,
                threshold=0.5
            )

        # Save Results
        count = 0
        with project.get_session() as session:
            for i, result in enumerate(results):
                job = valid_jobs[i]
                masks = self._extract_masks_numpy(result)

                if masks is not None and masks.size > 0:
                    if not multi_instance:
                        union_mask = np.logical_or.reduce(masks, axis=0)
                        masks = union_mask[None, :, :]

                    for idx, mask in enumerate(masks):
                        instance_id = f"{job.id:09d}_{idx:02d}"

                        # Save Mask
                        mask_path = project.root / "artifacts" / "masks" / f"{instance_id}.png"
                        self._save_atomic_mask(mask, mask_path)

                        # Create Instance
                        inst = Instance(
                            id=instance_id,
                            image_id=job.id,
                            station_id=job.station_id,
                            label="unlabeled"
                        )
                        session.add(inst)
                        count += 1

                # Mark Done (Update timestamp)
                job.segmented_at = datetime.utcnow()
                session.merge(job)

            session.commit()

        return len(valid_jobs)

    def _save_atomic_mask(self, mask: np.ndarray, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = path.with_suffix(".tmp.png")
        img = PILImage.fromarray((mask * 255).astype(np.uint8), mode="L")
        img.save(tmp_path, optimize=True)

        try:
            tmp_path.replace(path)
        except OSError:
            if path.exists(): path.unlink()
            tmp_path.rename(path)

    def _extract_masks_numpy(self, result) -> np.ndarray:
        if isinstance(result, dict) and 'masks' in result:
            m = result['masks']
            if torch.is_tensor(m):
                m = m.detach().cpu().numpy()
            return m.astype(bool)
        return None
