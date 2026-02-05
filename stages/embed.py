"""
Embed stage - DINOv2 feature extraction.

Reads:
    artifacts/crops/images/{split}/{label}/{instance_id}.png
Writes:
    artifacts/embed/{split}/{label}/{instance_id}.npy  (float16, H×W×C)
Updates registry: embed_shape
"""

import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from core.paths import ProjectPaths


def process(paths: ProjectPaths, cfg: dict, registry: pd.DataFrame) -> pd.DataFrame:
    """
    Embed all crop images using DINOv2.

    Returns updated registry with embed_shape column populated.
    """
    if paths.embed_dir.exists() and any(paths.embed_dir.rglob("*.npy")):
        print("✓ Embeddings already exist, skipping")
        return registry

    if not paths.crops_dir.exists():
        print("No crops found. Run crop stage first.")
        return registry

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    transformer_cfg = cfg.get("transformer", {})
    patch_size = int(transformer_cfg.get("patch_size", 14))
    batch_size = int(cfg.get("embed", {}).get("batch_size", 16))

    print(f"  Device: {device}")
    print(f"  Patch size: {patch_size}")
    print(f"  Batch size: {batch_size}")

    # Initialize model
    embedder = _DinoEmbedder(device=device, patch_size=patch_size)

    # Prepare output column
    registry["embed_shape"] = ""

    # Process in batches
    instances = registry.to_dict("records")
    indices = registry.index.tolist()

    total_done = 0
    total_skipped = 0

    for i in tqdm(range(0, len(instances), batch_size), desc="Embedding"):
        batch_instances = instances[i:i + batch_size]
        batch_indices = indices[i:i + batch_size]

        # Load crop images
        images = []
        valid_instances = []
        valid_indices = []

        for inst, idx in zip(batch_instances, batch_indices):
            crop_path = paths.crop_image(inst["split"], inst["label"], inst["instance_id"])

            if not crop_path.exists():
                total_skipped += 1
                continue

            img = cv2.imread(str(crop_path))
            if img is None:
                total_skipped += 1
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)
            valid_instances.append(inst)
            valid_indices.append(idx)

        if not images:
            continue

        # Run DINOv2
        patches = embedder.embed_batch(images)
        if patches is None:
            total_skipped += len(valid_instances)
            continue

        # Save embeddings and update registry
        for j, (inst, idx) in enumerate(zip(valid_instances, valid_indices)):
            embed_path = paths.embedding(inst["split"], inst["label"], inst["instance_id"])
            embed_path.parent.mkdir(parents=True, exist_ok=True)

            patch_embedding = patches[j].astype(np.float16)
            _save_atomic(patch_embedding, embed_path)

            H, W, C = patch_embedding.shape
            registry.loc[idx, "embed_shape"] = json.dumps([H, W, C])
            total_done += 1

    print(f"  ✓ Embedded {total_done} instances")
    if total_skipped:
        print(f"  Skipped: {total_skipped} instances")

    return registry


class _DinoEmbedder:
    """Lazy-loading DINOv2 embedder."""

    def __init__(self, device: str, patch_size: int):
        self.device = device
        self.patch_size = patch_size
        self._model = None
        self._mean = None
        self._std = None

        self._load()

    def _load(self):
        if self._model is not None:
            return

        print()
        print("  Loading DINOv2...")

        self._model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self._model.to(self.device).half().eval()


        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device).half()
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device).half()

        # Warmup
        dummy = torch.zeros(1, 3, self.patch_size * 2, self.patch_size * 2, device=self.device, dtype=torch.float16)
        with torch.inference_mode():
            self._model.forward_features(dummy)

        print("  ✓ DINOv2 loaded")
        print()

    @torch.no_grad()
    def embed_batch(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        """Embed batch of RGB images. Returns (N, H, W, C) float16."""
        self._load()

        if not images:
            return None

        # Stack to (N, 3, H, W)
        tensors = [torch.from_numpy(img).permute(2, 0, 1) for img in images]
        batch = torch.stack(tensors).to(self.device).half()

        # Normalize
        batch = batch / 255.0
        batch = (batch - self._mean) / self._std

        with torch.inference_mode():
            output = self._model.forward_features(batch)

            # Extract patch tokens
            if isinstance(output, dict):
                tokens = output["x_norm_patchtokens"]
            else:
                tokens = output[:, 1:]  # Skip CLS token if present

            N, L, C = tokens.shape
            H = batch.shape[2] // self.patch_size
            W = batch.shape[3] // self.patch_size
            tokens = tokens.reshape(N, H, W, C).contiguous()

        return tokens.cpu().numpy().astype(np.float16)


def _save_atomic(arr: np.ndarray, path: Path):
    """Save numpy array atomically."""
    tmp_path = path.with_suffix(".tmp.npy")
    np.save(tmp_path, arr)
    try:
        tmp_path.replace(path)
    except OSError:
        if path.exists():
            path.unlink()
        tmp_path.rename(path)
