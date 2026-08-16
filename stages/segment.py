"""
Segmentation stage - SAM3 instance segmentation.

Reads:  data/{split}/{label}/*.jpg
Writes: artifacts/masks/{split}/{label}/{stem}_{idx}.png
Adds:   instance_id, image, split, label, mask_area_px, segment_prompt
"""

import warnings
warnings.filterwarnings("ignore", message=".*xFormers is available.*")

import numpy as np
import pandas as pd
import torch
import torch
from pathlib import Path
from PIL import Image as PILImage
from tqdm import tqdm

from core.paths import ProjectPaths


class SegmentationError(Exception):
    """Raised when segmentation fails to produce masks for images."""
    pass


def process(paths: ProjectPaths, cfg: dict, registry: pd.DataFrame) -> pd.DataFrame:
    """
    Segment all images in a project.

    - If segmentation is disabled: write one blank mask per image as {stem}_00 and return a registry of those.
    - If segmentation is enabled: if SAM returns no masks for an image, write nothing for that image.
    - segmentation.prompts is always a list of strings.
    """
    seg_cfg = cfg.get("segmentation", {})
    enabled = seg_cfg.get("enabled", True)

    image_entries = _gather_images(paths.data_dir)
    if not image_entries:
        print("No images found")
        if not enabled:
            return pd.DataFrame(columns=["instance_id", "image", "split", "label", "mask_area_px", "segment_prompt"])
        return registry

    # If disabled, generate one blank mask per image and return
    if not enabled:
        print("✓ Segmentation disabled, writing blank _00 masks")
        print()

        rows = []
        for entry in tqdm(image_entries, desc="Blank masks", unit="img"):
            split = entry["split"]
            label = entry["label"]
            stem = entry["stem"]
            image_id = f"{split}/{label}/{stem}"

            try:
                pil = PILImage.open(entry["path"]).convert("RGB")
                w, h = pil.size
            except Exception as e:
                print(f"\n⚠ Failed to load {entry['path']}: {e}")
                continue

            mask = np.zeros((h, w), dtype=bool)
            instance_id = f"{stem}_00"
            mask_path = paths.mask(split, label, instance_id)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            _save_mask(mask, mask_path)

            rows.append({
                "instance_id": instance_id,
                "image": image_id,
                "split": split,
                "label": label,
                "mask_area_px": 0,
                "segment_prompt": None,
            })

        print()
        out = pd.DataFrame(rows)
        print(f"  ✓ Created {len(out)} blank instances")
        return out

    # Enabled: check if masks already exist
    if paths.masks_dir.exists() and any(paths.masks_dir.rglob("*.png")):
        print("✓ Masks already exist, skipping")
        return registry

    prompts = seg_cfg.get("prompts", ["object"])
    multi_instance = seg_cfg.get("multi_instance", True)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    print(f"  Prompts: {prompts}")
    print(f"  Multi-instance: {multi_instance}")
    print(f"  Device: {device}")
    print(f"  Found {len(image_entries)} images")

    print()
    print("  Loading SAM3...")
    from transformers import Sam3Processor, Sam3Model

    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained(
        "facebook/sam3",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    print("  ✓ SAM3 loaded")
    print()

    # Key: (split, label, stem) -> list[(mask, prompt)]
    all_masks: dict[tuple[str, str, str], list[tuple[np.ndarray, str]]] = {}

    # Track all images we attempt to segment
    all_image_keys: set[tuple[str, str, str]] = {
        (e["split"], e["label"], e["stem"]) for e in image_entries
    }

    batch_size = 8

    for prompt in prompts:
        print(f"  Processing prompt: '{prompt}'")
        print()

        pbar = tqdm(total=len(image_entries), desc=f"  '{prompt}'", unit="img")
        for i in range(0, len(image_entries), batch_size):
            batch = image_entries[i:i + batch_size]
            pbar.update(len(batch))

            pil_images = []
            valid_entries = []
            for entry in batch:
                try:
                    pil = PILImage.open(entry["path"]).convert("RGB")
                    pil_images.append(pil)
                    valid_entries.append(entry)
                except Exception as e:
                    print(f"\n⚠ Failed to load {entry['path']}: {e}")

            if not pil_images:
                continue

            inputs = processor(
                images=pil_images,
                text=[prompt] * len(pil_images),
                return_tensors="pt",
            ).to(device)
            inputs = {k: v.half() if torch.is_floating_point(v) else v for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = model(**inputs)
                target_sizes = [img.size[::-1] for img in pil_images]
                results = processor.post_process_instance_segmentation(
                    outputs,
                    target_sizes=target_sizes,
                    threshold=0.5,
                )

            for entry, result in zip(valid_entries, results):
                key = (entry["split"], entry["label"], entry["stem"])
                masks = _extract_masks(result)
                if not masks:
                    continue

                if not multi_instance and len(masks) > 1:
                    masks = [np.logical_or.reduce(masks, axis=0)]

                all_masks.setdefault(key, [])
                for mask in masks:
                    all_masks[key].append((mask, prompt))

        pbar.close()
        print()

    # Check for images that got no masks from any prompt
    images_without_masks = all_image_keys - set(all_masks.keys())
    if images_without_masks:
        failed_list = sorted(f"{s}/{l}/{stem}" for s, l, stem in images_without_masks)
        raise SegmentationError(
            f"Segmentation enabled but no masks found for {len(failed_list)} image(s):\n"
            + "\n".join(f"  - {img}" for img in failed_list[:10])
            + (f"\n  ... and {len(failed_list) - 10} more" if len(failed_list) > 10 else "")
        )

    # Save masks and build registry
    total_masks = sum(len(v) for v in all_masks.values())
    print(f"  Saving {total_masks} masks...")
    print()

    rows = []
    save_pbar = tqdm(total=total_masks, desc="  Saving masks", unit="mask")
    try:
        for (split, label, stem), mask_list in all_masks.items():
            image_id = f"{split}/{label}/{stem}"

            for idx, (mask, prompt) in enumerate(mask_list):
                instance_id = f"{stem}_{idx:02d}"
                mask_path = paths.mask(split, label, instance_id)
                mask_path.parent.mkdir(parents=True, exist_ok=True)

                _save_mask(mask, mask_path)
                save_pbar.update(1)

                rows.append({
                    "instance_id": instance_id,
                    "image": image_id,
                    "split": split,
                    "label": label,
                    "mask_area_px": int(mask.sum()),
                    "segment_prompt": prompt,
                })
    finally:
        save_pbar.close()
        print()

    out = pd.DataFrame(rows)

    if len(prompts) > 1 and not out.empty:
        print()
        print("  Summary by prompt:")
        for p in prompts:
            count = (out["segment_prompt"] == p).sum()
            print(f"    '{p}': {count} instances")

    print(f"  ✓ Created {len(out)} total instances")
    return out


def _gather_images(data_dir: Path) -> list[dict]:
    """Find all images in data/{split}/{label}/"""
    entries = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for split in ["train", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        for label_dir in split_dir.iterdir():
            if not label_dir.is_dir():
                continue

            label = label_dir.name
            for img_path in label_dir.iterdir():
                if img_path.suffix.lower() in valid_exts:
                    entries.append({
                        "split": split,
                        "label": label,
                        "stem": img_path.stem,
                        "path": img_path,
                    })

    return entries


def _extract_masks(result) -> list[np.ndarray]:
    """Extract boolean masks from SAM3 result."""
    if result is None:
        return []

    masks = result.get("masks", None) if isinstance(result, dict) else result
    if masks is None:
        return []

    if torch.is_tensor(masks):
        masks = masks.detach().cpu().numpy()

    masks = masks.astype(bool)

    if masks.ndim == 2:
        return [masks]
    if masks.ndim == 3:
        return [masks[i] for i in range(masks.shape[0])]

    return []


def _save_mask(mask: np.ndarray, path: Path):
    """Save boolean mask as PNG (0 or 255)."""
    img_data = (mask.astype(np.uint8) * 255)
    # optimize=True can be very slow across many files on Windows
    PILImage.fromarray(img_data, mode="L").save(path)
