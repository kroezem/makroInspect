"""
Refine stage - Post-process heatmaps for pixel-level accuracy.

Reads:
    artifacts/heatmaps/reverted/{split}/{label}/{stem}.npy
    data/{split}/{label}/*.jpg (for guided filtering)
Writes:
    artifacts/heatmaps/refined/{split}/{label}/{stem}.npy
Updates registry: (nothing)

Config:
    refine:
        method: "guided"  # raw, gaussian, bilateral, guided, morph, guided_morph
        gaussian_sigma: 2
        bilateral_d: 9
        bilateral_sigma_color: 75
        bilateral_sigma_space: 75
        guided_radius: 8
        guided_eps: 0.01
        morph_threshold_pct: 90
        morph_open_size: 3
        morph_close_size: 5
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from core.paths import ProjectPaths

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def process(paths: ProjectPaths, cfg: dict, registry: pd.DataFrame) -> pd.DataFrame:
    """
    Refine reverted heatmaps for better pixel-level localization.

    Returns registry unchanged (refinement is an artifact).
    """
    refine_cfg = cfg.get("refine", {})
    method = refine_cfg.get("method", "raw")

    if method == "raw":
        print("✓ Refine method is 'raw', skipping")
        return registry

    if not paths.heatmaps_dir.exists():
        print("No heatmaps found. Run heatmap stage first.")
        return registry

    reverted_dir = paths.heatmaps_dir / "reverted"
    refined_dir = paths.heatmaps_dir / "refined"

    if not reverted_dir.exists():
        print("No reverted heatmaps found.")
        return registry

    # Gather all reverted heatmaps
    heatmap_files = list(reverted_dir.rglob("*.npy"))
    if not heatmap_files:
        print("No reverted heatmap files found.")
        return registry

    print(f"  Method: {method}")
    print(f"  Heatmaps to refine: {len(heatmap_files)}")
    _print_method_params(refine_cfg, method)

    # Build refiner
    refiner = _build_refiner(refine_cfg, method)

    print()

    # Process each heatmap
    for heatmap_path in tqdm(heatmap_files, desc="  Refining", unit="file"):
        # Parse path: reverted/{split}/{label}/{stem}.npy
        rel_path = heatmap_path.relative_to(reverted_dir)
        split = rel_path.parts[0]
        label = rel_path.parts[1]
        stem = heatmap_path.stem

        # Load heatmap
        heatmap = np.load(heatmap_path)

        # Load corresponding image for guided methods
        image = None
        if method in ["guided", "guided_morph", "bilateral_guided"]:
            image = _load_source_image(paths.data_dir, split, label, stem)

        # Refine
        refined = refiner(heatmap, image)

        # Save
        out_path = refined_dir / split / label / f"{stem}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, refined.astype(np.float32))

    print()
    print(f"  ✓ Refined {len(heatmap_files)} heatmaps")
    return registry


def _print_method_params(cfg: dict, method: str):
    """Print relevant parameters for the chosen method."""
    if method == "gaussian":
        print(f"    sigma: {cfg.get('gaussian_sigma', 2)}")
    elif method == "bilateral":
        print(f"    d: {cfg.get('bilateral_d', 9)}")
        print(f"    sigma_color: {cfg.get('bilateral_sigma_color', 75)}")
        print(f"    sigma_space: {cfg.get('bilateral_sigma_space', 75)}")
    elif method == "guided":
        print(f"    radius: {cfg.get('guided_radius', 8)}")
        print(f"    eps: {cfg.get('guided_eps', 0.01)}")
    elif method == "morph":
        print(f"    threshold_pct: {cfg.get('morph_threshold_pct', 90)}")
        print(f"    open_size: {cfg.get('morph_open_size', 3)}")
        print(f"    close_size: {cfg.get('morph_close_size', 5)}")
    elif method == "guided_morph":
        print(f"    guided_radius: {cfg.get('guided_radius', 8)}")
        print(f"    guided_eps: {cfg.get('guided_eps', 0.01)}")
        print(f"    threshold_pct: {cfg.get('morph_threshold_pct', 90)}")
        print(f"    open_size: {cfg.get('morph_open_size', 3)}")
        print(f"    close_size: {cfg.get('morph_close_size', 5)}")


def _build_refiner(cfg: dict, method: str):
    """Build a refinement function based on config."""

    if method == "gaussian":
        sigma = cfg.get("gaussian_sigma", 2)
        return lambda h, img: _refine_gaussian(h, sigma)

    elif method == "bilateral":
        d = cfg.get("bilateral_d", 9)
        sigma_color = cfg.get("bilateral_sigma_color", 75)
        sigma_space = cfg.get("bilateral_sigma_space", 75)
        return lambda h, img: _refine_bilateral(h, d, sigma_color, sigma_space)

    elif method == "guided":
        radius = cfg.get("guided_radius", 8)
        eps = cfg.get("guided_eps", 0.01)
        return lambda h, img: _refine_guided(h, img, radius, eps)

    elif method == "morph":
        threshold_pct = cfg.get("morph_threshold_pct", 90)
        open_size = cfg.get("morph_open_size", 3)
        close_size = cfg.get("morph_close_size", 5)
        return lambda h, img: _refine_morph(h, threshold_pct, open_size, close_size)

    elif method == "guided_morph":
        radius = cfg.get("guided_radius", 8)
        eps = cfg.get("guided_eps", 0.01)
        threshold_pct = cfg.get("morph_threshold_pct", 90)
        open_size = cfg.get("morph_open_size", 3)
        close_size = cfg.get("morph_close_size", 5)
        return lambda h, img: _refine_guided_morph(
            h, img, radius, eps, threshold_pct, open_size, close_size
        )

    else:
        print(f"  Warning: Unknown method '{method}', using raw")
        return lambda h, img: h


def _refine_gaussian(heatmap: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing."""
    if sigma <= 0:
        return heatmap
    ksize = int(sigma * 4) | 1  # Ensure odd
    return cv2.GaussianBlur(heatmap.astype(np.float32), (ksize, ksize), sigma)


def _refine_bilateral(
    heatmap: np.ndarray,
    d: int,
    sigma_color: float,
    sigma_space: float
) -> np.ndarray:
    """Bilateral filter (edge-aware smoothing on heatmap only)."""
    heatmap_norm = _normalize(heatmap)
    return cv2.bilateralFilter(heatmap_norm, d, sigma_color, sigma_space)


def _refine_guided(
    heatmap: np.ndarray,
    image: np.ndarray | None,
    radius: int,
    eps: float
) -> np.ndarray:
    """Guided filter using original image as guide."""
    if image is None:
        return heatmap

    heatmap_norm = _normalize(heatmap)
    heatmap_u8 = (heatmap_norm * 255).astype(np.uint8)

    # Ensure image matches heatmap size
    if image.shape[:2] != heatmap.shape[:2]:
        image = cv2.resize(image, (heatmap.shape[1], heatmap.shape[0]))

    refined = cv2.ximgproc.guidedFilter(image, heatmap_u8, radius, eps)
    return refined.astype(np.float32) / 255.0


def _refine_morph(
    heatmap: np.ndarray,
    threshold_pct: float,
    open_size: int,
    close_size: int
) -> np.ndarray:
    """Morphological cleanup after percentile thresholding."""
    thresh = np.percentile(heatmap, threshold_pct)
    binary = (heatmap > thresh).astype(np.uint8)

    if open_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    if close_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary.astype(np.float32)


def _refine_guided_morph(
    heatmap: np.ndarray,
    image: np.ndarray | None,
    radius: int,
    eps: float,
    threshold_pct: float,
    open_size: int,
    close_size: int
) -> np.ndarray:
    """Guided filter followed by morphological cleanup."""
    refined = _refine_guided(heatmap, image, radius, eps)
    return _refine_morph(refined, threshold_pct, open_size, close_size)


def _normalize(heatmap: np.ndarray) -> np.ndarray:
    """Normalize heatmap to [0, 1]."""
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min < 1e-8:
        return np.zeros_like(heatmap, dtype=np.float32)
    return ((heatmap - h_min) / (h_max - h_min)).astype(np.float32)


def _load_source_image(data_dir: Path, split: str, label: str, stem: str) -> np.ndarray | None:
    """Load source image for guided filtering."""
    search_dir = data_dir / split / label
    if not search_dir.exists():
        return None

    for ext in IMG_EXTS:
        candidate = search_dir / f"{stem}{ext}"
        if candidate.exists():
            return cv2.imread(str(candidate))

    return None