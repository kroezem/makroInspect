"""
Score stage - Score using cached FULL refined heatmaps only (image-level).

Reads:
    artifacts/{project}/heatmaps/refined/test/{label}/{stem}.npy
    artifacts/{project}/data/ground_truth/
    projects/{project}/config.yaml

Writes:
    projects/{project}/runs/{run_id}/config.yaml
    projects/runs.csv (GLOBAL append, cross-project comparable)
    projects/{project}/runs.csv (PER-PROJECT append, includes per-defect AUROC columns)

Usage:
    uv run score.py capsules
    uv run score.py capsules --method mean --min-pct 99.9 --max-pct 99.999
    uv run score.py capsules --pixel-sample 100000

Also callable:
    from score import score_project
    row = score_project("capsules", method="mean", min_pct=99.9, max_pct=99.999, pixel_sample=100000)
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from core.paths import ProjectPaths, PROJECTS_DIR
from core.registry import generate_run_id


def score_project(
    project: str,
    method: str | None = None,
    min_pct: float | None = None,
    max_pct: float | None = None,
    pixel_sample: int | None = None,
    save: bool = True,
    verbose: bool = True,
) -> dict | None:
    """
    Score a project using cached refined heatmaps.

    Returns the row written to runs.csv (including metrics).
    If save=False, no files are written and run_id is not created.
    """
    paths = ProjectPaths(project)

    if not paths.config.exists():
        if verbose:
            print(f"Project '{project}' not found")
        return None

    with open(paths.config) as f:
        cfg = yaml.safe_load(f)

    # Pull map resolution area for logging
    map_area = np.nan
    try:
        mr = cfg.get("transformer", {}).get("map_resolution", None)
        if isinstance(mr, (list, tuple)) and len(mr) == 2:
            map_area = int(mr[0]) * int(mr[1])
    except Exception:
        map_area = np.nan

    # Use config section "score" if present, otherwise fall back to older "anomaly_score"
    score_cfg = cfg.get("score")
    if score_cfg is None:
        score_cfg = cfg.setdefault("anomaly_score", {})
    else:
        score_cfg = cfg.setdefault("score", score_cfg)

    score_cfg["mode"] = "image_heatmap"
    score_cfg["heatmap_source"] = "refined"

    # Apply overrides
    if method is not None:
        score_cfg["method"] = method
    if min_pct is not None:
        score_cfg["min_percentile"] = float(min_pct)
    if max_pct is not None:
        score_cfg["max_percentile"] = float(max_pct)
    if pixel_sample is not None:
        score_cfg["pixel_sample"] = int(pixel_sample)

    # Effective settings
    method = score_cfg.get("method", "mean")
    min_pct = float(score_cfg.get("min_percentile", 99.9))
    max_pct = float(score_cfg.get("max_percentile", 99.999))
    pixel_sample = score_cfg.get("pixel_sample", 100000)
    pixel_sample = int(pixel_sample) if pixel_sample is not None else None

    if verbose:
        print(f"Scoring project: {project}")
        print("  Heatmap source: refined (forced)")
        print(f"  Method: {method}")
        print(f"  Percentile band: {min_pct} - {max_pct}")
        print(f"  Map resolution area: {map_area}")
        if pixel_sample is not None:
            print(f"  Pixel AUROC sampling: {pixel_sample} pixels")
        print()

    t_start = time.time()

    refined_dir = paths.heatmaps_dir / "refined"
    test_dir = refined_dir / "test"
    if not test_dir.exists():
        if verbose:
            print(f"Missing refined heatmaps at {test_dir}")
        return None

    # Collect test heatmaps
    samples = []
    for label_dir in sorted(test_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for npy_path in sorted(label_dir.glob("*.npy")):
            samples.append((label, npy_path.stem, npy_path))

    if not samples:
        if verbose:
            print(f"No refined heatmaps found under {test_dir}")
        return None

    if verbose:
        print(f"Found {len(samples)} test heatmaps")
        print()
        print("Scoring images...")

    normal_scores = []
    anomaly_scores = []
    anomaly_labels = []

    for label, stem, npy_path in tqdm(samples, desc="  Images", disable=not verbose):
        heatmap = np.load(npy_path).astype(np.float32, copy=False)
        flat = heatmap.ravel()

        if flat.size == 0:
            score = np.nan
        else:
            lo_val = np.percentile(flat, min_pct)
            hi_val = np.percentile(flat, max_pct)
            band = flat[(flat >= lo_val) & (flat <= hi_val)]

            if band.size == 0:
                band = flat[flat >= lo_val]
            if band.size == 0:
                band = flat

            if method == "mean":
                score = float(band.mean())
            elif method == "max":
                score = float(band.max())
            elif method == "sum":
                score = float(band.sum())
            else:
                score = float(band.mean())

        if label == "good":
            normal_scores.append(score)
        else:
            anomaly_scores.append(score)
            anomaly_labels.append(label)

    # Drop NaNs
    normal_arr = np.array([s for s in normal_scores if np.isfinite(s)], dtype=np.float64)
    aligned_an = [(s, l) for s, l in zip(anomaly_scores, anomaly_labels) if np.isfinite(s)]
    anomaly_arr = np.array([s for s, _ in aligned_an], dtype=np.float64)
    anomaly_lbl = np.array([l for _, l in aligned_an]) if aligned_an else np.array([])

    if verbose:
        print()
        print(f"  Normal images (test/good): {len(normal_arr)}")
        print(f"  Anomaly images: {len(anomaly_arr)}")
        print()
        print("Computing metrics...")

    metrics = {}

    if len(normal_arr) > 0 and len(anomaly_arr) > 0:
        y_true = np.concatenate([np.zeros(len(normal_arr)), np.ones(len(anomaly_arr))])
        y_scores = np.concatenate([normal_arr, anomaly_arr])

        metrics["auroc_image"] = float(roc_auc_score(y_true, y_scores))
        metrics["ap_image"] = float(average_precision_score(y_true, y_scores))

        if verbose:
            print(f"  Image AUROC: {metrics['auroc_image']:.4f}")
            print(f"  Image AP: {metrics['ap_image']:.4f}")
    else:
        metrics["auroc_image"] = np.nan
        metrics["ap_image"] = np.nan
        if verbose:
            print("  Image AUROC: N/A (missing normal or anomaly images)")

    if len(normal_arr) > 0:
        metrics["normal_mean"] = float(np.mean(normal_arr))
        metrics["normal_std"] = float(np.std(normal_arr))
        metrics["normal_max"] = float(np.max(normal_arr))
    else:
        metrics["normal_mean"] = np.nan
        metrics["normal_std"] = np.nan
        metrics["normal_max"] = np.nan

    if len(anomaly_arr) > 0:
        metrics["anomaly_mean"] = float(np.mean(anomaly_arr))
        metrics["anomaly_std"] = float(np.std(anomaly_arr))
        metrics["anomaly_min"] = float(np.min(anomaly_arr))
    else:
        metrics["anomaly_mean"] = np.nan
        metrics["anomaly_std"] = np.nan
        metrics["anomaly_min"] = np.nan

    if np.isfinite(metrics["normal_mean"]) and np.isfinite(metrics["anomaly_mean"]):
        metrics["separation"] = float(metrics["anomaly_mean"] - metrics["normal_mean"])
        if verbose:
            print(f"  Separation: {metrics['separation']:.4f}")
    else:
        metrics["separation"] = np.nan

    # Per-defect AUROC (only per-project runs.csv)
    per_defect = {}
    defect_labels = sorted(set(anomaly_lbl.tolist())) if anomaly_lbl.size else []
    if len(normal_arr) > 0 and len(defect_labels) > 0:
        for defect in defect_labels:
            defect_scores = anomaly_arr[anomaly_lbl == defect]
            if defect_scores.size == 0:
                continue

            y_true = np.concatenate([np.zeros(len(normal_arr)), np.ones(len(defect_scores))])
            y_scores = np.concatenate([normal_arr, defect_scores])

            # roc_auc_score fails if y_true is all one class (should not happen here, but guard anyway)
            try:
                defect_clean = defect.replace(",", "_").replace(" ", "_")
                per_defect[f"auroc_{defect_clean}"] = float(roc_auc_score(y_true, y_scores))
            except Exception:
                continue

    if verbose:
        print()
        print("Computing pixel-level AUROC (refined)...")

    pixel_auroc = _compute_pixel_auroc(paths, refined_dir, sample_pixels=pixel_sample)
    if pixel_auroc is not None:
        metrics["auroc_pixel"] = float(pixel_auroc)
        if verbose:
            print(f"  Pixel AUROC: {metrics['auroc_pixel']:.4f}")
    else:
        metrics["auroc_pixel"] = np.nan
        if verbose:
            print("  Pixel AUROC: N/A (no ground truth)")

    t_end = time.time()
    score_time = t_end - t_start

    base_row = {
        "run_id": "" if not save else generate_run_id(),
        "project": project,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": "image_heatmap",
        "heatmap_source": "refined",
        "method": method,
        "min_pct": min_pct,
        "max_pct": max_pct,
        "map_resolution_area": map_area,
        "pixel_sample": pixel_sample if pixel_sample is not None else "",
        "n_test_heatmaps": int(len(samples)),
        "n_normal_images": int(len(normal_arr)),
        "n_anomaly_images": int(len(anomaly_arr)),
        "score_time_s": round(score_time, 2),
        **metrics,
    }

    if not save:
        return base_row

    run_id = base_row["run_id"]
    run_dir = paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    with open(paths.run_config(run_id), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # GLOBAL runs.csv (no per-defect columns)
    global_runs_csv = PROJECTS_DIR / "runs.csv"
    global_df = pd.read_csv(global_runs_csv) if global_runs_csv.exists() else pd.DataFrame()
    global_df = pd.concat([global_df, pd.DataFrame([base_row])], ignore_index=True)
    global_df.to_csv(global_runs_csv, index=False)

    # PER-PROJECT runs.csv (includes per-defect columns)
    project_runs_csv = paths.project_root / "runs.csv"
    project_row = {**base_row, **per_defect}
    proj_df = pd.read_csv(project_runs_csv) if project_runs_csv.exists() else pd.DataFrame()
    proj_df = pd.concat([proj_df, pd.DataFrame([project_row])], ignore_index=True)
    proj_df.to_csv(project_runs_csv, index=False)

    if verbose:
        print()
        print("Saving run...")
        print(f"  Run ID: {run_id}")
        print(f"  Config: {paths.run_config(run_id)}")
        print(f"  Global runs.csv: {global_runs_csv}")
        print(f"  Project runs.csv: {project_runs_csv}")
        print(f"  Time: {score_time:.1f}s")
        print()
        print("=" * 50)
        print("RESULTS")
        print("=" * 50)
        print(f"  Image AUROC:  {metrics.get('auroc_image', np.nan):.4f}")
        print(f"  Pixel AUROC:  {metrics.get('auroc_pixel', np.nan):.4f}")
        print(f"  Separation:   {metrics.get('separation', np.nan):.4f}")

        if per_defect:
            print()
            print("  Per-defect AUROC:")
            for k, v in sorted(per_defect.items()):
                print(f"    {k.replace('auroc_', '')}: {v:.4f}")

    return base_row


def main():
    parser = argparse.ArgumentParser(description="Score a project using cached refined heatmaps (image-level)")
    parser.add_argument("project", help="Project name")
    parser.add_argument("--method", help="Aggregation method (mean, max, sum)")
    parser.add_argument("--min-pct", type=float, help="Min percentile")
    parser.add_argument("--max-pct", type=float, help="Max percentile")
    parser.add_argument(
        "--pixel-sample",
        type=int,
        help="If set, compute pixel AUROC on a random sample of this many pixels",
    )
    args = parser.parse_args()

    score_project(
        args.project,
        method=args.method,
        min_pct=args.min_pct,
        max_pct=args.max_pct,
        pixel_sample=args.pixel_sample,
        save=True,
        verbose=True,
    )


def _compute_pixel_auroc(paths: ProjectPaths, heatmap_base: Path, sample_pixels: int | None = None) -> float | None:
    gt_dir = paths.data_dir / "ground_truth"
    if not gt_dir.exists():
        return None

    all_scores = []
    all_labels = []

    rng = np.random.default_rng(0)

    for label_dir in gt_dir.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name

        for gt_path in label_dir.glob("*.png"):
            stem = gt_path.stem
            heatmap_path = heatmap_base / "test" / label / f"{stem}.npy"
            if not heatmap_path.exists():
                continue

            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if gt is None:
                continue
            gt = (gt > 127)

            heatmap = np.load(heatmap_path).astype(np.float32, copy=False)

            if heatmap.shape != gt.shape:
                heatmap = cv2.resize(heatmap, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

            s = heatmap.ravel()
            y = gt.ravel().astype(np.uint8)

            if sample_pixels is not None and sample_pixels > 0:
                n = s.size
                k = min(sample_pixels, n)
                idx = rng.choice(n, size=k, replace=False)
                s = s[idx]
                y = y[idx]

            all_scores.append(s.astype(np.float32, copy=False))
            all_labels.append(y)

    if not all_scores:
        return None

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    pos = int(all_labels.sum())
    if pos == 0 or pos == len(all_labels):
        return None

    return float(roc_auc_score(all_labels, all_scores))


if __name__ == "__main__":
    main()
