"""
Registry management for instances.csv and runs.csv.

Defines schemas and provides I/O functions for tracking instances
through the pipeline and recording run results.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from core.paths import ProjectPaths

# Columns added by each stage (in order)
SEGMENT_COLUMNS = ["instance_id", "image", "split", "label", "mask_area_px"]
CROP_COLUMNS = ["crop_iou", "crop_size", "alignment_angle", "crop_transform"]
EMBED_COLUMNS = ["embed_shape"]
BANK_COLUMNS = ["in_bank"]

# Columns added by scoring
SCORE_COLUMNS = [
    "score", "heatmap_max", "heatmap_mean",
    "p90", "p95", "p99", "p99_5", "p99_9"
]

# All base columns (before scoring)
INSTANCE_BASE_COLUMNS = SEGMENT_COLUMNS + CROP_COLUMNS + EMBED_COLUMNS + BANK_COLUMNS

# All columns including scoring
INSTANCE_ALL_COLUMNS = INSTANCE_BASE_COLUMNS + SCORE_COLUMNS

# Runs.csv columns
RUNS_COLUMNS = [
    # Run identification
    "run_id", "timestamp",
    # Scoring config
    "method", "min_pct", "max_pct", "image_agg",
    # Bank config (for reference)
    "bank_k", "bank_method",
    # Timing
    "segment_time_s", "crop_time_s", "embed_time_s",
    "bank_time_s", "heatmap_time_s", "score_time_s", "total_time_s",
    # Dataset summary
    "n_instances", "n_train", "n_test", "mean_iou",
    # Primary metrics
    "auroc_image", "auroc_pixel", "separation",
    # Score distributions
    "normal_mean", "normal_std", "normal_max",
    "anomaly_mean", "anomaly_std", "anomaly_min",
    # Per-defect columns are dynamic, added as needed
]


def create_empty_registry() -> pd.DataFrame:
    """Create empty DataFrame with base columns."""
    return pd.DataFrame(columns=INSTANCE_BASE_COLUMNS)


def load_instances_base(paths: ProjectPaths) -> pd.DataFrame:
    """Load instances.csv from artifacts folder."""
    if not paths.instances_base.exists():
        return create_empty_registry()
    return pd.read_csv(paths.instances_base)


def save_instances_base(df: pd.DataFrame, paths: ProjectPaths) -> None:
    """Save instances.csv to artifacts folder."""
    paths.instances_base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.instances_base, index=False)


def load_run_instances(paths: ProjectPaths, run_id: str) -> pd.DataFrame:
    """Load instances.csv from a specific run."""
    path = paths.run_instances(run_id)
    if not path.exists():
        raise FileNotFoundError(f"Run instances not found: {path}")
    return pd.read_csv(path)


def save_run_instances(df: pd.DataFrame, paths: ProjectPaths, run_id: str) -> None:
    """Save instances.csv to a run folder."""
    path = paths.run_instances(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def create_empty_runs_csv(paths: ProjectPaths) -> None:
    """Create runs.csv with headers if it doesn't exist."""
    if paths.runs_csv.exists():
        return
    paths.runs_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.runs_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(RUNS_COLUMNS)


def load_runs(paths: ProjectPaths) -> pd.DataFrame:
    """Load runs.csv."""
    if not paths.runs_csv.exists():
        return pd.DataFrame(columns=RUNS_COLUMNS)
    return pd.read_csv(paths.runs_csv)


def append_run(paths: ProjectPaths, metrics: dict[str, Any]) -> None:
    """Append a row to runs.csv."""
    create_empty_runs_csv(paths)

    existing = load_runs(paths)
    all_columns = list(existing.columns)

    for key in metrics:
        if key not in all_columns:
            all_columns.append(key)

    row = {col: metrics.get(col, "") for col in all_columns}

    new_row = pd.DataFrame([row])
    updated = pd.concat([existing, new_row], ignore_index=True)
    updated.to_csv(paths.runs_csv, index=False)


def generate_run_id() -> str:
    """Generate unique run ID from timestamp."""
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"