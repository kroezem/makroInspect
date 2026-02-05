"""
Config-based invalidation logic.

Tracks which config parameters affect which stages.
Determines what needs rerunning when config changes.
"""

import hashlib
import json
from pathlib import Path

from core.paths import ProjectPaths

# Which stages each parameter invalidates
PARAM_INVALIDATES = {
    # Segmentation
    "segmentation.description": ["segment"],
    "segmentation.enabled": ["segment"],
    "segmentation.multi_instance": ["segment"],

    # Crop (note: transformer params affect crop size)
    "crop.reference_id": ["crop"],
    "crop.apply_mask": ["crop"],
    "crop.align_PCA": ["crop"],
    "crop.angle_candidates": ["crop"],
    "crop.offset_rotation": ["crop"],
    "crop.scale": ["crop"],
    "transformer.map_resolution": ["crop"],
    "transformer.patch_size": ["crop"],

    # Bank
    "memory_bank.max_tokens": ["bank"],
    "memory_bank.selection_method": ["bank"],

    # Refine (only invalidates refined heatmaps, not raw heatmaps)
    "refine.method": ["refine"],
    "refine.gaussian_sigma": ["refine"],
    "refine.bilateral_d": ["refine"],
    "refine.bilateral_sigma_color": ["refine"],
    "refine.bilateral_sigma_space": ["refine"],
    "refine.guided_radius": ["refine"],
    "refine.guided_eps": ["refine"],
    "refine.morph_threshold_pct": ["refine"],
    "refine.morph_open_size": ["refine"],
    "refine.morph_close_size": ["refine"],

    # Scoring params don't invalidate artifacts
    "anomaly_score.method": [],
    "anomaly_score.min_percentile": [],
    "anomaly_score.max_percentile": [],
    "anomaly_score.cmap": [],
}

# STAGE_ORDER = ["segment", "crop", ]
STAGE_ORDER = ["segment", "crop", "embed", "bank", "heatmap", "refine"]

# Map stages to their artifact directories
STAGE_ARTIFACTS = {
    "segment": "masks_dir",
    "crop": "crops_dir",
    "embed": "embed_dir",
    "bank": "bank_dir",
    "heatmap": "heatmaps_dir",
    "refine": "refine_dir",
}


def get_stage_params(stage: str) -> list[str]:
    """Get all config params that affect a stage."""
    params = []
    for param, stages in PARAM_INVALIDATES.items():
        if stage in stages:
            params.append(param)
    return sorted(params)


def get_param_value(cfg: dict, param_path: str):
    """Get nested config value from dot-notation path."""
    keys = param_path.split(".")
    value = cfg
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


def compute_stage_hash(cfg: dict, stage: str) -> str:
    """Compute hash of config params relevant to a stage."""
    params = get_stage_params(stage)
    values = {p: get_param_value(cfg, p) for p in params}
    content = json.dumps(values, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def load_stored_hash(paths: ProjectPaths, stage: str) -> str | None:
    """Load stored hash for a stage."""
    hash_path = paths.config_hash(stage)
    if hash_path.exists():
        return hash_path.read_text().strip()
    return None


def save_hash(paths: ProjectPaths, stage: str, hash_value: str):
    """Save hash for a stage."""
    hash_path = paths.config_hash(stage)
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(hash_value)


def compute_invalid_stages(paths: ProjectPaths, cfg: dict) -> list[str]:
    """
    Compare current config against stored hashes.
    Returns list of stages that need rerunning (with cascade).
    """
    invalid = set()

    for stage in STAGE_ORDER:
        current_hash = compute_stage_hash(cfg, stage)
        stored_hash = load_stored_hash(paths, stage)

        # Check if artifact directory exists
        artifact_dir = getattr(paths, STAGE_ARTIFACTS[stage])
        artifacts_exist = artifact_dir.exists() and any(artifact_dir.iterdir()) if artifact_dir.exists() else False

        if stored_hash != current_hash or not artifacts_exist:
            invalid.add(stage)

    # Cascade: if a stage is invalid, all downstream are too
    if invalid:
        earliest_idx = min(STAGE_ORDER.index(s) for s in invalid)
        return STAGE_ORDER[earliest_idx:]

    return []


def invalidate_stage_artifacts(paths: ProjectPaths, stage: str):
    """Delete artifact directory for a stage."""
    import shutil

    artifact_dir = getattr(paths, STAGE_ARTIFACTS[stage])
    if artifact_dir.exists():
        try:
            disp = artifact_dir.resolve().relative_to(paths.project_root.resolve())
        except ValueError:
            disp = artifact_dir.resolve()
        print(f"  Deleting {disp}")
        shutil.rmtree(artifact_dir)

    # Also delete the hash
    hash_path = paths.config_hash(stage)
    if hash_path.exists():
        hash_path.unlink()
