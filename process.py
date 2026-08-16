"""
Main processing pipeline for makroInspect.

Runs expensive compute stages with automatic invalidation checking.
Stages: segment → crop → embed → bank → heatmap → refine

Usage (CLI):
    uv run process.py capsules
    uv run process.py capsules --force crop

Usage (Python):
    from process import process_project
    result = process_project("capsules")
"""

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from ruamel.yaml import YAML

from core.paths import ProjectPaths
from core.registry import (
    create_empty_registry,
    load_instances_base,
    save_instances_base,
)
from core.invalidation import (
    compute_invalid_stages,
    invalidate_stage_artifacts,
    save_hash,
    compute_stage_hash,
    STAGE_ORDER,
)
from stages import segment, crop, embed, bank, heatmap, refine


def load_config(config_path: Path) -> dict:
    """Load YAML config file."""
    yaml = YAML()
    with open(config_path, "r") as f:
        return yaml.load(f)


def process_project(
    project: str,
    *,
    force_from: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run the processing pipeline for a project.

    This is the primary entry point for both CLI and notebook usage.

    Args:
        project: Project name (e.g., "capsules")
        force_from: Force rerun from this stage onward
                   (one of: segment, crop, embed, bank, heatmap, refine)
        verbose: Print progress messages

    Returns:
        Dictionary with:
            - project: Project name
            - instances: Number of instances processed
            - stages_run: List of stages that were executed
            - timings: Dict of stage -> seconds
            - total_time: Total processing time in seconds

    Raises:
        FileNotFoundError: If project or config not found
        ValueError: If force_from is not a valid stage name

    Example:
        >>> from process import process_project
        >>> result = process_project("capsules")
        >>> print(f"Processed {result['instances']} instances")
    """
    # Clear CUDA cache to prevent memory accumulation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    paths = ProjectPaths(project)

    if not paths.exists():
        raise FileNotFoundError(f"Project '{project}' not found at {paths.project_root}")

    if not paths.config.exists():
        raise FileNotFoundError(f"Config not found at {paths.config}")

    cfg = load_config(paths.config)

    if verbose:
        print(f"Processing project: {project}")
        print(f"Config: {paths.config}")

    # Determine which stages need to run
    if force_from:
        if force_from not in STAGE_ORDER:
            raise ValueError(f"Invalid stage '{force_from}'. Valid stages: {STAGE_ORDER}")
        idx = STAGE_ORDER.index(force_from)
        invalid_stages = STAGE_ORDER[idx:]
        if verbose:
            print(f"\nForcing rerun from '{force_from}': {invalid_stages}")
    else:
        invalid_stages = compute_invalid_stages(paths, cfg)
        if verbose:
            if invalid_stages:
                print(f"\nConfig changed, invalidating: {invalid_stages}")
            else:
                print("\nAll stages up to date")

    # Invalidate artifact folders for stages that need rerun
    for stage in invalid_stages:
        invalidate_stage_artifacts(paths, stage)

    # Load or create registry
    if "segment" in invalid_stages:
        registry = create_empty_registry()
    else:
        registry = load_instances_base(paths)

    # Stage dispatch
    stage_funcs = {
        "segment": segment.process,
        "crop": crop.process,
        "embed": embed.process,
        "bank": bank.process,
        "heatmap": heatmap.process,
        "refine": refine.process,
    }

    timings = {}
    stages_run = []
    total_start = time.time()

    for stage in STAGE_ORDER:
        if stage in invalid_stages:
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Running: {stage}")
                print(f"{'=' * 60}")

            start = time.time()
            registry = stage_funcs[stage](paths, cfg, registry)
            elapsed = time.time() - start

            timings[stage] = elapsed
            stages_run.append(stage)

            if verbose:
                print(f"✓ {stage} complete ({elapsed:.1f}s)")

            # Save hash after successful completion
            save_hash(paths, stage, compute_stage_hash(cfg, stage))

            # Save registry after each stage
            save_instances_base(registry, paths)
        else:
            if verbose:
                print(f"\n✓ {stage}: up to date, skipping")

    total_elapsed = time.time() - total_start

    if verbose:
        print(f"\n{'=' * 60}")
        print("Pipeline complete")
        print(f"{'=' * 60}")
        print(f"Instances: {len(registry)}")
        print(f"Total time: {total_elapsed:.1f}s")
        if timings:
            print("\nStage timings:")
            for stage, t in timings.items():
                print(f"  {stage}: {t:.1f}s")

    return {
        "project": project,
        "instances": len(registry),
        "stages_run": stages_run,
        "timings": timings,
        "total_time": total_elapsed,
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Process a makroInspect project")
    parser.add_argument("project", help="Project name")
    parser.add_argument(
        "--force",
        choices=STAGE_ORDER,
        help="Force rerun from this stage onward"
    )
    args = parser.parse_args()

    try:
        process_project(args.project, force_from=args.force, verbose=True)
        return 0
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    except ValueError as e:
        print(f"❌ {e}")
        return 1


if __name__ == "__main__":
    exit(main())
