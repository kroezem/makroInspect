"""
Main processing pipeline for makroInspect.

Runs expensive compute stages with automatic invalidation checking.
Stages: segment → crop → embed → bank → heatmap

Usage:
    uv run process.py capsules
    uv run process.py capsules --force crop
"""

import argparse
import time
from pathlib import Path

import pandas as pd
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


def main():
    parser = argparse.ArgumentParser(description="Process a makroInspect project")
    parser.add_argument("project", help="Project name")
    parser.add_argument(
        "--force",
        choices=STAGE_ORDER,
        help="Force rerun from this stage onward"
    )
    args = parser.parse_args()

    paths = ProjectPaths(args.project)

    # Validate project exists
    if not paths.exists():
        print(f"❌ Project '{args.project}' not found at {paths.project_root}")
        return 1

    if not paths.config_exists():
        print(f"❌ Config not found at {paths.config}")
        return 1

    cfg = load_config(paths.config)
    print(f"Processing project: {args.project}")
    print(f"Config: {paths.config}")

    # Determine which stages need to run
    if args.force:
        idx = STAGE_ORDER.index(args.force)
        invalid_stages = STAGE_ORDER[idx:]
        print(f"\nForcing rerun from '{args.force}': {invalid_stages}")
    else:
        invalid_stages = compute_invalid_stages(paths, cfg)
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
    total_start = time.time()

    for stage in STAGE_ORDER:
        if stage in invalid_stages:
            print(f"\n{'=' * 60}")
            print(f"Running: {stage}")
            print(f"{'=' * 60}")

            start = time.time()
            registry = stage_funcs[stage](paths, cfg, registry)
            elapsed = time.time() - start

            timings[stage] = elapsed
            print(f"✓ {stage} complete ({elapsed:.1f}s)")

            # Save hash after successful completion
            save_hash(paths, stage, compute_stage_hash(cfg, stage))

            # Save registry after each stage
            save_instances_base(registry, paths)
        else:
            print(f"\n✓ {stage}: up to date, skipping")

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'=' * 60}")
    print("Pipeline complete")
    print(f"{'=' * 60}")
    print(f"Instances: {len(registry)}")
    print(f"Total time: {total_elapsed:.1f}s")
    if timings:
        print("\nStage timings:")
        for stage, t in timings.items():
            print(f"  {stage}: {t:.1f}s")

    return 0


if __name__ == "__main__":
    exit(main())
