"""
Core infrastructure for makroInspect.

Provides path management, registry schemas, and config invalidation tracking.

Example:
    >>> from core import ProjectPaths, STAGE_ORDER
    >>> paths = ProjectPaths("capsules")
    >>> print(paths.data_dir)
"""

from core.paths import (
    REPO_ROOT,
    PROJECTS_DIR,
    ARTIFACTS_DIR,
    DATASETS_DIR,
    CONFIG_DIR,
    ProjectPaths,
)
from core.registry import (
    INSTANCE_BASE_COLUMNS,
    INSTANCE_ALL_COLUMNS,
    RUNS_COLUMNS,
    create_empty_registry,
    load_instances_base,
    save_instances_base,
    load_run_instances,
    save_run_instances,
    create_empty_runs_csv,
    load_runs,
    append_run,
    generate_run_id,
)
from core.invalidation import (
    STAGE_ORDER,
    compute_invalid_stages,
    invalidate_stage_artifacts,
    save_hash,
    compute_stage_hash,
)

__all__ = [
    # paths
    "REPO_ROOT",
    "PROJECTS_DIR",
    "ARTIFACTS_DIR",
    "DATASETS_DIR",
    "CONFIG_DIR",
    "ProjectPaths",
    # registry
    "INSTANCE_BASE_COLUMNS",
    "INSTANCE_ALL_COLUMNS",
    "RUNS_COLUMNS",
    "create_empty_registry",
    "load_instances_base",
    "save_instances_base",
    "load_run_instances",
    "save_run_instances",
    "create_empty_runs_csv",
    "load_runs",
    "append_run",
    "generate_run_id",
    # invalidation
    "STAGE_ORDER",
    "compute_invalid_stages",
    "invalidate_stage_artifacts",
    "save_hash",
    "compute_stage_hash",
]
