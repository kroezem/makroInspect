"""
Centralized path management for makroInspect projects.
All path construction happens here - no path strings elsewhere in codebase.

Structure:
    artifacts/{project}/     - Regenerable (~30GB per project)
        data/                - Dataset (copied from datasets/)
        masks/
        crops/
        embed/
        bank/
        heatmaps/
        instances.csv        - Base instance metadata
        .hashes/

    projects/{project}/      - Permanent (~5MB per project)
        config.yaml
        runs.csv
        runs/
            run_XXX/
                config.yaml  - Frozen snapshot
                instances.csv - With scores

    datasets/                - Downloaded source datasets
        visa/
        mvtec/
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = REPO_ROOT / "projects"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DATASETS_DIR = REPO_ROOT / "datasets"
CONFIG_DIR = REPO_ROOT / "config"


class ProjectPaths:
    """
    Centralized path management for a single makroInspect project.

    All artifact and project paths are constructed through this class to ensure
    consistency and prevent path string duplication across the codebase.

    Attributes:
        project: The project name (e.g., "capsules")
        project_root: Permanent project folder (projects/{project}/)
        artifacts_root: Regenerable artifacts folder (artifacts/{project}/)

    Example:
        >>> paths = ProjectPaths("capsules")
        >>> paths.data_dir
        PosixPath('artifacts/capsules/data')
        >>> paths.mask("train", "good", "001_00")
        PosixPath('artifacts/capsules/masks/train/good/001_00.png')
    """

    project: str
    project_root: Path
    artifacts_root: Path

    def __init__(self, project: str) -> None:
        self.project = project
        self.project_root = PROJECTS_DIR / project
        self.artifacts_root = ARTIFACTS_DIR / project

    # === Data (input, lives in artifacts) ===

    @property
    def data_dir(self) -> Path:
        return self.artifacts_root / "data"

    def data_split_label(self, split: str, label: str) -> Path:
        return self.data_dir / split / label

    def ground_truth_dir(self, label: str) -> Path:
        return self.data_dir / "ground_truth" / label

    # === Masks (segment output) ===

    @property
    def masks_dir(self) -> Path:
        return self.artifacts_root / "masks"

    def mask(self, split: str, label: str, instance_id: str) -> Path:
        return self.masks_dir / split / label / f"{instance_id}.png"

    # === Crops (crop output) ===

    @property
    def crops_dir(self) -> Path:
        return self.artifacts_root / "crops"

    def crop_image(self, split: str, label: str, instance_id: str) -> Path:
        return self.crops_dir / "images" / split / label / f"{instance_id}.png"

    def crop_mask(self, split: str, label: str, instance_id: str) -> Path:
        return self.crops_dir / "masks" / split / label / f"{instance_id}.png"

    def crop_metadata(self, split: str, label: str, stem: str) -> Path:
        return self.crops_dir / "metadata" / split / label / f"{stem}.json"

    # === Embeddings (embed output) ===

    @property
    def embed_dir(self) -> Path:
        return self.artifacts_root / "embed"

    def embedding(self, split: str, label: str, instance_id: str) -> Path:
        return self.embed_dir / split / label / f"{instance_id}.npy"

    # === Bank (bank output) ===

    @property
    def bank_dir(self) -> Path:
        return self.artifacts_root / "bank"

    @property
    def bank_embeddings(self) -> Path:
        return self.bank_dir / "embeddings.npy"

    @property
    def bank_instances(self) -> Path:
        return self.bank_dir / "instances.txt"

    # === Heatmaps (heatmap output) ===

    @property
    def heatmaps_dir(self) -> Path:
        return self.artifacts_root / "heatmaps"

    def heatmap_instance(self, split: str, label: str, instance_id: str) -> Path:
        return self.heatmaps_dir / "instance" / split / label / f"{instance_id}.npy"

    def heatmap_reverted(self, split: str, label: str, stem: str) -> Path:
        return self.heatmaps_dir / "reverted" / split / label / f"{stem}.npy"

    @property
    def refine_dir(self) -> Path:
        return self.heatmaps_dir / "refined"

    # === Registry (in artifacts) ===

    @property
    def instances_base(self) -> Path:
        return self.artifacts_root / "instances.csv"

    # === Config hashes (in artifacts) ===

    @property
    def hashes_dir(self) -> Path:
        return self.artifacts_root / ".hashes"

    def config_hash(self, stage: str) -> Path:
        return self.hashes_dir / f"{stage}.hash"

    # === Project config and runs (permanent) ===

    @property
    def config(self) -> Path:
        return self.project_root / "config.yaml"

    @property
    def runs_csv(self) -> Path:
        return self.project_root / "runs.csv"

    @property
    def runs_dir(self) -> Path:
        return self.project_root / "runs"

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def run_config(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "config.yaml"

    def run_instances(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "instances.csv"

    # === Helpers ===

    def exists(self) -> bool:
        """Check if project exists (project folder and config file)."""
        return self.project_root.exists() and self.config.exists()

    def artifacts_exist(self) -> bool:
        """Check if artifacts folder exists."""
        return self.artifacts_root.exists()

    def data_exists(self) -> bool:
        """Check if data has been prepared."""
        return self.data_dir.exists() and any(self.data_dir.rglob("*"))
