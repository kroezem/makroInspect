import shutil
import numpy as np
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString
from sqlalchemy import create_engine, text
from sqlmodel import Session, select, func, delete

from core.database import Image, Instance, ConfigState, init_db

CONFIG_NAME = "config.yaml"
DB_NAME = "project.db"
DEFAULTS_NAME = "defaults.yaml"


class Project:
    def __init__(self, root_path: Path):
        self.root = root_path
        self.config_path = self.root / CONFIG_NAME
        self.db_path = self.root / DB_NAME

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found at {self.config_path}")

        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        with open(self.config_path, "r") as f:
            self.cfg = self.yaml.load(f)

        self.engine = create_engine(f"sqlite:///{self.db_path}")
        init_db(self.engine)

        # In-Memory Bank Cache
        self.bank_matrix: Optional[np.ndarray] = None
        self.bank_timestamp: float = 0.0

        # Sync state immediately upon loading
        self._sync_config_state()

    def save_config(self):
        with open(self.config_path, "w") as f:
            self.yaml.dump(self.cfg, f)

    def get_session(self) -> Session:
        return Session(self.engine)

    # --- FACTORY ---

    @classmethod
    def create(cls, name: str, description: str, projects_root: Path, defaults_path: Path) -> 'Project':
        target_dir = projects_root / name
        if target_dir.exists():
            raise FileExistsError(f"Project '{name}' already exists.")

        target_dir.mkdir(parents=True)

        yaml = YAML()
        yaml.preserve_quotes = True
        if not defaults_path.exists():
            raise FileNotFoundError(f"Defaults not found at {defaults_path}")

        with open(defaults_path, "r") as f:
            default_cfg = yaml.load(f)

        if "segmentation" not in default_cfg:
            default_cfg["segmentation"] = {}
        default_cfg["segmentation"]["description"] = DoubleQuotedScalarString(description)

        with open(target_dir / CONFIG_NAME, "w") as f:
            yaml.dump(default_cfg, f)

        (target_dir / "inbox").mkdir()
        artifacts = target_dir / "artifacts"

        folders = ["raw", "masks", "crop_images", "crop_masks", "embeddings", "bank", "heatmaps"]
        for folder in folders:
            (artifacts / folder).mkdir(parents=True, exist_ok=True)

        return cls(target_dir)

    # --- PATH HELPERS ---

    def get_raw_path(self, image: Image) -> Path:
        return self.root / "artifacts" / "raw" / f"{image.id:09d}.{image.extension}"

    # --- QUERIES ---

    def get_pending_segmentation(self, limit: int = 8) -> List[Image]:
        with self.get_session() as session:
            return session.exec(
                select(Image).where(Image.segmented_at == None).order_by(Image.ingested_at).limit(limit)
            ).all()

    def get_pending_crops(self, limit: int = 16) -> List[Instance]:
        from sqlalchemy.orm import joinedload
        with self.get_session() as session:
            return session.exec(
                select(Instance).options(joinedload(Instance.image)).where(Instance.cropped_at == None).limit(limit)
            ).all()

    def get_pending_evaluation(self, limit: int = 32) -> List[Instance]:
        with self.get_session() as session:
            return session.exec(
                select(Instance).where(Instance.cropped_at != None).where(Instance.evaluated_at == None).order_by(
                    Instance.cropped_at).limit(limit)
            ).all()

    # --- BANK MANAGEMENT ---

    def load_bank(self):
        bank_dir = self.root / "artifacts" / "bank"
        if not bank_dir.exists(): return
        try:
            current_mtime = bank_dir.stat().st_mtime
            if self.bank_matrix is not None and current_mtime <= self.bank_timestamp: return
        except OSError:
            return

        features = []
        for p in sorted(bank_dir.glob("*.npy")):
            try:
                data = np.load(p)
                features.append(data.reshape(-1, data.shape[-1]))
            except Exception:
                pass

        if features:
            self.bank_matrix = np.concatenate(features, axis=0)
            self.bank_timestamp = current_mtime
        else:
            self.bank_matrix = None

    # --- INVALIDATION ---

    def invalidate(self, stage: str):
        stage = stage.lower().strip()
        print(f"[{self.root.name}] ♻️  INVALIDATING from '{stage}' downstream...")

        artifacts = self.root / "artifacts"

        with self.get_session() as session:
            # LEVEL 1: EVALUATION / BANK
            if stage in ["segment", "crop", "embed", "evaluate", "bank"]:
                session.exec(
                    text("UPDATE instance SET evaluated_at = NULL, score = NULL, in_bank = 0")
                )
                self._wipe_dir(artifacts / "embeddings")
                self._wipe_dir(artifacts / "bank")
                self._wipe_dir(artifacts / "heatmaps")

            # LEVEL 2: CROPS
            if stage in ["segment", "crop"]:
                session.exec(
                    text("UPDATE instance SET cropped_at = NULL, obb_json = '{}', crop_iou = NULL")
                )
                self._wipe_dir(artifacts / "crop_images")
                self._wipe_dir(artifacts / "crop_masks")

            # LEVEL 3: SEGMENTATION
            if stage == "segment":
                session.exec(delete(Instance))
                session.exec(text("UPDATE image SET segmented_at = NULL"))
                self._wipe_dir(artifacts / "masks")

            session.commit()
        print(f"[{self.root.name}] ✅ Invalidation complete.")

    def _wipe_dir(self, path: Path):
        if not path.exists(): return
        for item in path.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                print(f"Warning: Failed to delete {item}: {e}")

    # --- STATE MANAGEMENT ---

    def _sync_config_state(self):
        """Checks YAML vs DB state and triggers cascading invalidation on mismatch."""
        checks = {
            # Segmentation
            "segmentation.description": "segment",
            "segmentation.enabled": "segment",
            "segmentation.multi_instance": "segment",

            # Transformer & Scaling
            "transformer.patch_size": "crop",
            "transformer.output_resolution": "crop",

            # Crop Logic
            "crop.reference_id": "crop",
            "crop.offset_rotation": "crop",
            "crop.scale": "crop",
            "crop.angle_candidates": "crop",

            # Model & Scoring
            "model.model_id": "evaluate",
            "anomaly_score.method": "evaluate",

            # Bank Logic
            "bank.size": "bank"
        }

        with self.get_session() as session:
            highest_invalidation = None
            severity = {"segment": 3, "crop": 2, "evaluate": 1, "bank": 0}

            for path, stage in checks.items():
                # Robust nested key retrieval
                keys = path.split('.')
                val = self.cfg
                for k in keys:
                    val = val.get(k, {}) if isinstance(val, dict) else None

                # Normalize value to string for comparison
                current_val = str(val) if val is not None else "None"

                db_record = session.get(ConfigState, path)

                if not db_record:
                    # FIRST RUN CASE: Just track it, don't explode.
                    print(f"[{self.root.name}] ️  Tracking new config: {path} = {current_val}")
                    session.add(ConfigState(key=path, value=current_val))
                elif db_record.value != current_val:
                    # CHANGE DETECTED CASE
                    print(f"[{self.root.name}] ⚠️  Config Change: {path} ('{db_record.value}' -> '{current_val}')")

                    if highest_invalidation is None or severity[stage] > severity[highest_invalidation]:
                        highest_invalidation = stage

                    db_record.value = current_val
                    db_record.updated_at = datetime.utcnow()
                    session.add(db_record)

            session.commit()

            if highest_invalidation:
                self.invalidate(highest_invalidation)
