import uuid
import shutil
import numpy as np
from pathlib import Path
from ruamel.yaml import YAML
from datetime import datetime
from typing import List, Optional

from sqlmodel import select, create_engine, desc, or_, func, text, delete
from sqlalchemy.orm import scoped_session, sessionmaker

from core.database import Image, Instance, ConfigState, init_db


class Project:
    def __init__(self, root: Path):
        self.root = root
        self.inbox_path = self.root / "inbox"
        self.db_path = self.root / "project.db"
        self._bank_cache = None

        # 1. Database Connection
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            connect_args={"timeout": 10}
        )
        init_db(self.engine)
        self.SessionLocal = scoped_session(sessionmaker(bind=self.engine))

        # 2. Config State Check
        # This will auto-invalidate data if config changed since last run
        self._sync_config_state()

    @classmethod
    def create(cls, name: str, desc: str, root_root: Path, defaults_path: Path):
        proj_root = root_root / name
        proj_root.mkdir(parents=True, exist_ok=True)

        folders = ["inbox", "artifacts/raw", "artifacts/masks", "artifacts/crop_images",
                   "artifacts/crop_masks", "artifacts/embeddings", "artifacts/heatmaps"]
        for f in folders: (proj_root / f).mkdir(parents=True, exist_ok=True)

        yaml = YAML()
        yaml.preserve_quotes = True
        with open(defaults_path, 'r') as f:
            base_cfg = yaml.load(f)
        base_cfg['segmentation']['description'] = desc
        with open(proj_root / "config.yaml", 'w') as f:
            yaml.dump(base_cfg, f)

        return cls(proj_root)

    def get_session(self):
        return self.SessionLocal()

    @property
    def cfg(self) -> dict:
        cfg_path = self.root / "config.yaml"
        if not cfg_path.exists():
            return {}
        yaml = YAML()
        with open(cfg_path, 'r') as f:
            return yaml.load(f) or {}

    def get_raw_path(self, img: Image) -> Path:
        return self.root / "artifacts" / "raw" / f"{img.id:09d}.{img.extension}"

    # --- WORK QUEUES ---
    def get_pending_images(self, min_priority: int = 0, limit: int = 10) -> List[Image]:
        with self.get_session() as session:
            stmt = select(Image).where(Image.segmented_at == None).where(Image.priority >= min_priority).order_by(
                desc(Image.priority), Image.ingested_at).limit(limit)
            return list(session.execute(stmt).scalars().all())

    def get_pending_crops(self, min_priority: int = 0, limit: int = 50) -> List[Instance]:
        with self.get_session() as session:
            stmt = select(Instance).where(Instance.cropped_at == None).where(
                Instance.priority >= min_priority).order_by(desc(Instance.priority), Instance.created_at).limit(limit)
            return list(session.execute(stmt).scalars().all())

    def get_pending_evaluation(self, min_priority: int = 0, limit: int = 32) -> List[Instance]:
        with self.get_session() as session:
            # LIFO for evaluation to prioritize new uploads, but handle old refactors via priority
            stmt = select(Instance).where(Instance.evaluated_at == None).where(Instance.cropped_at != None).where(
                Instance.priority >= min_priority).order_by(desc(Instance.priority), desc(Instance.cropped_at)).limit(
                limit)
            return list(session.execute(stmt).scalars().all())

    def get_total_pending_eval(self, min_priority: int = 0) -> int:
        with self.get_session() as session:
            stmt = select(func.count(Instance.id)).where(Instance.evaluated_at == None).where(
                Instance.cropped_at != None).where(Instance.priority >= min_priority)
            return session.execute(stmt).scalar() or 0

    # --- BANK MANAGEMENT ---
    @property
    def bank_matrix(self) -> Optional[np.ndarray]:
        bank_path = self.root / "bank.npy"
        if not bank_path.exists(): return None
        if self._bank_cache is not None: return self._bank_cache
        try:
            self._bank_cache = np.load(bank_path)
            return self._bank_cache
        except:
            return None

    @property
    def current_bank_id(self) -> Optional[str]:
        with self.get_session() as session:
            res = session.get(ConfigState, "active_bank_id")
            return res.value if res else None

    def save_bank(self, matrix: np.ndarray):
        np.save(self.root / "bank.npy", matrix)
        self._bank_cache = matrix
        new_id = str(uuid.uuid4())
        with self.get_session() as session:
            session.merge(ConfigState(key="active_bank_id", value=new_id))
            session.commit()
        return new_id

    def get_maintenance_candidates(self, limit=100) -> int:
        """
        Find instances that need rescoring because bank changed.
        Resets their evaluated_at and sets low priority for re-evaluation.
        """
        curr_uuid = self.current_bank_id
        if not curr_uuid:
            return 0
        with self.get_session() as session:
            stmt = (
                select(Instance)
                .where(Instance.evaluated_at != None)
                .where(Instance.cropped_at != None)
                .where(or_(Instance.bank_id != curr_uuid, Instance.bank_id == None))
                .order_by(Instance.evaluated_at.asc())  # Oldest first
                .limit(limit)
            )
            candidates = list(session.execute(stmt).scalars().all())
            if not candidates:
                return 0
            for inst in candidates:
                inst.priority = 1
                inst.evaluated_at = None
                session.add(inst)
            session.commit()
            return len(candidates)

    def get_normal_average_score(self) -> float:
        """
        Get the average anomaly score of all normal samples.
        Used as baseline for heatmap rendering.
        """
        with self.get_session() as session:
            stmt = (
                select(func.avg(Instance.score))
                .where(Instance.label == "normal")
                .where(Instance.score != None)
            )
            avg = session.execute(stmt).scalar()
            return float(avg) if avg else 0.0

    def get_bank_target_ids(self) -> list:
        """Get instance IDs that should be in the bank (have bank_index >= 0)."""
        with self.get_session() as session:
            stmt = (
                select(Instance.id, Instance.bank_index)
                .where(Instance.bank_index != None)
                .where(Instance.bank_index >= 0)
                .order_by(Instance.bank_index)
            )
            results = session.execute(stmt).all()
            return [r[0] for r in results]

    def get_pending_rescore_count(self) -> int:
        """Count instances that need rescoring due to bank change."""
        curr_uuid = self.current_bank_id
        if not curr_uuid:
            return 0
        with self.get_session() as session:
            stmt = (
                select(func.count(Instance.id))
                .where(Instance.evaluated_at != None)
                .where(Instance.cropped_at != None)
                .where(or_(Instance.bank_id != curr_uuid, Instance.bank_id == None))
            )
            return session.execute(stmt).scalar() or 0

    # --- INVALIDATION LOGIC ---

    def _sync_config_state(self):
        """Checks YAML vs DB state and triggers cascading invalidation on mismatch."""
        # Maps config keys to the 'highest' stage they affect.
        # Changing a key invalidates that stage AND everything downstream.
        checks = {
            # 1. Segmentation
            "segmentation.description": "segment",
            "segmentation.enabled": "segment",
            "segmentation.multi_instance": "segment",

            # 2. Crop
            "crop.scale": "crop",
            "crop.offset_rotation": "crop",
            "crop.align_PCA": "crop",
            "crop.angle_candidates": "crop",
            "crop.reference_id": "crop",

            # 3. Evaluate (Embedding/Scoring)
            "transformer.patch_size": "evaluate",  # Changes embedding dimensions
            "transformer.map_resolution": "evaluate",  # Changes embedding dimensions
            "anomaly_score.method": "evaluate",
            "anomaly_score.cmap": "evaluate",
            "anomaly_score.min_percentile": "evaluate",
            "anomaly_score.max_percentile": "evaluate",

            # 4. Bank (Curation)
            "memory_bank.replacement.policy": "bank"
        }

        severity = {"segment": 4, "crop": 3, "evaluate": 2, "bank": 1}
        highest_invalidation = None

        with self.get_session() as session:
            for path, stage in checks.items():
                # Extract value from YAML
                keys = path.split('.')
                val = self.cfg
                try:
                    for k in keys: val = val[k]
                    current_val = str(val)
                except (KeyError, TypeError):
                    current_val = "None"

                # Check against DB
                db_record = session.get(ConfigState, path)

                if not db_record:
                    # Initial Tracking
                    session.add(ConfigState(key=path, value=current_val))
                elif db_record.value != current_val:
                    # CHANGE DETECTED
                    print(f"[{self.root.name}] ⚠️  Config Change: {path} ('{db_record.value}' -> '{current_val}')")

                    if highest_invalidation is None or severity[stage] > severity[highest_invalidation]:
                        highest_invalidation = stage

                    db_record.value = current_val
                    session.add(db_record)

            session.commit()

        if highest_invalidation:
            self.invalidate(highest_invalidation)

    def invalidate(self, stage: str):
        """
        Wipes data downstream from the changed stage.
        Stages: segment -> crop -> evaluate -> bank
        """
        print(f"[{self.root.name}] ♻️  INVALIDATING downstream from '{stage}'...")
        artifacts = self.root / "artifacts"

        with self.get_session() as session:
            # LEVEL 1: BANK (Reset Index & Delete File)
            if stage in ["segment", "crop", "evaluate", "bank"]:
                session.execute(text("UPDATE instance SET bank_index = NULL"))
                (artifacts / "bank.npy").unlink(missing_ok=True)
                self._bank_cache = None

            # LEVEL 2: EVALUATION (Reset Scores & Delete Embeddings/Heatmaps)
            if stage in ["segment", "crop", "evaluate"]:
                session.execute(text("UPDATE instance SET evaluated_at = NULL, score = NULL, bank_id = NULL"))
                self._wipe_dir(artifacts / "embeddings")
                self._wipe_dir(artifacts / "heatmaps")

            # LEVEL 3: CROP (Reset Crops & Delete Images)
            if stage in ["segment", "crop"]:
                session.execute(text("UPDATE instance SET cropped_at = NULL, crop_iou = NULL, obb_json = '{}'"))
                self._wipe_dir(artifacts / "crop_images")
                self._wipe_dir(artifacts / "crop_masks")

            # LEVEL 4: SEGMENTATION (Delete Instances & Reset Image)
            if stage == "segment":
                # Delete all instances (Cascades to evaluated_at, etc.)
                session.execute(delete(Instance))
                session.execute(text("UPDATE image SET segmented_at = NULL"))
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
            except Exception:
                pass