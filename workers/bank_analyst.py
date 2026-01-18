import time
import threading
import numpy as np
from typing import List, Optional, Callable
from dataclasses import dataclass

from sqlmodel import select
from core.database import Instance


@dataclass
class AnalystResult:
    project_name: str
    selected_count: int
    total_candidates: int
    changed: bool


class BankAnalyst(threading.Thread):
    """
    The Analyst: Background daemon thread for bank composition analysis.

    Responsibilities:
    - Watch DB for new normal embeddings
    - Perform K-Center Greedy selection (CPU-bound math)
    - Write bank_index to DB for selected instances

    Only considers instances with label="normal" for bank inclusion.
    """

    def __init__(
        self,
        projects: List,
        target_bank_size: int = 128,
        poll_interval: float = 5.0,
        on_update: Optional[Callable[[AnalystResult], None]] = None
    ):
        super().__init__(daemon=True, name="BankAnalyst")
        self.projects = projects
        self.target_bank_size = target_bank_size
        self.poll_interval = poll_interval
        self.on_update = on_update
        self._stop_event = threading.Event()
        self._pending_rebuild = threading.Event()
        self._last_hashes = {}

    def stop(self):
        """Signal the thread to stop."""
        self._stop_event.set()

    def has_pending_rebuild(self) -> bool:
        """Check if analyst has marked a project for rebuild."""
        return self._pending_rebuild.is_set()

    def clear_pending_rebuild(self):
        """Clear the pending rebuild flag."""
        self._pending_rebuild.clear()

    def run(self):
        """Main loop: poll projects for changes and recompute targets."""
        while not self._stop_event.is_set():
            for project in self.projects:
                try:
                    result = self._analyze_project(project)
                    if result and result.changed and self.on_update:
                        self.on_update(result)
                except Exception:
                    pass

            self._stop_event.wait(self.poll_interval)

    def _analyze_project(self, project) -> Optional[AnalystResult]:
        """Check if project needs reanalysis and compute new target if needed."""
        proj_name = project.root.name
        embed_dir = project.root / "artifacts" / "embeddings"

        if not embed_dir.exists():
            return None

        with project.get_session() as session:
            # Only consider normal samples for bank
            stmt = (
                select(Instance)
                .where(Instance.cropped_at != None)
                .where(Instance.label == "normal")
            )
            candidates = list(session.execute(stmt).scalars().all())

        if len(candidates) < 10:
            return None

        # Check if anything changed by hashing candidate IDs and their embedding existence
        current_ids = sorted([c.id for c in candidates])
        current_hash = hash(tuple(current_ids))

        if self._last_hashes.get(proj_name) == current_hash:
            return None

        # Load embeddings
        vectors = []
        inst_ids = []

        for inst in candidates:
            embed_path = embed_dir / f"{inst.id}.npy"
            if embed_path.exists():
                try:
                    vec = np.load(embed_path).flatten()
                    vectors.append(vec)
                    inst_ids.append(inst.id)
                except:
                    pass

        if len(vectors) < 10:
            return None

        # K-Center Greedy selection
        matrix = np.array(vectors, dtype=np.float32)
        indices = self._k_center_greedy(matrix, self.target_bank_size)
        selected_ids = {inst_ids[i]: slot for slot, i in enumerate(indices)}

        # Check if selection actually changed
        current_bank_ids = set(project.get_bank_target_ids())
        new_bank_ids = set(selected_ids.keys())

        if current_bank_ids == new_bank_ids:
            self._last_hashes[proj_name] = current_hash
            return AnalystResult(proj_name, len(selected_ids), len(vectors), changed=False)

        # Write to DB
        with project.get_session() as session:
            # Get all normal candidates
            stmt = (
                select(Instance)
                .where(Instance.cropped_at != None)
                .where(Instance.label == "normal")
            )
            all_normal = list(session.execute(stmt).scalars().all())

            for inst in all_normal:
                db_inst = session.merge(inst)
                if db_inst.id in selected_ids:
                    db_inst.bank_index = selected_ids[db_inst.id]
                else:
                    db_inst.bank_index = -1
                session.add(db_inst)

            session.commit()
            self._pending_rebuild.set()  # Signal main loop

        self._last_hashes[proj_name] = current_hash

        return AnalystResult(
            project_name=proj_name,
            selected_count=len(selected_ids),
            total_candidates=len(vectors),
            changed=True
        )

    def _k_center_greedy(self, matrix: np.ndarray, k: int) -> List[int]:
        """
        K-Center Greedy algorithm for coreset selection.
        Selects k diverse samples that maximize coverage.
        """
        n_samples = matrix.shape[0]
        if n_samples <= k:
            return list(range(n_samples))

        centers = [0]
        min_dist = np.linalg.norm(matrix - matrix[centers[0]], axis=1)

        for _ in range(1, k):
            new_center = int(np.argmax(min_dist))
            centers.append(new_center)
            new_dist = np.linalg.norm(matrix - matrix[new_center], axis=1)
            min_dist = np.minimum(min_dist, new_dist)

        return centers

    def force_analyze(self, project) -> Optional[AnalystResult]:
        """Force immediate analysis of a single project (blocking call)."""
        self._last_hashes[project.root.name] = None
        return self._analyze_project(project)
