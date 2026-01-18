import numpy as np
from typing import List
from dataclasses import dataclass

from sqlmodel import select
from core.project import Project
from core.database import Instance


@dataclass
class BankBuildResult:
    project_name: str
    regenerated: int
    total_slots: int
    success: bool


class BankBuilder:
    """
    The Builder: GPU worker for bank reconstruction.

    Responsibilities:
    - Check if Current Bank != Target Bank (from DB)
    - Use GPU to reconstruct bank.npy
    - Sync to VRAM

    Only runs in Maintenance Lane (when system is idle).
    Reads bank composition from Instance.bank_index in DB.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    def needs_initial_build(self, project: Project) -> bool:
        """
        Bootstrap check: No bank exists but we have enough normal candidates.
        This breaks the chicken-and-egg problem where bank_index hasn't been set yet.
        """
        proj_name = project.root.name
        bank_path = project.root / "bank.npy"

        if bank_path.exists():
            return False

        embed_dir = project.root / "artifacts" / "embeddings"
        if not embed_dir.exists():
            return False

        with project.get_session() as session:
            stmt = (
                select(Instance)
                .where(Instance.cropped_at != None)
                .where(Instance.label == "normal")
            )
            candidates = list(session.execute(stmt).scalars().all())

        if not candidates:
            return False

        count = sum(1 for c in candidates if (embed_dir / f"{c.id}.npy").exists())

        if count >= 10:
            print(f"[BANK]  {proj_name:<10} | Bootstrap ready: {count} normal with embeddings")
            return True

        return False

    def needs_rebuild(self, project: Project) -> bool:
        """Check if project's bank needs rebuilding based on DB state."""
        proj_name = project.root.name
        target_ids = project.get_bank_target_ids()

        if not target_ids:
            return False

        bank_path = project.root / "bank.npy"
        if not bank_path.exists():
            print(f"[BANK]  {proj_name:<10} | Rebuild needed - no bank.npy")
            return True

        embed_dir = project.root / "artifacts" / "embeddings"
        missing = [i for i in target_ids if not (embed_dir / f"{i}.npy").exists()]
        if missing:
            print(f"[BANK]  {proj_name:<10} | Waiting for {len(missing)} embeddings")
            return False

        bank_mtime = bank_path.stat().st_mtime
        crops_dir = project.root / "artifacts" / "crop_images"

        stale = 0
        for inst_id in target_ids:
            crop_path = crops_dir / f"{inst_id}.png"
            if crop_path.exists() and crop_path.stat().st_mtime > bank_mtime:
                stale += 1

        if stale > 0:
            print(f"[BANK]  {proj_name:<10} | Rebuild needed - {stale} crops newer than bank")
            return True

        try:
            bank = np.load(bank_path)
            if bank.shape[0] != len(target_ids):
                print(f"[BANK]  {proj_name:<10} | Rebuild needed - size mismatch ({bank.shape[0]} vs {len(target_ids)})")
                return True
        except Exception as e:
            print(f"[BANK]  {proj_name:<10} | Rebuild needed - load error: {e}")
            return True

        return False

    def build(self, project: Project, models) -> BankBuildResult:
        """
        Rebuild the bank using GPU feature extraction.
        Reads target composition from DB (Instance.bank_index >= 0).
        """
        proj_name = project.root.name

        with project.get_session() as session:
            stmt = (
                select(Instance)
                .where(Instance.bank_index != None)
                .where(Instance.bank_index >= 0)
                .order_by(Instance.bank_index)
            )
            target_instances = list(session.execute(stmt).scalars().all())

        if not target_instances:
            print(f"[BANK]  {proj_name:<10} | Build failed - no indexed instances")
            return BankBuildResult(proj_name, 0, 0, False)

        target_n = len(target_instances)
        target_ids = [inst.id for inst in target_instances]

        paths_to_regen = []
        slots_to_regen = []

        crops_dir = project.root / "artifacts" / "crop_images"
        for slot, inst_id in enumerate(target_ids):
            crop_path = crops_dir / f"{inst_id}.png"
            if crop_path.exists():
                paths_to_regen.append(crop_path)
                slots_to_regen.append(slot)

        if not paths_to_regen:
            print(f"[BANK]  {proj_name:<10} | Build failed - no crop images found")
            return BankBuildResult(proj_name, 0, 0, False)

        trans_cfg = project.cfg.get("transformer", {})
        grid = trans_cfg.get("map_resolution", [48, 32])

        bank_arr = np.zeros((target_n, int(grid[1]), int(grid[0]), 768), dtype=np.float32)

        print(f"[BANK]  {proj_name:<10} | Building with {len(paths_to_regen)} crops...")
        feats = models.compute_patches(paths_to_regen)
        if feats is None:
            print(f"[BANK]  {proj_name:<10} | Build failed - feature extraction returned None")
            return BankBuildResult(proj_name, 0, target_n, False)

        feats_np = feats.cpu().numpy().astype(np.float32)
        regenerated = 0

        for i, slot in enumerate(slots_to_regen):
            if i < feats_np.shape[0]:
                bank_arr[slot] = feats_np[i]
                regenerated += 1

        project.save_bank(bank_arr)

        return BankBuildResult(
            project_name=proj_name,
            regenerated=regenerated,
            total_slots=target_n,
            success=True
        )

    def build_all(self, projects: List[Project], models) -> List[BankBuildResult]:
        """Build banks for all projects that need it."""
        results = []
        for project in projects:
            if self.needs_rebuild(project):
                result = self.build(project, models)
                results.append(result)
        return results
