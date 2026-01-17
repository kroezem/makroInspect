import numpy as np
from pathlib import Path

from core.project import Project
from core.database import Instance
from workers.base import BaseWorker


class BankWorker(BaseWorker):
    def load_model(self):
        pass

    def process(self, project: Project) -> int:
        bank_cfg = project.cfg.get("bank", {})
        # Default to 50 if not set. If set to 0, bank is disabled.
        target_size = int(bank_cfg.get("size", 50))

        dir_embeds = project.root / "artifacts" / "embeddings"
        dir_bank = project.root / "artifacts" / "bank"
        dir_bank.mkdir(parents=True, exist_ok=True)

        # 1. Gather Candidates (From Lightweight Index)
        candidates = []
        ids = []

        # We scan the embeddings folder because it is the "Inbox" of available items
        for f in dir_embeds.glob("*.npy"):
            try:
                vec = np.load(f)
                candidates.append(vec)
                ids.append(f.stem)
            except:
                pass

        # If we have fewer candidates than the target, we generally just take them all
        # unless the pool is empty.
        if len(candidates) == 0:
            return 0

        effective_target = min(len(candidates), target_size)

        # 2. Select Winners (Coreset)
        all_vectors = np.stack(candidates)
        selected_indices = self._coreset_sampling(all_vectors, effective_target)
        selected_ids = set(ids[i] for i in selected_indices)

        updates = 0
        with project.get_session() as session:

            # 3. PURGE: Remove Losers from Disk & DB
            # We scan the BANK folder (heavy files) to see what needs to be deleted
            for f in dir_bank.glob("*.npy"):
                if f.stem not in selected_ids:
                    f.unlink()  # Delete heavy file

                    inst = session.get(Instance, f.stem)
                    if inst:
                        inst.in_bank = False
                        session.add(inst)
                        updates += 1

            # 4. REQUEST: Command Evaluator to fill the Bank
            for inst_id in selected_ids:
                inst = session.get(Instance, inst_id)
                if not inst: continue

                bank_file = dir_bank / f"{inst_id}.npy"

                # We need to trigger a job if:
                # A. The DB says it's not in the bank (flag update needed)
                # B. The DB says it IS in the bank, but the file is missing (corruption/deletion)
                needs_status_update = (inst.in_bank == False)
                needs_file_generation = (not bank_file.exists())

                if needs_status_update or needs_file_generation:
                    inst.in_bank = True
                    # FORCE RE-EVALUATION
                    # By setting evaluated_at to NULL, Evaluator picks it up again.
                    # Since in_bank is now True, Evaluator will save the patches this time.
                    inst.evaluated_at = None
                    session.add(inst)
                    updates += 1

            session.commit()

        return updates

    def _coreset_sampling(self, data: np.ndarray, n: int) -> list:
        """
        Greedy Farthest Point Sampling (k-Center).
        Selects 'n' points that minimize the maximum distance from any point to a selected point.
        """
        n_samples = data.shape[0]
        if n_samples <= n:
            return list(range(n_samples))

        # Deterministic start
        current_idx = 0
        selected_indices = [current_idx]

        # Squared Euclidean distance
        min_dists = np.sum((data - data[current_idx]) ** 2, axis=1)

        for _ in range(1, n):
            farthest_idx = np.argmax(min_dists)
            selected_indices.append(farthest_idx)

            new_dists = np.sum((data - data[farthest_idx]) ** 2, axis=1)
            min_dists = np.minimum(min_dists, new_dists)

        return selected_indices
