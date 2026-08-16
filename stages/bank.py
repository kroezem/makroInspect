"""
Bank stage - K-center greedy selection of memory bank.

Reads:
    artifacts/embed/{split}/{label}/{instance_id}.npy
Writes:
    artifacts/bank/embeddings.npy  (float16, K×H×W×C)
    artifacts/bank/instances.txt   (list of instance IDs)
Updates registry: in_bank
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from core.paths import ProjectPaths


def process(paths: ProjectPaths, cfg: dict, registry: pd.DataFrame) -> pd.DataFrame:
    """
    Select memory bank instances using k-center greedy.

    Returns updated registry with in_bank column populated.
    """
    if paths.bank_embeddings.exists():
        print("✓ Bank already exists, skipping")
        return registry

    if not paths.embed_dir.exists():
        print("No embeddings found. Run embed stage first.")
        return registry

    # Get bank config
    bank_cfg = cfg.get("memory_bank", {})
    max_gb = float(bank_cfg.get("max_gb", 0.5))
    selection_method = bank_cfg.get("selection_method", "k_center")

    # Get embedding info from first file
    train_good = registry[(registry["split"] == "train") & (registry["label"] == "good")]
    if train_good.empty:
        print("No train/good instances found for bank selection")
        return registry

    first_instance = train_good.iloc[0]
    first_embed_path = paths.embedding(
        first_instance["split"],
        first_instance["label"],
        first_instance["instance_id"]
    )
    first_embed = np.load(first_embed_path)
    H, W, C = first_embed.shape

    # Calculate bytes per instance (stored as float16)
    bytes_per_instance = H * W * C * 2  # float16 = 2 bytes
    max_bytes = max_gb * (1024 ** 3)

    # Compute k (max instances that fit in GB budget)
    k = int(max_bytes // bytes_per_instance)
    k = min(k, len(train_good))

    print(f"  Embedding shape: {H}×{W}×{C}")
    print(f"  Bytes per instance: {bytes_per_instance:,} ({bytes_per_instance / 1024 / 1024:.2f} MB)")
    print(f"  Max bank size: {max_gb} GB")
    print(f"  Bank capacity (k): {k}")
    print(f"  Selection method: {selection_method}")

    # Load pooled embeddings for selection (mean over spatial dims)
    print()
    print("  Loading pooled embeddings for selection...")
    print()

    instance_ids = train_good["instance_id"].tolist()
    pooled = []
    valid_ids = []

    for instance_id in tqdm(instance_ids, desc="  Loading", unit="inst"):
        row = train_good[train_good["instance_id"] == instance_id].iloc[0]
        embed_path = paths.embedding(row["split"], row["label"], instance_id)

        if not embed_path.exists():
            continue

        embed = np.load(embed_path)  # (H, W, C)
        pooled_embed = embed.mean(axis=(0, 1))  # (C,)
        pooled.append(pooled_embed)
        valid_ids.append(instance_id)

    print()
    pooled = np.stack(pooled).astype(np.float32)  # (N, C)
    print(f"  Loaded {len(pooled)} pooled embeddings")

    # Select bank instances
    print()
    if selection_method == "k_center":
        print("  Running k-center greedy selection...")
        print()
        selected_indices = _select_kcenter_cosine_gpu(pooled, k)
        print()
    else:
        print("  Running random selection...")
        selected_indices = _select_random(pooled, k)

    selected_ids = [valid_ids[i] for i in selected_indices]
    print(f"  Selected {len(selected_ids)} instances")

    # Load full patch embeddings for selected instances
    print()
    print("  Loading patch embeddings for bank...")
    print()

    bank_patches = []
    for instance_id in tqdm(selected_ids, desc="  Loading", unit="inst"):
        row = train_good[train_good["instance_id"] == instance_id].iloc[0]
        embed_path = paths.embedding(row["split"], row["label"], instance_id)
        embed = np.load(embed_path)
        bank_patches.append(embed)

    print()
    bank_patches = np.stack(bank_patches).astype(np.float16)  # (K, H, W, C)
    print(f"  Bank shape: {bank_patches.shape}")

    # Save bank
    paths.bank_dir.mkdir(parents=True, exist_ok=True)

    np.save(paths.bank_embeddings, bank_patches)
    bank_size_gb = bank_patches.nbytes / (1024 ** 3)
    print(f"  ✓ Saved {paths.bank_embeddings} ({bank_size_gb:.2f} GB)")

    with open(paths.bank_instances, "w") as f:
        for instance_id in selected_ids:
            f.write(f"{instance_id}\n")
    print(f"  ✓ Saved {paths.bank_instances}")

    # Update registry
    registry["in_bank"] = registry["instance_id"].isin(selected_ids)
    n_in_bank = registry["in_bank"].sum()
    print(f"  ✓ Marked {n_in_bank} instances as in_bank")

    return registry


def _select_random(pooled: np.ndarray, k: int) -> list[int]:
    """Random selection of k instances."""
    return np.random.choice(len(pooled), k, replace=False).tolist()


def _select_kcenter_cosine_gpu(pooled: np.ndarray, k: int) -> list[int]:
    """
    K-center greedy selection using cosine similarity on GPU.

    Iteratively selects the instance most dissimilar to the current set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize and compute similarity matrix
    M_tensor = torch.from_numpy(pooled).to(device).float()
    M_norm = torch.nn.functional.normalize(M_tensor, p=2, dim=1)
    sim_matrix = torch.matmul(M_norm, M_norm.T)

    # Start with random instance
    indices = [np.random.randint(0, len(pooled))]
    current_max_sim = sim_matrix[:, indices[0]]

    # Greedily add most distant instance
    for _ in tqdm(range(1, k), desc="  K-center"):
        farthest = torch.argmin(current_max_sim).item()
        indices.append(farthest)
        current_max_sim = torch.max(current_max_sim, sim_matrix[:, farthest])

    if device.type == "cuda":
        torch.cuda.synchronize()

    return indices