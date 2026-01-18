import os
import shutil
import time
import random
import sys
from pathlib import Path

# --- CONFIG ---
DATASET_ROOT = Path("dataset/VisA")
PROJECTS_ROOT = Path("projects")

# Just list the project names - they all follow the same structure
PROJECTS = ["capsules", "pcb4", "cashew", "macaroni1"]


def get_scenario_config(project_name):
    """Generate config for any VisA project."""
    return {
        "normal_src": DATASET_ROOT / project_name / "Data/Images/Normal",
        "anomaly_src": DATASET_ROOT / project_name / "Data/Images/Anomaly",
        "dst_root": PROJECTS_ROOT / project_name,
    }


def clean_slate():
    """Nukes everything except config.yaml."""
    print("--- 🧨 PERFORMING DEEP CLEAN ---")

    for proj_name in PROJECTS:
        cfg = get_scenario_config(proj_name)
        root = cfg["dst_root"]

        if not root.exists():
            print(f"[{proj_name}] Root not found, skipping.")
            continue

        # 1. Identify items to delete
        for item in root.iterdir():
            if item.name == "config.yaml":
                continue  # The only survivor

            try:
                if item.is_file():
                    item.unlink()  # Delete project.db, etc.
                elif item.is_dir():
                    shutil.rmtree(item)  # Delete inbox, artifacts, etc.
            except Exception as e:
                print(f"[{proj_name}] ❌ Failed to delete {item.name}: {e}")
                print("Make sure main.py and any DB browsers are closed!")
                return False

        # 2. Rebuild the required structure
        inbox = root / "inbox"
        inbox.mkdir(exist_ok=True)
        (inbox / "_normal").mkdir(exist_ok=True)
        (inbox / "_anomaly").mkdir(exist_ok=True)

        # Create artifacts structure
        subfolders = [
            "raw", "masks", "crop_images", "crop_masks", "embeddings", "heatmaps"
        ]
        for sub in subfolders:
            (root / "artifacts" / sub).mkdir(parents=True, exist_ok=True)

        print(f"[{proj_name}] Cleaned and rebuilt structure.")

    print("--- ✅ SLATE IS CLEAN ---")
    return True


def get_source_images(src_dir):
    """Get all images from a directory."""
    if not src_dir.exists():
        return []
    return sorted(list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png")))


def run_simulation():
    print("--- 🧪 STARTING CHAOS SIMULATION ---")

    # Load all source images for all projects
    sources = {}
    for proj_name in PROJECTS:
        cfg = get_scenario_config(proj_name)
        normal_imgs = get_source_images(cfg["normal_src"])
        anomaly_imgs = get_source_images(cfg["anomaly_src"])
        sources[proj_name] = {
            "normal": normal_imgs,
            "anomaly": anomaly_imgs
        }
        print(f"[{proj_name}] Loaded {len(normal_imgs)} normal, {len(anomaly_imgs)} anomaly images")

    try:
        while True:
            # Pick random project
            proj_name = random.choice(PROJECTS)
            cfg = get_scenario_config(proj_name)

            # Pick random image type (bias towards normal ~80%)
            img_type = "normal" if random.random() > 0.2 else "anomaly"
            src_imgs = sources[proj_name][img_type]

            if not src_imgs:
                print(f"⚠️  [{proj_name}] No {img_type} images available")
                time.sleep(1)
                continue

            # Determine mode
            mode = "SPRINT" if random.random() > 0.2 else "BATCH"

            # Target folder based on image type
            folder_name = "_normal" if img_type == "normal" else "_anomaly"
            target_dir = cfg["dst_root"] / "inbox" / folder_name
            target_dir.mkdir(parents=True, exist_ok=True)

            if mode == "SPRINT":
                img = random.choice(src_imgs)
                shutil.copy(img, target_dir / img.name)
                print(f"⚡ [Sprint] {proj_name}/{folder_name} <- {img.name}")
                time.sleep(random.uniform(0.5, 2.0))
            else:
                batch_size = random.randint(10, 50)
                batch = random.sample(src_imgs, min(len(src_imgs), batch_size))
                print(f"🌊 [Tidal] {proj_name}/{folder_name} <- Uploading {len(batch)} {img_type} files...")
                for img in batch:
                    shutil.copy(img, target_dir / img.name)
                time.sleep(random.uniform(5.0, 10.0))

    except KeyboardInterrupt:
        print("\n--- 🛑 SIMULATION STOPPED ---")


if __name__ == "__main__":
    confirm = input("Type 'yes' to wipe all data (except config.yaml): ").strip().lower()
    if confirm == "yes":
        if not clean_slate():
            sys.exit(1)

    input("Press Enter to start traffic...")
    run_simulation()