"""
Initialize projects from various dataset sources.

Usage:
    uv run prepare.py visa capsules           # Single VisA category
    uv run prepare.py visa --all              # All VisA categories
    uv run prepare.py mvtec_ad bottle         # Single MVTec AD category
    uv run prepare.py mvtec_ad --all          # All MVTec AD categories
    uv run prepare.py mvtec_loco pushpins     # Single MVTec LOCO category
    uv run prepare.py mvtec_ad2 can           # Single MVTec AD2 category

Creates:
    projects/{project}/config.yaml        # Permanent config
    projects/{project}/runs.csv           # Empty runs file
    artifacts/{project}/data/             # Dataset files (regenerable)
"""

import argparse
import csv
import shutil
import tarfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
from tqdm import tqdm

from core.paths import REPO_ROOT, PROJECTS_DIR, ARTIFACTS_DIR, DATASETS_DIR, CONFIG_DIR, ProjectPaths
from core.registry import create_empty_runs_csv

VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
DEFAULTS_CONFIG = CONFIG_DIR / "defaults.yaml"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

# MVTec AD - individual category downloads
MVTEC_AD_URLS = {
    "bottle": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937370-1629958698/bottle.tar.xz",
    "cable": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937413-1629958794/cable.tar.xz",
    "capsule": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937454-1629958872/capsule.tar.xz",
    "carpet": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937484-1629959013/carpet.tar.xz",
    "grid": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937487-1629959044/grid.tar.xz",
    "hazelnut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937545-1629959162/hazelnut.tar.xz",
    "leather": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937607-1629959262/leather.tar.xz",
    "metal_nut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937637-1629959294/metal_nut.tar.xz",
    "pill": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938129-1629960351/pill.tar.xz",
    "screw": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938130-1629960389/screw.tar.xz",
    "tile": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938133-1629960456/tile.tar.xz",
    "toothbrush": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938134-1629960477/toothbrush.tar.xz",
    "transistor": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938166-1629960554/transistor.tar.xz",
    "wood": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938383-1629960649/wood.tar.xz",
    "zipper": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938385-1629960680/zipper.tar.xz",
}

# MVTec LOCO
MVTEC_LOCO_URLS = {
    "breakfast_box": "https://www.mydrive.ch/shares/48238/d94999f7d56b82c26cc22e36a8cfe02b/download/430647124-1646843193/breakfast_box.tar.xz",
    "juice_bottle": "https://www.mydrive.ch/shares/48239/3ad28d48636eada48f0caded666a804a/download/430647100-1646843074/juice_bottle.tar.xz",
    "pushpins": "https://www.mydrive.ch/shares/48240/e1942c84f417afccee149ef5121846c1/download/430647097-1646843009/pushpins.tar.xz",
    "screw_bag": "https://www.mydrive.ch/shares/48241/be5dc3a69c92932644aaac945451d4e7/download/430647115-1646843129/screw_bag.tar.xz",
    "splicing_connectors": "https://www.mydrive.ch/shares/48242/9202635b7b15beba2ceea1674c7deebc/download/430647132-1646843288/splicing_connectors.tar.xz",
}

# MVTec AD2
MVTEC_AD2_URLS = {
    "can": "https://www.mydrive.ch/shares/121501/26456e2f3ef813930866f8f9b072593a/download/466651130-1743159807/can.tar.gz",
    "fabric": "https://www.mydrive.ch/shares/121502/812590f745083da1f5edb338a6c321c4/download/466651519-1743160379/fabric.tar.gz",
    "fruit_jelly": "https://www.mydrive.ch/shares/121503/951a46ce30a3af3787ce9671cfa8613a/download/466651800-1743164023/fruit_jelly.tar.gz",
    "rice": "https://www.mydrive.ch/shares/121504/0014676292c3c44931712a54fb3bdbe8/download/466653907-1743164943/rice.tar.gz",
    "sheet_metal": "https://www.mydrive.ch/shares/121505/2d8fcdc8e988456bdd18696746eda0a0/download/466654829-1743166795/sheet_metal.tar.gz",
    "vial": "https://www.mydrive.ch/shares/121506/739dc6459c939fe464c0d26acc6c2d55/download/466654885-1743167505/vial.tar.gz",
    "wallplugs": "https://www.mydrive.ch/shares/121507/66fe6e114b498e03be8d48c711794be7/download/466655287-1743168151/wallplugs.tar.gz",
    "walnuts": "https://www.mydrive.ch/shares/121508/9fcf67e49f0dc61a9608f57ba0482356/download/466656233-1743168988/walnuts.tar.gz",
}


# =============================================================================
# VisA Dataset
# =============================================================================

def download_visa() -> Path:
    """Download and extract VisA dataset if not present."""
    visa_dir = DATASETS_DIR / "visa"
    tar_path = visa_dir / "VisA_20220922.tar"

    if visa_dir.exists() and (visa_dir / "split_csv").exists():
        print(f"VisA dataset already exists at {visa_dir}")
        return visa_dir

    print("=" * 60)
    print("Downloading VisA dataset")
    print("=" * 60)

    if not tar_path.exists():
        visa_dir.mkdir(parents=True, exist_ok=True)

        req = Request(VISA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req) as r, open(tar_path, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))

    print()
    print("Extracting VisA dataset...")
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(visa_dir, filter="data")
    print("✓ Extraction complete")

    return visa_dir


def get_visa_categories(visa_root: Path) -> list[str]:
    """Get all available categories from VisA 1cls.csv."""
    split_csv = visa_root / "split_csv" / "1cls.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Missing {split_csv}")

    categories = set()
    with open(split_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            categories.add(row["object"])
    return sorted(categories)


def init_visa_project(category: str, visa_root: Path):
    """Initialize a single VisA project."""
    print()
    print("=" * 60)
    print(f"Initializing project: {category}")
    print("=" * 60)

    paths = ProjectPaths(category)

    # === Create permanent project structure ===

    print()
    print("Setting up project folder...")

    paths.project_root.mkdir(parents=True, exist_ok=True)

    # Copy default config
    if DEFAULTS_CONFIG.exists():
        if paths.config.exists():
            print(f"  ✓ config.yaml already exists, not overwriting")
        else:
            shutil.copy2(DEFAULTS_CONFIG, paths.config)
            print(f"  ✓ Copied config.yaml")
    else:
        print(f"  ⚠ No defaults.yaml found at {DEFAULTS_CONFIG}")

    # Create empty runs.csv
    create_empty_runs_csv(paths)
    print(f"  ✓ Created runs.csv")

    # === Create artifacts/data structure ===

    print()
    print("Setting up data folder...")

    data_root = paths.data_dir

    for folder in [
        data_root / "train" / "good",
        data_root / "test" / "good",
        data_root / "ground_truth",
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    # === Read VisA metadata ===

    print()
    print("Reading VisA metadata...")

    split_csv = visa_root / "split_csv" / "1cls.csv"
    split_info = {}
    with open(split_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["object"] == category:
                split_info[row["image"]] = {
                    "split": row["split"],
                    "label": row["label"],
                    "mask": row["mask"]
                }

    # Read defect types from image_anno.csv
    anno_csv = visa_root / category / "image_anno.csv"
    defect_labels = {}
    if anno_csv.exists():
        with open(anno_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                defect_labels[row["image"]] = row["label"]

    if not split_info:
        print(f"  ⚠ No images found for category '{category}'")
        return

    print(f"  Found {len(split_info)} images")

    # === Copy images ===

    print()
    print("Copying images...")

    stats = {"train": 0, "test_normal": 0, "test_anomaly": 0, "masks": 0}
    defect_types = set()

    for img_rel, info in tqdm(split_info.items(), desc="Processing"):
        split = info["split"]
        label_1cls = info["label"]
        mask_rel = info["mask"]

        img_src = visa_root / img_rel
        if not img_src.exists():
            continue

        if label_1cls == "normal":
            if split == "train":
                dst_dir = data_root / "train" / "good"
                stats["train"] += 1
            else:
                dst_dir = data_root / "test" / "good"
                stats["test_normal"] += 1
            shutil.copy2(img_src, dst_dir / img_src.name)
        else:
            defect_type = defect_labels.get(img_rel, "anomaly")
            defect_types.add(defect_type)

            test_dir = data_root / "test" / defect_type
            gt_dir = data_root / "ground_truth" / defect_type
            test_dir.mkdir(parents=True, exist_ok=True)
            gt_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(img_src, test_dir / img_src.name)
            stats["test_anomaly"] += 1

            if mask_rel:
                mask_src = visa_root / mask_rel
                if mask_src.exists():
                    mask_name = img_src.stem + ".png"
                    _convert_mask_to_255(mask_src, gt_dir / mask_name)
                    stats["masks"] += 1

    # === Summary ===

    print()
    print(f"✓ Project '{category}' initialized")
    print()
    print("  Project folder (permanent):")
    print(f"    {paths.project_root}")
    print()
    print("  Artifacts folder (regenerable):")
    print(f"    {paths.artifacts_root}")
    print()
    print("  Dataset stats:")
    print(f"    Train (normal):  {stats['train']}")
    print(f"    Test (normal):   {stats['test_normal']}")
    print(f"    Test (anomaly):  {stats['test_anomaly']}")
    print(f"    GT masks:        {stats['masks']}")
    if defect_types:
        print(f"    Defect types:    {', '.join(sorted(defect_types))}")


def _convert_mask_to_255(src_path: Path, dst_path: Path):
    """Convert 0-1 binary mask to 0-255."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src_path).convert("L")
    arr = np.array(img)
    arr = (arr > 0).astype(np.uint8) * 255
    Image.fromarray(arr, mode="L").save(dst_path)


def cmd_visa(args):
    """Handle visa subcommand."""
    visa_root = download_visa()

    if args.all:
        categories = get_visa_categories(visa_root)
        print()
        print(f"Found {len(categories)} categories: {', '.join(categories)}")
    elif args.category:
        categories = [args.category]
    else:
        print("Error: specify a category or use --all")
        return

    for category in categories:
        try:
            init_visa_project(category, visa_root)
        except Exception as e:
            print(f"❌ Error processing {category}: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)


# =============================================================================
# MVTec AD Dataset
# =============================================================================

def download_mvtec_category(category: str, url_dict: dict, dataset_name: str) -> Path:
    """Download and extract a single MVTec category."""
    dataset_dir = DATASETS_DIR / dataset_name
    category_dir = dataset_dir / category

    # Check if already extracted
    if category_dir.exists() and (category_dir / "train").exists():
        print(f"Category '{category}' already exists at {category_dir}")
        return category_dir

    if category not in url_dict:
        raise ValueError(f"Unknown category '{category}'. Available: {', '.join(sorted(url_dict.keys()))}")

    url = url_dict[category]

    # Determine file extension from URL
    if url.endswith(".tar.gz"):
        tar_name = f"{category}.tar.gz"
    else:
        tar_name = f"{category}.tar.xz"

    tar_path = dataset_dir / tar_name

    print()
    print("=" * 60)
    print(f"Downloading {dataset_name}/{category}")
    print("=" * 60)

    dataset_dir.mkdir(parents=True, exist_ok=True)

    if not tar_path.exists():
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req) as r, open(tar_path, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))

    print()
    print(f"Extracting {category}...")
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(dataset_dir, filter="data")
    print("✓ Extraction complete")

    return category_dir


def init_mvtec_project(category: str, category_dir: Path, project_name: str | None = None):
    """Initialize a single MVTec-style project (works for AD, LOCO, AD2)."""

    if project_name is None:
        project_name = category

    print()
    print("=" * 60)
    print(f"Initializing project: {project_name}")
    print("=" * 60)

    train_good = category_dir / "train" / "good"
    test_root = category_dir / "test"
    gt_root = category_dir / "ground_truth"

    if not train_good.exists() or not test_root.exists():
        raise FileNotFoundError(f"Category does not look like MVTec: {category_dir}")

    paths = ProjectPaths(project_name)

    # === Create permanent project structure ===

    print()
    print("Setting up project folder...")

    paths.project_root.mkdir(parents=True, exist_ok=True)

    # Copy default config (do not overwrite)
    if DEFAULTS_CONFIG.exists():
        if paths.config.exists():
            print(f"  ✓ config.yaml already exists, not overwriting")
        else:
            shutil.copy2(DEFAULTS_CONFIG, paths.config)
            print(f"  ✓ Copied config.yaml")
    else:
        print(f"  ⚠ No defaults.yaml found at {DEFAULTS_CONFIG}")

    # Create empty runs.csv
    create_empty_runs_csv(paths)
    print(f"  ✓ Created runs.csv")

    # === Create artifacts/data structure ===

    print()
    print("Setting up data folder...")

    data_root = paths.data_dir
    for folder in [
        data_root / "train" / "good",
        data_root / "test" / "good",
        data_root / "ground_truth",
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    # === Copy images ===

    print()
    print("Copying images...")

    stats = {"train": 0, "test_normal": 0, "test_anomaly": 0, "masks": 0}
    defect_types = set()

    # Train good
    for img_path in sorted(train_good.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        shutil.copy2(img_path, data_root / "train" / "good" / img_path.name)
        stats["train"] += 1

    # Test good
    test_good = test_root / "good"
    if test_good.exists():
        for img_path in sorted(test_good.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            shutil.copy2(img_path, data_root / "test" / "good" / img_path.name)
            stats["test_normal"] += 1

    # Test anomalies + GT
    for defect_dir in sorted(test_root.iterdir()):
        if not defect_dir.is_dir():
            continue
        defect = defect_dir.name
        if defect == "good":
            continue

        defect_types.add(defect)

        out_test = data_root / "test" / defect
        out_gt = data_root / "ground_truth" / defect
        out_test.mkdir(parents=True, exist_ok=True)
        out_gt.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(defect_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue

            shutil.copy2(img_path, out_test / img_path.name)
            stats["test_anomaly"] += 1

            # Ground truth masks
            gt_defect_dir = gt_root / defect
            if gt_defect_dir.exists():
                # Try common naming patterns
                for mask_name in [f"{img_path.stem}.png", f"{img_path.stem}_mask.png"]:
                    mask_path = gt_defect_dir / mask_name
                    if mask_path.exists():
                        _convert_mask_to_255(mask_path, out_gt / f"{img_path.stem}.png")
                        stats["masks"] += 1
                        break

    # === Summary ===

    print()
    print(f"✓ Project '{project_name}' initialized")
    print()
    print("  Project folder (permanent):")
    print(f"    {paths.project_root}")
    print()
    print("  Artifacts folder (regenerable):")
    print(f"    {paths.artifacts_root}")
    print()
    print("  Dataset stats:")
    print(f"    Train (normal):  {stats['train']}")
    print(f"    Test (normal):   {stats['test_normal']}")
    print(f"    Test (anomaly):  {stats['test_anomaly']}")
    print(f"    GT masks:        {stats['masks']}")
    if defect_types:
        print(f"    Defect types:    {', '.join(sorted(defect_types))}")


def cmd_mvtec_ad(args):
    """Handle mvtec_ad subcommand."""
    if args.all:
        categories = sorted(MVTEC_AD_URLS.keys())
        print(f"Processing {len(categories)} categories: {', '.join(categories)}")
    elif args.category:
        categories = [args.category]
    else:
        print("Error: specify a category or use --all")
        print(f"Available categories: {', '.join(sorted(MVTEC_AD_URLS.keys()))}")
        return

    for category in categories:
        try:
            category_dir = download_mvtec_category(category, MVTEC_AD_URLS, "mvtec_ad")
            init_mvtec_project(category, category_dir)
        except Exception as e:
            print(f"❌ Error processing {category}: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)


def cmd_mvtec_loco(args):
    """Handle mvtec_loco subcommand."""
    if args.all:
        categories = sorted(MVTEC_LOCO_URLS.keys())
        print(f"Processing {len(categories)} categories: {', '.join(categories)}")
    elif args.category:
        categories = [args.category]
    else:
        print("Error: specify a category or use --all")
        print(f"Available categories: {', '.join(sorted(MVTEC_LOCO_URLS.keys()))}")
        return

    for category in categories:
        try:
            category_dir = download_mvtec_category(category, MVTEC_LOCO_URLS, "mvtec_loco")
            init_mvtec_project(category, category_dir)  # No prefix
        except Exception as e:
            print(f"❌ Error processing {category}: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)


def cmd_mvtec_ad2(args):
    """Handle mvtec_ad2 subcommand."""
    if args.all:
        categories = sorted(MVTEC_AD2_URLS.keys())
        print(f"Processing {len(categories)} categories: {', '.join(categories)}")
    elif args.category:
        categories = [args.category]
    else:
        print("Error: specify a category or use --all")
        print(f"Available categories: {', '.join(sorted(MVTEC_AD2_URLS.keys()))}")
        return

    for category in categories:
        try:
            category_dir = download_mvtec_category(category, MVTEC_AD2_URLS, "mvtec_ad2")
            init_mvtec_project(category, category_dir)  # No prefix
        except Exception as e:
            print(f"❌ Error processing {category}: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)


# =============================================================================
# Local Dataset (placeholder)
# =============================================================================

def cmd_local(args):
    """Handle local subcommand."""
    raise NotImplementedError("Local dataset support coming soon")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Initialize makroInspect projects from datasets"
    )
    subparsers = parser.add_subparsers(dest="source", required=True)

    # visa subcommand
    visa_parser = subparsers.add_parser("visa", help="Initialize from VisA dataset")
    visa_parser.add_argument("category", nargs="?", help="Category (e.g., capsules)")
    visa_parser.add_argument("--all", action="store_true", help="Initialize all categories")
    visa_parser.set_defaults(func=cmd_visa)

    # mvtec_ad subcommand
    mvtec_ad_parser = subparsers.add_parser("mvtec_ad", help="Initialize from MVTec AD dataset")
    mvtec_ad_parser.add_argument("category", nargs="?", help="Category (e.g., bottle)")
    mvtec_ad_parser.add_argument("--all", action="store_true", help="Initialize all categories")
    mvtec_ad_parser.set_defaults(func=cmd_mvtec_ad)

    # mvtec_loco subcommand
    mvtec_loco_parser = subparsers.add_parser("mvtec_loco", help="Initialize from MVTec LOCO dataset")
    mvtec_loco_parser.add_argument("category", nargs="?", help="Category (e.g., pushpins)")
    mvtec_loco_parser.add_argument("--all", action="store_true", help="Initialize all categories")
    mvtec_loco_parser.set_defaults(func=cmd_mvtec_loco)

    # mvtec_ad2 subcommand
    mvtec_ad2_parser = subparsers.add_parser("mvtec_ad2", help="Initialize from MVTec AD2 dataset")
    mvtec_ad2_parser.add_argument("category", nargs="?", help="Category (e.g., can)")
    mvtec_ad2_parser.add_argument("--all", action="store_true", help="Initialize all categories")
    mvtec_ad2_parser.set_defaults(func=cmd_mvtec_ad2)

    # local subcommand
    local_parser = subparsers.add_parser("local", help="Initialize from local folder")
    local_parser.add_argument("project", help="Project name")
    local_parser.add_argument("--source", type=Path, required=True, help="Source data path")
    local_parser.set_defaults(func=cmd_local)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
