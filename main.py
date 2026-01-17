import os
import warnings
import time
import sys
import torch  # <--- Essential for memory management
from pathlib import Path
from collections import defaultdict

# 1. Disable Hugging Face/Transformer junk
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# 2. Suppress Warnings
# specific filters for xFormers noise
warnings.filterwarnings("ignore", message=".*xFormers.*")
warnings.filterwarnings("ignore", message="Failed to find CUDA")

# Ensure we can import from local modules
sys.path.append(str(Path(__file__).parent))

from sqlalchemy import func
from sqlmodel import select

from core.project import Project
from core.database import Image, Instance
from workers.ingest import IngestWorker
from workers.segment import SegmentWorker
from workers.crop import CropWorker
from workers.evaluate import EvaluateWorker
from workers.bank import BankWorker

# --- Configuration ---
PROJECTS_ROOT = Path("projects")
DEFAULTS_PATH = PROJECTS_ROOT / "defaults.yaml"

PROJECT_CONFIGS = [
    {"name": "pcb4", "desc": "electronic"},
    # {"name": "capsules", "desc": "pill"},
]

performance = defaultdict(dict)


def update_perf(proj_name, stage, time_taken, count):
    if count == 0: return
    avg_time = time_taken / count
    current = performance[proj_name].get(stage)
    if current is None:
        performance[proj_name][stage] = avg_time
    else:
        performance[proj_name][stage] = (current * 0.8) + (avg_time * 0.2)


def format_eta(seconds):
    if seconds is None: return "..."
    if seconds < 60: return f"{int(seconds)}s"
    if seconds < 3600: return f"{int(seconds / 60)}m {int(seconds % 60)}s"
    return f"{int(seconds / 3600)}h {int((seconds % 3600) / 60)}m"


def get_pending_counts(project):
    with project.get_session() as session:
        n_seg = session.exec(select(func.count(Image.id)).where(Image.segmented_at == None)).one()
        n_crop = session.exec(select(func.count(Instance.id)).where(Instance.cropped_at == None)).one()
        n_eval = session.exec(select(func.count(Instance.id)).where(Instance.cropped_at != None).where(
            Instance.evaluated_at == None)).one()
    return n_seg, n_crop, n_eval


def main():
    print(f"--- INITIALIZING SYSTEM ---")

    projects = []
    for cfg in PROJECT_CONFIGS:
        p = PROJECTS_ROOT / cfg["name"]
        if p.exists():
            print(f"Loading: {cfg['name']}")
            proj = Project(p)
            proj.invalidate("embed")
        else:
            print(f"Creating: {cfg['name']}")
            proj = Project.create(cfg['name'], cfg['desc'], PROJECTS_ROOT, DEFAULTS_PATH)
        projects.append(proj)

    # Initialize Workers
    # segment_worker = SegmentWorker(device="cuda")
    evaluate_worker = EvaluateWorker(device="cuda")

    ingest_worker = IngestWorker()
    crop_worker = CropWorker()
    bank_worker = BankWorker()

    bank_last_run = 0
    BANK_INTERVAL = 30.0

    # Track which heavy model was last active
    last_heavy_stage = None

    print(f"--- STARTING PRODUCTION LOOP (Ctrl+C to stop) ---")

    try:
        while True:
            any_work_done = False

            # =========================================
            # STAGE 1: INGEST (CPU)
            # =========================================
            for proj in projects:
                n = ingest_worker.process(proj)
                if n > 0:
                    print(f"[Ingest]  {proj.root.name:<10} | {n} files")
                    any_work_done = True

            # =========================================
            # STAGE 2: SEGMENT (GPU - Model A)
            # =========================================
            # segment_worker.ensure_loaded()
            # for proj in projects:
            #     # If we were just Evaluating, CLEAR MEMORY before loading SAM3 context
            #     if last_heavy_stage == 'evaluate':
            #         torch.cuda.empty_cache()
            #         last_heavy_stage = 'segment'
            #
            #     t0 = time.perf_counter()
            #     n = segment_worker.process(proj)
            #     dt = time.perf_counter() - t0
            #     if n > 0:
            #         last_heavy_stage = 'segment'
            #         update_perf(proj.root.name, "segment", dt, n)
            #         any_work_done = True
            #         print(f"[Segment] {proj.root.name:<10} | {n:>2} imgs  | {dt:.3f}s | {(dt / n) * 1000:.0f}ms/img")

            # =========================================
            # STAGE 3: CROP (CPU)
            # =========================================
            for proj in projects:
                t0 = time.perf_counter()
                n = crop_worker.process(proj)
                dt = time.perf_counter() - t0
                if n > 0:
                    update_perf(proj.root.name, "crop", dt, n)
                    any_work_done = True
                    print(f"[Crop]    {proj.root.name:<10} | {n:>2} crops | {dt:.3f}s | {(dt / n) * 1000:.0f}ms/crop")

            # =========================================
            # STAGE 4: EVALUATE (GPU - Model B)
            # =========================================
            evaluate_worker.ensure_loaded()
            for proj in projects:
                # If we were just Segmenting, CLEAR MEMORY before DINO Scoring
                if last_heavy_stage == 'segment':
                    torch.cuda.empty_cache()
                    last_heavy_stage = 'evaluate'

                t0 = time.perf_counter()
                n = evaluate_worker.process(proj)
                dt = time.perf_counter() - t0
                if n > 0:
                    last_heavy_stage = 'evaluate'
                    update_perf(proj.root.name, "evaluate", dt, n)
                    any_work_done = True
                    fps = n / dt if dt > 0 else 0
                    print(
                        f"[Evaluate] {proj.root.name:<10} | {n:>2} insts | {dt:.3f}s | {(dt / n) * 1000:.0f}ms/inst ({fps:.1f} fps)")

            # =========================================
            # STAGE 5: BANKER (Throttled)
            # =========================================
            if (time.time() - bank_last_run) > BANK_INTERVAL:
                n_total = 0
                for proj in projects:
                    n_total += bank_worker.process(proj)
                if n_total > 0:
                    print(f"[Bank]    Maintenance cycle complete.")
                    any_work_done = True
                bank_last_run = time.time()

            # =========================================
            # DASHBOARD
            # =========================================
            if any_work_done:
                print("-" * 75)
                for proj in projects:
                    name = proj.root.name
                    n_seg, n_crop, n_eval = get_pending_counts(proj)
                    if n_seg + n_crop + n_eval > 0:
                        r_seg = performance[name].get("segment")
                        r_crop = performance[name].get("crop")
                        r_eval = performance[name].get("evaluate")

                        eta_seg = format_eta(r_seg * n_seg) if r_seg else "..."
                        eta_crop = format_eta(r_crop * n_crop) if r_crop else "..."
                        eta_eval = format_eta(r_eval * n_eval) if r_eval else "..."

                        print(
                            f"  Status: {name:<10} | Seg: {n_seg:>4} ({eta_seg:>5}) | Crop: {n_crop:>4} ({eta_crop:>5}) | Eval: {n_eval:>4} ({eta_eval:>5})")
                print("-" * 75 + "\n")
            else:
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping...")


if __name__ == "__main__":
    main()
