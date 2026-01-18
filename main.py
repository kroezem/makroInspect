import os
import sys
import time
import warnings
import torch
from pathlib import Path

# --- SETUP & SUPPRESSION ---
def suppress_c_warnings():
    original_stderr = sys.stderr
    try:
        with open(os.devnull, 'w') as f:
            sys.stderr = f
            import torchvision
    except ImportError:
        pass
    finally:
        sys.stderr = original_stderr


suppress_c_warnings()
warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).parent))

from core.project import Project
from core.models import ModelManager
from core.scheduler import Scheduler, TaskType, Lane
from core.telemetry import Telemetry
from workers.ingest_process import IngestProcess
from workers.segment import SegmentWorker
from workers.crop import CropWorker
from workers.evaluate import EvaluateWorker
from workers.bank_analyst import BankAnalyst
from workers.bank_builder import BankBuilder

# --- CONFIG ---
PROJECTS_ROOT = Path("projects")
DEFAULTS_PATH = PROJECTS_ROOT / "defaults.yaml"

PROJECT_CONFIGS = [
    {"name": "pcb4", "desc": "electronic"},
    {"name": "capsules", "desc": "pill"},
    {"name": "cashew", "desc": "food"},
    {"name": "macaroni1", "desc": "food"},
]

def main():
    telemetry = Telemetry()

    t_load = time.perf_counter()
    models = ModelManager(device="cuda")
    models.load_segmenter()
    models.load_evaluator()
    telemetry.emit_startup(time.perf_counter() - t_load)

    projects = [
        Project(PROJECTS_ROOT / c["name"]) if (PROJECTS_ROOT / c["name"]).exists()
        else Project.create(c["name"], c["desc"], PROJECTS_ROOT, DEFAULTS_PATH)
        for c in PROJECT_CONFIGS
    ]

    scheduler = Scheduler(projects)

    workers = {
        TaskType.SEGMENT: SegmentWorker(),
        TaskType.CROP: CropWorker(),
        TaskType.EVALUATE: EvaluateWorker(),
    }
    bank_builder = BankBuilder()

    # Start ingest in separate CPU-only process
    project_roots = [str(p.root) for p in projects]
    ingest_proc = IngestProcess(project_roots, poll_interval=0.3)
    ingest_proc.start()

    analyst = BankAnalyst(
        projects=projects,
        target_bank_size=128,
        poll_interval=5.0,
        on_update=lambda r: telemetry.emit_analyst(r.project_name, r.selected_count, r.total_candidates)
    )
    analyst.start()

    telemetry.emit_running()

    try:
        run_loop(
            projects=projects,
            scheduler=scheduler,
            workers=workers,
            bank_builder=bank_builder,
            analyst=analyst,
            models=models,
            telemetry=telemetry
        )
    except KeyboardInterrupt:
        telemetry.emit_shutdown()
    finally:
        analyst.stop()
        ingest_proc.stop()
        ingest_proc.join(timeout=2)


def run_loop(projects, scheduler, workers, bank_builder, analyst, models, telemetry):
    """
    Clean orchestration loop.

    The main loop is simple:
    1. Ask scheduler for next job (ingest runs in separate process)
    2. Execute job via appropriate worker
    3. Report stats via telemetry
    4. Check if analyst flagged a bank rebuild
    5. When idle: bootstrap/build banks and trigger rescoring
    6. Repeat

    No if/else priority logic here - scheduler owns that.
    """
    while True:
        did_work = False

        # --- ASK SCHEDULER FOR NEXT JOB ---
        job = scheduler.get_next_job()

        if job:
            did_work = True
            worker = workers.get(job.task_type)

            if worker:
                stats = worker.process(projects, job, models)

                if stats.items_processed > 0:
                    telemetry.record(job.task_type, stats)
                    remaining = _get_remaining(scheduler, job)
                    telemetry.emit_status(job, stats, remaining)

            continue

        # --- MAINTENANCE: BANK BOOTSTRAP, BUILD & RESCORE (Only when truly idle) ---
        if scheduler.is_idle():
            for proj in projects:
                # BOOTSTRAP: Force analyst if no bank exists but we have embeddings
                if bank_builder.needs_initial_build(proj):
                    telemetry.emit_system(f"Bootstrap: Analyzing {proj.root.name}")
                    result = analyst.force_analyze(proj)
                    if result and result.changed:
                        telemetry.emit_analyst(result.project_name, result.selected_count, result.total_candidates)

                # BUILD: If analyst has set targets, build the bank
                if bank_builder.needs_rebuild(proj):
                    result = bank_builder.build(proj, models)
                    if result.success:
                        telemetry.emit_bank_update(result.project_name, result.regenerated, result.total_slots)
                        analyst.clear_pending_rebuild()
                        # After bank rebuild, trigger rescoring of all instances
                        rescore_count = scheduler.trigger_rescore(limit=500)
                        if rescore_count > 0:
                            telemetry.emit_system(f"Queued {rescore_count} instances for rescore")
                        did_work = True
                        break

        # --- IDLE ---
        if not did_work:
            time.sleep(0.1)
            if torch.cuda.memory_allocated() > 8 * 1024 ** 3:
                torch.cuda.empty_cache()


def _get_remaining(scheduler, job) -> int:
    """Get remaining count for current lane/task."""
    state = scheduler.get_queue_state(job.lane)
    if job.task_type == TaskType.SEGMENT:
        return state.seg
    elif job.task_type == TaskType.CROP:
        return state.crop
    elif job.task_type == TaskType.EVALUATE:
        return state.eval
    return 0


if __name__ == "__main__":
    main()
