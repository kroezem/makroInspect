import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional
from contextlib import contextmanager

from core.scheduler import Job, Lane, TaskType


@dataclass
class WorkerStats:
    """Standardized output from workers - no printing inside workers."""
    items_processed: int = 0
    errors: int = 0
    duration_ms: float = 0.0
    project_name: str = ""
    extra: Dict = field(default_factory=dict)

    @property
    def ms_per_item(self) -> float:
        if self.items_processed == 0:
            return 0.0
        return self.duration_ms / self.items_processed


@dataclass
class IngestStats(WorkerStats):
    duplicates: int = 0
    is_sprint: bool = False


class Telemetry:
    """
    The Voice: Centralized logging and metrics for the pipeline.

    Responsibilities:
    - Manage timing/stopwatch for consistent metrics
    - Maintain rolling averages for ETA calculations
    - Produce standardized, clean status output
    """

    STAGE_ICONS = {
        TaskType.INGEST: "",
        TaskType.SEGMENT: "",
        TaskType.CROP: "",
        TaskType.EVALUATE: "",
        TaskType.BANK_BUILD: "",
    }

    LANE_LABELS = {
        Lane.SPRINT: "SPRINT",
        Lane.BULK: "BULK  ",
        Lane.MAINTENANCE: "BACK  ",
    }

    def __init__(self, window_size: int = 20):
        self._metrics: Dict[TaskType, deque] = {
            TaskType.SEGMENT: deque(maxlen=window_size),
            TaskType.CROP: deque(maxlen=window_size),
            TaskType.EVALUATE: deque(maxlen=window_size),
        }
        self._start_time: Optional[float] = None

    @contextmanager
    def stopwatch(self):
        """Context manager for timing operations."""
        start = time.perf_counter()
        yield
        self._last_duration = (time.perf_counter() - start) * 1000

    def get_last_duration_ms(self) -> float:
        return getattr(self, '_last_duration', 0.0)

    def record(self, task_type: TaskType, stats: WorkerStats):
        """Record metrics from a completed batch."""
        if stats.items_processed <= 0:
            return
        if task_type in self._metrics:
            self._metrics[task_type].append(stats.ms_per_item)

    def get_eta_seconds(self, task_type: TaskType, remaining: int) -> Optional[float]:
        """Calculate ETA based on rolling average."""
        if task_type not in self._metrics or remaining <= 0:
            return None
        history = self._metrics[task_type]
        if not history:
            return None
        avg_ms = sum(history) / len(history)
        return (avg_ms * remaining) / 1000

    def format_eta(self, seconds: Optional[float]) -> str:
        if seconds is None:
            return ""
        if seconds < 60:
            return f"(ETA: {int(seconds)}s)"
        elif seconds < 3600:
            return f"(ETA: {int(seconds // 60)}m {int(seconds % 60)}s)"
        else:
            return f"(ETA: {seconds / 3600:.1f}h)"

    def emit_status(self, job: Job, stats: WorkerStats, remaining: int = 0):
        """
        Emit a standardized status line.
        Format: [SEG ] SPRINT | project    | 4 items | 120ms (30ms/obj) | Rem: 0 (ETA: 5s)
        """
        if stats.items_processed == 0:
            return

        icon = self.STAGE_ICONS.get(job.task_type, "")
        lane_label = self.LANE_LABELS.get(job.lane, "???")
        stage_label = job.task_type.name[:4].ljust(4)
        proj_label = (stats.project_name[:10] if stats.project_name else "").ljust(10)

        eta_sec = self.get_eta_seconds(job.task_type, remaining)
        eta_str = self.format_eta(eta_sec)

        print(
            f"[{stage_label}] {icon} {lane_label} | {proj_label} | "
            f"{stats.items_processed:>3} items | "
            f"{stats.duration_ms:>5.0f}ms ({stats.ms_per_item:>3.0f}ms/obj) | "
            f"Rem: {remaining:<5} {eta_str}"
        )

    def emit_ingest(self, stats: IngestStats):
        """Emit ingest-specific status."""
        if stats.items_processed == 0 and stats.duplicates == 0:
            return

        lane_emoji = "" if stats.is_sprint else ""
        lane_text = "SPRINT" if stats.is_sprint else "BATCH"

        if stats.duplicates > 0:
            print(f"[INGE]   {stats.project_name:<10} | Ignored {stats.duplicates} duplicates")

        if stats.items_processed > 0:
            print(f"[INGE]   {stats.project_name:<10} | {stats.items_processed} new -> {lane_emoji} {lane_text}")

    def emit_bank_update(self, project_name: str, regenerated: int, total: int):
        """Emit bank build status."""
        print(f"[BANK]  {project_name:<10} | Rebuilt {regenerated}/{total} entries")

    def emit_analyst(self, project_name: str, selected: int, total_candidates: int):
        """Emit analyst (K-center) status."""
        print(f"[ANLY]  {project_name:<10} | Selected {selected} from {total_candidates} candidates")

    def emit_system(self, message: str):
        """Emit system-level messages."""
        print(f"[System] {message}")

    def emit_startup(self, load_time: float):
        """Emit startup banner."""
        print("--- ENGINE STARTING ---")
        self.emit_system(f"Models loaded in {load_time:.2f}s")

    def emit_running(self):
        """Emit running state banner."""
        print("--- ENGINE RUNNING (Atomic Preemption) ---")

    def emit_shutdown(self):
        """Emit shutdown message."""
        print("\nStopping Engine...")
