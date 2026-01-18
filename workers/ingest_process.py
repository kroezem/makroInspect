import os
import sys
import time
import warnings
import multiprocessing
from pathlib import Path


def _suppress_warnings():
    """Suppress torchvision CUDA warnings in subprocess."""
    warnings.filterwarnings("ignore")
    original_stderr = sys.stderr
    try:
        with open(os.devnull, 'w') as f:
            sys.stderr = f
            import torchvision
    except ImportError:
        pass
    finally:
        sys.stderr = original_stderr


class IngestProcess(multiprocessing.Process):
    """
    Separate process for CPU-only ingest work.

    Runs independently from the main GPU loop, communicating via
    the shared SQLite database (WAL mode).
    """

    def __init__(self, project_roots: list, poll_interval: float = 0.5):
        super().__init__(daemon=True, name="IngestProcess")
        self.project_roots = project_roots
        self.poll_interval = poll_interval
        self._stop_event = multiprocessing.Event()

    def stop(self):
        """Signal the process to stop."""
        self._stop_event.set()

    def run(self):
        """Main loop: poll inboxes and ingest new images."""
        _suppress_warnings()

        # Imports after suppression
        from workers.ingest import IngestWorker
        from core.project import Project
        from core.telemetry import Telemetry

        # Create fresh DB connections in subprocess
        projects = [Project(Path(root)) for root in self.project_roots]
        worker = IngestWorker()
        telemetry = Telemetry()

        while not self._stop_event.is_set():
            for proj in projects:
                stats = worker.process(proj)
                if stats.items_processed > 0 or stats.duplicates > 0:
                    telemetry.emit_ingest(stats)

            self._stop_event.wait(self.poll_interval)
