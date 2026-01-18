from abc import ABC, abstractmethod
from typing import List

from core.scheduler import Job
from core.telemetry import WorkerStats


class BaseWorker(ABC):
    """
    Base class for dumb workers.

    Workers are purely functional:
    - Input: List of Project objects + Job config
    - Output: WorkerStats object
    - No internal looping for bursts
    - No print statements (telemetry handles output)
    - Run one atomic batch and return control immediately
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    @abstractmethod
    def process(self, projects: List, job: Job, models=None) -> WorkerStats:
        """
        Execute a single atomic batch.

        Args:
            projects: List of Project objects to process across
            job: Job config from scheduler with limits and priority bounds
            models: Optional ModelManager for GPU workers

        Returns:
            WorkerStats with processing results
        """
        pass
