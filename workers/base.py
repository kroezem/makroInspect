from abc import ABC, abstractmethod
from typing import Any
from core.project import Project


class BaseWorker(ABC):
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self._is_loaded = False

    def ensure_loaded(self):
        """Lazy loading. Only loads the heavy model if not already in VRAM."""
        if not self._is_loaded:
            print(f"[{self.__class__.__name__}] Loading model to {self.device}...")
            self.load_model()
            self._is_loaded = True

    @abstractmethod
    def load_model(self):
        """
        Implementation must load the specific model (SAM, DINO, etc.)
        and assign it to self.model.
        """
        pass

    @abstractmethod
    def process(self, project: Project) -> int:
        """
        Asks the project for pending work and processes a batch.
        Returns the number of items processed.
        """
        pass
