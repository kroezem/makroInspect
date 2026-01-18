from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional
from sqlmodel import select, func

from core.database import Image, Instance


class Lane(Enum):
    SPRINT = auto()
    BULK = auto()
    MAINTENANCE = auto()


class TaskType(Enum):
    INGEST = auto()
    SEGMENT = auto()
    CROP = auto()
    EVALUATE = auto()
    BANK_BUILD = auto()


@dataclass
class Job:
    task_type: TaskType
    lane: Lane
    batch_limit: int
    min_priority: int
    max_priority: int

    @property
    def lane_name(self) -> str:
        return {
            Lane.SPRINT: "sprint",
            Lane.BULK: "bulk",
            Lane.MAINTENANCE: "back"
        }[self.lane]


@dataclass
class QueueState:
    seg: int = 0
    crop: int = 0
    eval: int = 0

    @property
    def total(self) -> int:
        return self.seg + self.crop + self.eval

    def is_empty(self) -> bool:
        return self.total == 0


class Scheduler:
    """
    The Brain: Centralized decision-making for what runs next.

    Priority System:
    - Sprint (>=10): Latency-critical, greedy execution
    - Bulk (5-9): Throughput-critical, tidal execution (seg->crop->eval)
    - Maintenance (<5): Background tasks, only when idle
    """

    LANE_BOUNDS = {
        Lane.SPRINT: (10, 100),
        Lane.BULK: (5, 9),
        Lane.MAINTENANCE: (0, 4)
    }

    DEFAULT_LIMITS = {
        TaskType.SEGMENT: 8,
        TaskType.CROP: 50,
        TaskType.EVALUATE: 32,
    }

    def __init__(self, projects: List):
        self.projects = projects
        self._queue_cache: dict[Lane, QueueState] = {}
        self._sprint_phase: TaskType = TaskType.SEGMENT
        self._bulk_phase: TaskType = TaskType.SEGMENT

    def get_next_job(self) -> Optional[Job]:
        """
        Returns the next Job to execute, or None if system is idle.
        Implements preemption: Sprint always checked first.
        """
        self._refresh_queues()

        if job := self._check_sprint():
            return job

        if job := self._check_bulk():
            return job

        if job := self._check_maintenance():
            return job

        return None

    def is_idle(self) -> bool:
        """Returns True if no work pending across all lanes."""
        self._refresh_queues()
        for state in self._queue_cache.values():
            if not state.is_empty():
                return False
        return True

    def has_sprint_work(self) -> bool:
        """Returns True if sprint lane has pending work."""
        self._refresh_queues()
        state = self._queue_cache.get(Lane.SPRINT, QueueState())
        return not state.is_empty()

    def get_queue_state(self, lane: Lane) -> QueueState:
        """Get cached queue state for a lane."""
        return self._queue_cache.get(lane, QueueState())

    def _refresh_queues(self):
        """Efficiently query DB for pending counts per lane."""
        self._queue_cache = {
            Lane.SPRINT: self._count_lane(Lane.SPRINT),
            Lane.BULK: self._count_lane(Lane.BULK),
            Lane.MAINTENANCE: self._count_lane(Lane.MAINTENANCE),
        }

    def _count_lane(self, lane: Lane) -> QueueState:
        min_p, max_p = self.LANE_BOUNDS[lane]
        state = QueueState()

        for proj in self.projects:
            with proj.get_session() as session:
                state.seg += session.execute(
                    select(func.count(Image.id))
                    .where(Image.segmented_at == None)
                    .where(Image.priority >= min_p)
                    .where(Image.priority <= max_p)
                ).scalar() or 0

                state.crop += session.execute(
                    select(func.count(Instance.id))
                    .where(Instance.cropped_at == None)
                    .where(Instance.priority >= min_p)
                    .where(Instance.priority <= max_p)
                ).scalar() or 0

                state.eval += session.execute(
                    select(func.count(Instance.id))
                    .where(Instance.evaluated_at == None)
                    .where(Instance.cropped_at != None)
                    .where(Instance.priority >= min_p)
                    .where(Instance.priority <= max_p)
                ).scalar() or 0

        return state

    def _check_sprint(self) -> Optional[Job]:
        """
        Sprint lane: Tidal execution - complete each phase before moving on.
        Phase order: SEGMENT -> CROP -> EVALUATE -> repeat
        New arrivals wait for current cycle to complete.
        """
        state = self._queue_cache.get(Lane.SPRINT, QueueState())
        bounds = self.LANE_BOUNDS[Lane.SPRINT]

        if state.is_empty():
            self._sprint_phase = TaskType.SEGMENT
            return None

        if self._sprint_phase == TaskType.SEGMENT:
            if state.seg > 0:
                return self._make_job(TaskType.SEGMENT, Lane.SPRINT, bounds)
            self._sprint_phase = TaskType.CROP

        if self._sprint_phase == TaskType.CROP:
            if state.crop > 0:
                return self._make_job(TaskType.CROP, Lane.SPRINT, bounds)
            self._sprint_phase = TaskType.EVALUATE

        if self._sprint_phase == TaskType.EVALUATE:
            if state.eval > 0:
                return self._make_job(TaskType.EVALUATE, Lane.SPRINT, bounds)
            self._sprint_phase = TaskType.SEGMENT

        return None

    def _check_bulk(self) -> Optional[Job]:
        """
        Bulk lane: Tidal execution - finish all of one phase before moving on.
        Phase order: SEGMENT -> CROP -> EVALUATE -> repeat
        """
        state = self._queue_cache.get(Lane.BULK, QueueState())
        bounds = self.LANE_BOUNDS[Lane.BULK]

        if state.is_empty():
            self._bulk_phase = TaskType.SEGMENT
            return None

        if self._bulk_phase == TaskType.SEGMENT:
            if state.seg > 0:
                return self._make_job(TaskType.SEGMENT, Lane.BULK, bounds)
            self._bulk_phase = TaskType.CROP

        if self._bulk_phase == TaskType.CROP:
            if state.crop > 0:
                return self._make_job(TaskType.CROP, Lane.BULK, bounds)
            self._bulk_phase = TaskType.EVALUATE

        if self._bulk_phase == TaskType.EVALUATE:
            if state.eval > 0:
                return self._make_job(TaskType.EVALUATE, Lane.BULK, bounds)
            self._bulk_phase = TaskType.SEGMENT

        return None

    def _check_maintenance(self) -> Optional[Job]:
        """Maintenance lane: Background work when sprint/bulk are empty."""
        state = self._queue_cache.get(Lane.MAINTENANCE, QueueState())
        bounds = self.LANE_BOUNDS[Lane.MAINTENANCE]

        if state.seg > 0:
            return self._make_job(TaskType.SEGMENT, Lane.MAINTENANCE, bounds)
        if state.crop > 0:
            return self._make_job(TaskType.CROP, Lane.MAINTENANCE, bounds)
        if state.eval > 0:
            return self._make_job(TaskType.EVALUATE, Lane.MAINTENANCE, bounds)

        # Check for rescore candidates (instances that need re-evaluation due to bank change)
        if self._has_rescore_candidates():
            return self._make_job(TaskType.EVALUATE, Lane.MAINTENANCE, bounds)

        return None

    def _has_rescore_candidates(self) -> bool:
        """Check if any project has instances needing rescore."""
        for proj in self.projects:
            if proj.get_pending_rescore_count() > 0:
                return True
        return False

    def trigger_rescore(self, limit: int = 100) -> int:
        """
        Trigger rescoring of instances that need it.
        Called after bank is rebuilt.
        Returns number of instances queued for rescore.
        """
        total = 0
        for proj in self.projects:
            count = proj.get_maintenance_candidates(limit=limit)
            total += count
        return total

    def _make_job(self, task_type: TaskType, lane: Lane, bounds: tuple) -> Job:
        return Job(
            task_type=task_type,
            lane=lane,
            batch_limit=self.DEFAULT_LIMITS.get(task_type, 32),
            min_priority=bounds[0],
            max_priority=bounds[1]
        )

