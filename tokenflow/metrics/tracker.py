from collections import deque
from threading import Lock
from uuid import uuid4

from tokenflow.models import OptimizationResult


class MetricsTracker:
    """Bounded per-process metrics. Deliberately stores no prompts or responses."""

    def __init__(self, capacity: int = 1000):
        self._events = deque(maxlen=capacity)
        self._lock = Lock()

    def record(self, result: OptimizationResult, **numeric_metrics) -> str:
        request_id = str(uuid4())
        event = {"request_id": request_id, **result.metrics.model_dump(), **numeric_metrics}
        with self._lock:
            self._events.append(event)
        return request_id

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [dict(event) for event in self._events]
