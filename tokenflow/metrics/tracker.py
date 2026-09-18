from collections import Counter, deque
from threading import Lock
from typing import Literal
from uuid import uuid4

from tokenflow.llm.providers import ProviderResult
from tokenflow.models import OptimizationResult

Operation = Literal["optimize", "generate"]
Outcome = Literal[
    "success", "budget_exceeded", "optimizer_error", "provider_error", "provider_unavailable"
]


class MetricsTracker:
    """Bounded event history plus process-lifetime outcome counts; no request content."""

    def __init__(self, capacity: int = 1000):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Metrics capacity must be a positive integer")
        self._events = deque(maxlen=capacity)
        self._capacity = capacity
        self._counts = Counter()
        self._total = 0
        self._provider_attempts = 0
        self._lock = Lock()

    def record(
        self,
        result: OptimizationResult | None = None,
        *,
        operation: Operation,
        outcome: Outcome,
        total_latency_ms: float,
        provider_attempted: bool = False,
        response: ProviderResult | None = None,
    ) -> str:
        request_id = str(uuid4())
        event = {
            "request_id": request_id,
            "operation": operation,
            "outcome": outcome,
            "total_latency_ms": total_latency_ms,
            "provider_attempted": provider_attempted,
            "billing_usage": (
                "reported" if response else "unknown" if provider_attempted else "not_called"
            ),
            "provider_input_tokens": response.input_tokens if response else None,
            "provider_cached_input_tokens": response.cached_input_tokens if response else None,
            "provider_output_tokens": response.output_tokens if response else None,
            "provider_latency_ms": response.latency_ms if response else None,
        }
        if result is not None:
            event.update(result.metrics.model_dump())
        with self._lock:
            self._events.append(event)
            self._counts[f"{operation}.{outcome}"] += 1
            self._total += 1
            self._provider_attempts += int(provider_attempted)
        return request_id

    def snapshot(self) -> list[dict]:
        return self.report()["events"]

    def report(self) -> dict:
        with self._lock:
            return {
                "scope": "bounded_single_process_history",
                "counter_scope": "completed_route_operations_since_process_start",
                "event_capacity": self._capacity,
                "operations_recorded": self._total,
                "events_evicted": self._total - len(self._events),
                "provider_attempts": self._provider_attempts,
                "outcome_counts": dict(sorted(self._counts.items())),
                "events": [dict(event) for event in self._events],
            }
