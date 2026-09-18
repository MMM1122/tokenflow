from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from tokenflow.api.app import create_app
from tokenflow.llm.providers import ProviderResult
from tokenflow.metrics.tracker import MetricsTracker

AUTH = {"Authorization": "Bearer test-gateway-key"}


class FailingProvider:
    def __init__(self):
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        raise RuntimeError("PRIVATE_PROMPT PRIVATE_KEY PRIVATE_RESPONSE")


def metrics(client):
    result = client.get("/v1/metrics", headers=AUTH)
    assert result.status_code == 200
    return result.json()


def test_failed_generation_is_correlated_and_keeps_billing_unknown(engine):
    provider = FailingProvider()
    client = TestClient(create_app(engine, provider, api_key="test-gateway-key"))
    response = client.post("/v1/generate", json={"query": "PRIVATE_PROMPT"}, headers=AUTH)
    assert response.status_code == 502
    report = metrics(client)
    event = report["events"][0]
    assert event["request_id"] == response.headers["X-Request-ID"]
    assert report["outcome_counts"] == {"generate.provider_error": 1}
    assert report["provider_attempts"] == provider.calls == 1
    assert event["billing_usage"] == "unknown"
    assert event["provider_input_tokens"] is None
    assert event["provider_output_tokens"] is None
    assert event["provider_latency_ms"] is None
    assert event["total_latency_ms"] >= event["optimization_ms"]
    assert "PRIVATE_" not in response.text
    assert "PRIVATE_" not in client.get("/v1/metrics", headers=AUTH).text


def test_success_has_same_header_and_body_id(engine):
    client = TestClient(create_app(engine, FailingProvider(), api_key="test-gateway-key"))
    response = client.post("/v1/optimize", json={"query": "private input"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == response.json()["request_id"]
    event = metrics(client)["events"][0]
    assert event["request_id"] == response.json()["request_id"]
    assert event["operation"] == "optimize"
    assert event["outcome"] == "success"
    assert event["billing_usage"] == "not_called"
    assert not event["provider_attempted"]


@pytest.mark.parametrize("operation", ["optimize", "generate"])
def test_budget_failure_recorded_without_provider_attempt(engine, operation):
    provider = FailingProvider()
    client = TestClient(create_app(engine, provider, api_key="test-gateway-key"))
    response = client.post(
        f"/v1/{operation}", json={"query": "private", "input_budget": 1}, headers=AUTH
    )
    assert response.status_code == 422
    report = metrics(client)
    assert report["outcome_counts"] == {f"{operation}.budget_exceeded": 1}
    assert report["events"][0]["request_id"] == response.headers["X-Request-ID"]
    assert report["events"][0]["billing_usage"] == "not_called"
    assert report["provider_attempts"] == provider.calls == 0


def test_missing_provider_is_counted_without_provider_attempt(engine, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    client = TestClient(create_app(engine, api_key="test-gateway-key"))
    result = client.post("/v1/generate", json={"query": "test"}, headers=AUTH)
    assert result.status_code == 503
    report = metrics(client)
    assert report["outcome_counts"] == {"generate.provider_unavailable": 1}
    assert report["events"][0]["request_id"] == result.headers["X-Request-ID"]
    assert report["provider_attempts"] == 0


def test_auth_and_schema_rejection_never_execute_or_enter_route_metrics(engine):
    provider = FailingProvider()
    client = TestClient(create_app(engine, provider, api_key="test-gateway-key"))
    assert client.post("/v1/generate", json={"query": "test"}).status_code == 401
    assert client.post("/v1/generate", json={"query": ""}, headers=AUTH).status_code == 422
    assert client.get("/v1/metrics").status_code == 401
    report = metrics(client)
    assert report["operations_recorded"] == provider.calls == 0
    assert report["events"] == []


def test_internal_optimizer_failure_is_recorded_without_leaking_details():
    class BrokenOptimizer:
        def optimize(self, request):
            raise RuntimeError("PRIVATE_INTERNAL_INPUT")

    provider = FailingProvider()
    client = TestClient(create_app(BrokenOptimizer(), provider, api_key="test-gateway-key"))
    response = client.post("/v1/generate", json={"query": "private"}, headers=AUTH)
    assert response.status_code == 500
    assert "PRIVATE_INTERNAL_INPUT" not in response.text
    report = metrics(client)
    assert report["outcome_counts"] == {"generate.optimizer_error": 1}
    assert report["events"][0]["request_id"] == response.headers["X-Request-ID"]
    assert report["provider_attempts"] == provider.calls == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"cached_input_tokens": 101},
        {"latency_ms": float("nan")},
        {"latency_ms": float("inf")},
        {"text": ""},
    ],
)
def test_malformed_provider_result_is_failure_not_success(engine, changes):
    class MalformedProvider:
        def generate(self, messages):
            # model_copy deliberately bypasses validation; the boundary must revalidate.
            return ProviderResult(
                text="PRIVATE_RESPONSE",
                model="test",
                input_tokens=100,
                output_tokens=5,
                latency_ms=1,
            ).model_copy(update=changes)

    client = TestClient(create_app(engine, MalformedProvider(), api_key="test-gateway-key"))
    result = client.post("/v1/generate", json={"query": "test"}, headers=AUTH)
    assert result.status_code == 502
    report = metrics(client)
    assert report["outcome_counts"] == {"generate.provider_error": 1}
    assert report["events"][0]["billing_usage"] == "unknown"
    assert "PRIVATE_RESPONSE" not in result.text


def test_counters_survive_history_eviction_and_snapshot_mutation():
    tracker = MetricsTracker(capacity=2)
    for _ in range(5):
        tracker.record(
            operation="generate",
            outcome="provider_error",
            total_latency_ms=1,
            provider_attempted=True,
        )
    report = tracker.report()
    assert report["operations_recorded"] == report["provider_attempts"] == 5
    assert report["events_evicted"] == 3
    assert len(report["events"]) == 2
    assert report["outcome_counts"] == {"generate.provider_error": 5}
    report["events"][0]["outcome"] = "changed"
    report["outcome_counts"].clear()
    assert tracker.report()["events"][0]["outcome"] == "provider_error"
    assert tracker.report()["outcome_counts"] == {"generate.provider_error": 5}


def test_concurrent_recording_preserves_counts_and_unique_request_ids():
    tracker = MetricsTracker(capacity=10)

    def record(_):
        return tracker.record(operation="optimize", outcome="success", total_latency_ms=1)

    with ThreadPoolExecutor(max_workers=8) as workers:
        ids = list(workers.map(record, range(100)))
    report = tracker.report()
    assert len(set(ids)) == report["operations_recorded"] == 100
    assert report["events_evicted"] == 90
    assert report["outcome_counts"] == {"optimize.success": 100}
