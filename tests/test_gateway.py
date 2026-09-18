from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tokenflow import OptimizationRequest
from tokenflow.api.app import create_app
from tokenflow.llm.client import TokenFlow
from tokenflow.llm.providers import OpenAIProvider, ProviderResult


class FakeProvider:
    def __init__(self):
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        return ProviderResult(
            text="fixture response",
            model="fake-test-model",
            input_tokens=42,
            cached_input_tokens=10,
            output_tokens=3,
            latency_ms=1,
        )


def test_api_optimize_auth_generation_and_metrics(engine):
    provider = FakeProvider()
    client = TestClient(create_app(engine, provider, api_key="test-key"))
    assert client.get("/health").status_code == 200
    assert client.post("/v1/optimize", json={"query": "private text"}).status_code == 200
    assert not provider.calls
    assert client.post("/v1/generate", json={"query": "x"}).status_code == 401
    headers = {"Authorization": "Bearer test-key"}
    response = client.post("/v1/generate", json={"query": "private text"}, headers=headers)
    assert response.status_code == 200
    assert response.json()["response"]["input_tokens"] == 42
    metrics = client.get("/v1/metrics", headers=headers)
    assert metrics.status_code == 200
    assert "private text" not in metrics.text
    assert "fixture response" not in metrics.text
    assert len(provider.calls) == 1
    report = metrics.json()
    assert report["outcome_counts"] == {"generate.success": 1, "optimize.success": 1}
    assert report["operations_recorded"] == 2
    assert report["provider_attempts"] == 1
    event = report["events"][-1]
    assert event["billing_usage"] == "reported"
    assert event["provider_input_tokens"] == 42
    assert event["request_id"] == response.json()["request_id"]
    assert event["request_id"] == response.headers["X-Request-ID"]


def test_impossible_budget_never_reaches_provider(engine):
    provider = FakeProvider()
    client = TestClient(create_app(engine, provider, api_key="key"))
    response = client.post(
        "/v1/generate",
        json={"query": "x", "input_budget": 1},
        headers={"Authorization": "Bearer key"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "budget_exceeded"
    assert not provider.calls


def test_missing_configuration_and_invalid_payload(engine, monkeypatch):
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "TOKENFLOW_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    client = TestClient(create_app(engine))
    assert client.post("/v1/generate", json={"query": "x"}).status_code == 503
    assert client.post("/v1/optimize", json={"query": ""}).status_code == 422


def test_upstream_error_does_not_leak_secret(engine):
    class Broken:
        def generate(self, messages):
            raise RuntimeError("secret-api-key and private prompt")

    client = TestClient(create_app(engine, Broken(), api_key="key"))
    result = client.post(
        "/v1/generate", json={"query": "x"}, headers={"Authorization": "Bearer key"}
    )
    assert result.status_code == 502
    assert "secret-api-key" not in result.text


def test_sdk_routes_the_actual_optimized_messages(engine):
    provider = FakeProvider()
    result = TokenFlow(provider, engine).generate(OptimizationRequest(query="test"))
    assert provider.calls == [result.optimization.messages]


def test_openai_request_contract_and_usage():
    class Responses:
        def create(self, **kwargs):
            assert kwargs["store"] is False
            assert kwargs["truncation"] == "disabled"
            assert kwargs["model"] == "configured-model"
            return SimpleNamespace(
                status="completed",
                model="configured-model-snapshot",
                output_text="answer",
                usage=SimpleNamespace(
                    input_tokens=100,
                    output_tokens=20,
                    input_tokens_details=SimpleNamespace(cached_tokens=70),
                ),
            )

    provider = OpenAIProvider("configured-model", client=SimpleNamespace(responses=Responses()))
    result = provider.generate(({"role": "user", "content": "test"},))
    assert result.cached_input_tokens == 70
    assert result.model == "configured-model-snapshot"


def test_incomplete_provider_output_is_not_counted_as_success():
    responses = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(status="incomplete"))
    with pytest.raises(RuntimeError, match="incomplete"):
        OpenAIProvider("configured-model", client=SimpleNamespace(responses=responses)).generate(())
