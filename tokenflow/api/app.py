"""M2 local preview; deliberately no persistence or production deployment claims."""

import hmac
import os
from time import perf_counter

from fastapi import Depends, FastAPI, Header, HTTPException, Response

from tokenflow.core.optimizer import Optimizer
from tokenflow.llm.providers import OpenAIProvider, Provider, ProviderResult
from tokenflow.metrics.tracker import MetricsTracker, Operation
from tokenflow.models import BudgetExceeded, OptimizationRequest


def create_app(
    optimizer: Optimizer | None = None,
    provider: Provider | None = None,
    api_key: str | None = None,
) -> FastAPI:
    engine = optimizer or Optimizer()
    tracker = MetricsTracker()
    key = api_key if api_key is not None else os.getenv("TOKENFLOW_API_KEY")
    if provider is None and os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL"):
        provider = OpenAIProvider(os.environ["OPENAI_MODEL"])
    app = FastAPI(title="TokenFlow local preview", version="0.1.0")

    def authenticate(authorization: str | None = Header(default=None)):
        if not key:
            raise HTTPException(503, "Set TOKENFLOW_API_KEY to enable this endpoint")
        if not authorization or not hmac.compare_digest(
            authorization.encode("utf-8"), ("Bearer " + key).encode("utf-8")
        ):
            raise HTTPException(401, "Invalid gateway credentials")

    def elapsed(started: float) -> float:
        return (perf_counter() - started) * 1000

    def optimize_request(request: OptimizationRequest, operation: Operation, started: float):
        try:
            return engine.optimize(request)
        except BudgetExceeded as exc:
            request_id = tracker.record(
                operation=operation, outcome="budget_exceeded", total_latency_ms=elapsed(started)
            )
            raise HTTPException(
                422,
                {
                    "code": "budget_exceeded",
                    "budget": exc.budget,
                    "minimum_tokens": exc.minimum_tokens,
                },
                headers={"X-Request-ID": request_id},
            ) from None
        except Exception:
            request_id = tracker.record(
                operation=operation, outcome="optimizer_error", total_latency_ms=elapsed(started)
            )
            raise HTTPException(
                500, "Context optimization failed", headers={"X-Request-ID": request_id}
            ) from None

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": "local_preview", "provider_ready": provider is not None}

    @app.post("/v1/optimize")
    def optimize(request: OptimizationRequest, http_response: Response):
        # Public local endpoint never invokes a paid provider.
        started = perf_counter()
        result = optimize_request(request, "optimize", started)
        request_id = tracker.record(
            result, operation="optimize", outcome="success", total_latency_ms=elapsed(started)
        )
        http_response.headers["X-Request-ID"] = request_id
        return {"request_id": request_id, **result.model_dump()}

    @app.post("/v1/generate", dependencies=[Depends(authenticate)])
    def generate(request: OptimizationRequest, http_response: Response):
        started = perf_counter()
        if provider is None:
            request_id = tracker.record(
                operation="generate",
                outcome="provider_unavailable",
                total_latency_ms=elapsed(started),
            )
            raise HTTPException(
                503,
                "Configure OPENAI_API_KEY and OPENAI_MODEL",
                headers={"X-Request-ID": request_id},
            )
        optimized = optimize_request(request, "generate", started)
        try:
            response = ProviderResult.model_validate(provider.generate(optimized.messages))
        except Exception:
            # Upstream exceptions and validation errors can contain private data.
            request_id = tracker.record(
                optimized,
                operation="generate",
                outcome="provider_error",
                total_latency_ms=elapsed(started),
                provider_attempted=True,
            )
            raise HTTPException(
                502,
                "Upstream generation failed; no response returned",
                headers={"X-Request-ID": request_id},
            ) from None
        request_id = tracker.record(
            optimized,
            operation="generate",
            outcome="success",
            total_latency_ms=elapsed(started),
            provider_attempted=True,
            response=response,
        )
        http_response.headers["X-Request-ID"] = request_id
        return {
            "request_id": request_id,
            "response": response.model_dump(),
            "optimization": optimized.model_dump(),
        }

    @app.get("/v1/metrics", dependencies=[Depends(authenticate)])
    def metrics():
        return tracker.report()

    return app
