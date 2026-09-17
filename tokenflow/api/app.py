"""M2 local preview; deliberately no persistence or production deployment claims."""

import hmac
import os

from fastapi import Depends, FastAPI, Header, HTTPException

from tokenflow.core.optimizer import Optimizer
from tokenflow.llm.providers import OpenAIProvider, Provider
from tokenflow.metrics.tracker import MetricsTracker
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

    def optimize_request(request: OptimizationRequest):
        try:
            return engine.optimize(request)
        except BudgetExceeded as exc:
            raise HTTPException(
                422,
                {
                    "code": "budget_exceeded",
                    "budget": exc.budget,
                    "minimum_tokens": exc.minimum_tokens,
                },
            ) from None

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": "local_preview", "provider_ready": provider is not None}

    @app.post("/v1/optimize")
    def optimize(request: OptimizationRequest):
        # Public local endpoint never invokes a paid provider.
        result = optimize_request(request)
        request_id = tracker.record(result)
        return {"request_id": request_id, **result.model_dump()}

    @app.post("/v1/generate", dependencies=[Depends(authenticate)])
    def generate(request: OptimizationRequest):
        if provider is None:
            raise HTTPException(503, "Configure OPENAI_API_KEY and OPENAI_MODEL")
        optimized = optimize_request(request)
        try:
            response = provider.generate(optimized.messages)
        except Exception:
            # Upstream exception text can contain prompts, credentials or account data.
            raise HTTPException(502, "Upstream generation failed; no response returned") from None
        request_id = tracker.record(
            optimized,
            provider_input_tokens=response.input_tokens,
            provider_cached_input_tokens=response.cached_input_tokens,
            provider_output_tokens=response.output_tokens,
            provider_latency_ms=response.latency_ms,
        )
        return {
            "request_id": request_id,
            "response": response.model_dump(),
            "optimization": optimized.model_dump(),
        }

    @app.get("/v1/metrics", dependencies=[Depends(authenticate)])
    def metrics():
        return {"scope": "bounded_single_process_history", "events": tracker.snapshot()}

    return app
