from pydantic import Field

from tokenflow.llm.providers import ProviderResult
from tokenflow.models import StrictModel


class Prices(StrictModel):
    """Explicit per-million-token prices; no automatic model-price assumptions."""

    input_per_million: float = Field(ge=0, allow_inf_nan=False)
    cached_input_per_million: float = Field(ge=0, allow_inf_nan=False)
    output_per_million: float = Field(ge=0, allow_inf_nan=False)
    currency: str = Field(min_length=1)
    as_of: str = Field(min_length=1)
    model: str = Field(min_length=1)


def cost(result: ProviderResult, prices: Prices | None) -> float | None:
    if result.cached_input_tokens > result.input_tokens:
        raise ValueError("Provider cached tokens cannot exceed input tokens")
    if prices is None:
        return None
    return (
        (result.input_tokens - result.cached_input_tokens) * prices.input_per_million
        + result.cached_input_tokens * prices.cached_input_per_million
        + result.output_tokens * prices.output_per_million
    ) / 1_000_000
