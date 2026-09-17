from time import perf_counter
from typing import Protocol

from pydantic import Field

from tokenflow.models import StrictModel


class ProviderResult(StrictModel):
    text: str
    model: str
    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0)


class Provider(Protocol):
    def generate(self, messages: tuple[dict[str, str], ...]) -> ProviderResult: ...


class OpenAIProvider:
    def __init__(self, model: str, *, max_output_tokens: int = 512, client=None):
        if not model.strip():
            raise ValueError("A model must be supplied explicitly")
        if type(max_output_tokens) is not int or max_output_tokens < 1:
            raise ValueError("max_output_tokens must be a positive integer")
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=30, max_retries=0)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.client = client

    def generate(self, messages: tuple[dict[str, str], ...]) -> ProviderResult:
        start = perf_counter()
        response = self.client.responses.create(
            model=self.model,
            input=list(messages),
            store=False,
            max_output_tokens=self.max_output_tokens,
            truncation="disabled",
        )
        if response.status != "completed" or response.usage is None:
            raise RuntimeError("Provider response incomplete or usage missing")
        if not response.output_text:
            raise RuntimeError("Provider returned no text")
        usage = response.usage
        details = getattr(usage, "input_tokens_details", None)
        return ProviderResult(
            text=response.output_text,
            model=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=getattr(details, "cached_tokens", 0) or 0,
            latency_ms=(perf_counter() - start) * 1000,
        )
