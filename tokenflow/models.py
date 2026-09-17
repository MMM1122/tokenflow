from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Message(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)


class Document(StrictModel):
    id: str = Field(min_length=1, max_length=100)
    source: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=100_000)
    format: Literal["prose", "code", "json", "table"] = "prose"
    protected: bool = False
    importance: float = Field(default=0.5, ge=0, le=1)


class OptimizationRequest(StrictModel):
    query: str = Field(min_length=1, max_length=100_000)
    instructions: str = Field(default="", max_length=100_000)
    history: tuple[Message, ...] = Field(default=(), max_length=200)
    documents: tuple[Document, ...] = Field(default=(), max_length=200)
    mode: Literal["conservative", "balanced"] = "conservative"
    input_budget: int = Field(default=8000, ge=1, le=200_000)

    @model_validator(mode="after")
    def valid_input(self):
        ids = [d.id for d in self.documents]
        if len(set(ids)) != len(ids):
            raise ValueError("Document ids must be unique")
        total = sum(len(m.content) for m in self.history)
        total += sum(len(d.content) for d in self.documents)
        total += len(self.query) + len(self.instructions)
        if total > 500_000:
            raise ValueError("Text limit is 500,000 characters per request")
        return self


class Decision(StrictModel):
    document_id: str
    action: Literal["keep", "remove_exact", "remove_near", "compress", "drop_budget"]
    reason: str
    score: float | None = None
    retained_document_id: str | None = None


class OptimizationMetrics(StrictModel):
    tokens_before: int
    tokens_after: int
    reduction_rate: float
    optimization_ms: float
    tokenizer: str
    token_measurement: str = "canonical_messages_json_estimate_not_provider_billing"
    lossy: bool


class OptimizationResult(StrictModel):
    messages: tuple[dict[str, str], ...]
    documents: tuple[Document, ...]
    decisions: tuple[Decision, ...]
    metrics: OptimizationMetrics
    warnings: tuple[str, ...] = ()


class BudgetExceeded(ValueError):
    def __init__(self, budget: int, minimum_tokens: int):
        self.budget = budget
        self.minimum_tokens = minimum_tokens
        super().__init__(
            f"Input budget {budget} cannot fit {minimum_tokens} tokens retained by policy; "
            "increase budget or explicitly revise the supplied context."
        )
