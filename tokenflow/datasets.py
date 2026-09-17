"""Validate evaluation data before any paid provider request."""

import json
from pathlib import Path

from pydantic import Field, ValidationError, field_validator, model_validator

from tokenflow.models import OptimizationRequest, StrictModel


class BenchmarkCase(StrictModel):
    id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    category: str = Field(min_length=1)
    split: str = Field(min_length=1)
    variant: int | None = Field(default=None, ge=0)
    request: OptimizationRequest
    required_context: tuple[str, ...]
    answer_contains: tuple[str, ...] = Field(min_length=1)
    answer_forbidden: tuple[str, ...] = ()

    @field_validator("id", "family", "category", "split")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Metadata must not be blank")
        return value

    @field_validator("required_context", "answer_contains", "answer_forbidden")
    @classmethod
    def meaningful_checks(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("Checks must contain nonblank strings")
        if len({value.casefold() for value in values}) != len(values):
            raise ValueError("Checks must be unique")
        return values

    @model_validator(mode="after")
    def labels_match_input(self):
        text = "\n".join(
            [
                self.request.query,
                self.request.instructions,
                *(message.content for message in self.request.history),
                *(document.content for document in self.request.documents),
            ]
        )
        if any(value not in text for value in self.required_context):
            raise ValueError("Required context must be present in the original request")
        if any(
            forbidden.casefold() in expected.casefold()
            for forbidden in self.answer_forbidden
            for expected in self.answer_contains
        ):
            raise ValueError("Expected answers cannot contain forbidden substrings")
        return self


def load_cases(path: Path) -> list[dict]:
    cases, seen = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            case = BenchmarkCase.model_validate(raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            # Input can contain private traces. Report locations, never input values.
            fields = (
                "JSON"
                if isinstance(exc, json.JSONDecodeError)
                else ", ".join(
                    ".".join(map(str, error["loc"])) or "case"
                    for error in exc.errors(include_input=False)
                )
            )
            raise ValueError(f"Invalid benchmark case at line {number}: {fields}") from None
        if case.id in seen:
            raise ValueError(f"Duplicate case id at line {number}")
        seen.add(case.id)
        # Preserve the original shape; labels are never merged into the request.
        cases.append(raw)
    if not cases:
        raise ValueError("Dataset must be nonempty with unique case ids")
    return cases
