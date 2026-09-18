"""Known context-loss counterexamples; these are regression cases, not quality scores."""

import pytest

from tokenflow import Document, OptimizationRequest
from tokenflow.core.tokenizer import render_messages
from tokenflow.models import BudgetExceeded

CONTEXT = (
    "The release guide describes deployment workflows, environment setup, artifact packaging, "
    "source review, team coordination, service monitoring, incident response, rollback procedures, "
    "documentation updates, dependency checks, access management, ownership boundaries, approval "
    "records, operational readiness, maintenance windows, staging validation, "
    "and production support. "
) * 2


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Alice approves Bob.", "Bob approves Alice."),
        ("The service is active.", "The service is inactive."),
        ("Shipping is free.", "Shipping is free?"),
        ("Compare a > b.", "Compare a < b."),
        ("Select US.", "Select us."),
    ],
)
def test_similar_but_distinct_claims_are_not_merged(engine, first, second):
    documents = tuple(
        Document(id=str(i), source="guide", content=CONTEXT + text)
        for i, text in enumerate((first, second))
    )
    request = OptimizationRequest(
        query="Summarize the guide.", documents=documents, mode="balanced"
    )
    result = engine.optimize(request)
    assert result.documents == documents
    assert not result.metrics.lossy


@pytest.mark.parametrize(
    "constraint",
    [
        "No customer data may leave the private network.",
        "The export job doesn't send private records.",
        "The service isn't available outside the private network.",
        "This user can't approve deployments.",
        "The export job doesn\u2019t send private records.",
        "The exporter won\u2019t send private records.",
        "Neither deployment target is permitted.",
        "The team permits neither export nor external storage.",
    ],
)
def test_negated_constraints_survive_or_fail_explicitly(engine, constraint):
    document = Document(id="policy", source="policy-guide", content=constraint)
    request = OptimizationRequest(
        query="Summarize deployment.", documents=(document,), mode="balanced"
    )
    result = engine.optimize(request)
    assert result.documents == (document,)
    assert not result.metrics.lossy
    empty_budget = engine.tokenizer.count(render_messages(request, ()))
    with pytest.raises(BudgetExceeded):
        engine.optimize(request.model_copy(update={"input_budget": empty_budget}))


def test_horizontal_whitespace_dedup_keeps_exact_source_text(engine):
    first = Document(id="a", source="guide", content="Docker  packages\tapplication dependencies.")
    second = Document(id="b", source="guide", content="Docker packages application dependencies.")
    result = engine.optimize(
        OptimizationRequest(query="Explain containers.", documents=(first, second), mode="balanced")
    )
    assert result.documents == (first,)
    decision = next(d for d in result.decisions if d.action == "remove_near")
    assert decision.retained_document_id == first.id
    assert decision.reason == "opt_in_horizontal_whitespace_duplicate"
    assert result.metrics.lossy


def test_paragraph_boundaries_are_not_whitespace_duplicates(engine):
    documents = (
        Document(id="a", source="guide", content="Alpha beta\n\nGamma delta"),
        Document(id="b", source="guide", content="Alpha beta Gamma delta"),
    )
    result = engine.optimize(
        OptimizationRequest(query="Explain the guide.", documents=documents, mode="balanced")
    )
    assert result.documents == documents


@pytest.mark.parametrize("format", ["code", "json", "table"])
def test_whitespace_matching_never_removes_structured_documents(engine, format):
    documents = (
        Document(id="a", source="guide", content="alpha  beta", format=format),
        Document(id="b", source="guide", content="alpha beta", format=format),
    )
    result = engine.optimize(
        OptimizationRequest(query="Explain.", documents=documents, mode="balanced")
    )
    assert result.documents == documents
