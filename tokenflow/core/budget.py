from tokenflow.core.tokenizer import Tokenizer, render_messages
from tokenflow.models import BudgetExceeded, Decision, Document, OptimizationRequest


def enforce_budget(
    request: OptimizationRequest,
    docs: list[Document],
    protected: dict[str, str],
    scores: dict[str, float],
    tokenizer: Tokenizer,
) -> tuple[list[Document], list[Decision]]:
    decisions = []
    remaining = list(docs)
    count = tokenizer.count(render_messages(request, tuple(remaining)))
    if count <= request.input_budget:
        return remaining, decisions
    removable = (
        []
        if request.mode == "conservative"
        else sorted(
            [d for d in docs if d.id not in protected],
            key=lambda d: (scores[d.id], d.id),
        )
    )
    for doc in removable:
        remaining = [d for d in remaining if d.id != doc.id]
        decisions.append(
            Decision(
                document_id=doc.id,
                action="drop_budget",
                reason="lowest_ranked_unprotected_context",
                score=scores[doc.id],
            )
        )
        count = tokenizer.count(render_messages(request, tuple(remaining)))
        if count <= request.input_budget:
            return remaining, decisions
    raise BudgetExceeded(request.input_budget, count)
