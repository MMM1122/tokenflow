from time import perf_counter

from tokenflow.core.budget import enforce_budget
from tokenflow.core.classifier import protection_reason
from tokenflow.core.compressor import extract_paragraphs
from tokenflow.core.deduplicator import deduplicate
from tokenflow.core.relevance import RelevanceScorer, importance_score
from tokenflow.core.tokenizer import Tokenizer, render_messages
from tokenflow.models import (
    Decision,
    OptimizationMetrics,
    OptimizationRequest,
    OptimizationResult,
)


class Optimizer:
    def __init__(self, tokenizer: Tokenizer | None = None):
        self.tokenizer = tokenizer or Tokenizer()

    def optimize(self, request: OptimizationRequest) -> OptimizationResult:
        started = perf_counter()
        before = self.tokenizer.count(render_messages(request))
        protected = {
            d.id: reason for d in request.documents if (reason := protection_reason(d, request))
        }
        docs, decisions = deduplicate(
            request.documents,
            protected,
            near=request.mode == "balanced",
        )
        scorer = RelevanceScorer(request.query, [d.content for d in docs])
        scores = {
            d.id: importance_score(
                scorer.score(d.content),
                (i + 1) / max(1, len(docs)),
                d.importance,
            )
            for i, d in enumerate(docs)
        }
        if request.mode == "balanced":
            for i, doc in enumerate(docs):
                if doc.id in protected:
                    continue
                content = extract_paragraphs(doc.content, scorer)
                if self.tokenizer.count_text(content) < self.tokenizer.count_text(doc.content):
                    candidate = doc.model_copy(update={"content": content})
                    # Compare the actual serialized payload too, including escaping.
                    if self.tokenizer.count(render_messages(request, (candidate,))) < (
                        self.tokenizer.count(render_messages(request, (doc,)))
                    ):
                        docs[i] = candidate
                        decisions.append(
                            Decision(
                                document_id=doc.id,
                                action="compress",
                                reason="opt_in_verbatim_paragraph_selection",
                                score=scores[doc.id],
                            )
                        )
        docs, budget_decisions = enforce_budget(request, docs, protected, scores, self.tokenizer)
        decisions.extend(budget_decisions)
        messages = render_messages(request, tuple(docs))
        after = self.tokenizer.count(messages)
        # BPE boundaries can be surprising. A result may never grow the prompt.
        if after > before:
            docs, decisions = list(request.documents), []
            messages, after = render_messages(request), before
        for doc in docs:
            if not any(d.document_id == doc.id and d.action == "compress" for d in decisions):
                decisions.append(
                    Decision(
                        document_id=doc.id,
                        action="keep",
                        reason=protected.get(doc.id, "retained_by_policy"),
                        score=scores.get(doc.id),
                    )
                )
        lossy = any(d.action in {"remove_near", "compress", "drop_budget"} for d in decisions)
        warnings = ["Local token count estimates provider input; provider framing may differ."]
        if lossy:
            warnings.append("Lossy optimization applied; answer quality has not been validated.")
        return OptimizationResult(
            messages=messages,
            documents=tuple(docs),
            decisions=tuple(decisions),
            metrics=OptimizationMetrics(
                tokens_before=before,
                tokens_after=after,
                reduction_rate=(before - after) / before if before else 0.0,
                optimization_ms=(perf_counter() - started) * 1000,
                tokenizer=self.tokenizer.name,
                lossy=lossy,
            ),
            warnings=tuple(warnings),
        )
