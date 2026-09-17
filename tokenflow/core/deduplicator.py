from difflib import SequenceMatcher

from tokenflow.core.relevance import terms
from tokenflow.models import Decision, Document


def deduplicate(
    docs: tuple[Document, ...], protected: dict[str, str], *, near: bool
) -> tuple[list[Document], list[Decision]]:
    kept: list[Document] = []
    decisions: list[Decision] = []
    exact: dict[tuple[str, str, str], Document] = {}
    comparisons_left = 64
    for doc in docs:
        key = (doc.source, doc.format, doc.content)
        previous = exact.get(key)
        if doc.id in protected:
            kept.append(doc)
            exact.setdefault(key, doc)
            continue
        if previous:
            decisions.append(
                Decision(
                    document_id=doc.id,
                    action="remove_exact",
                    reason="identical_content_and_source",
                    retained_document_id=previous.id,
                )
            )
            continue
        # Bounded by request's 200-document limit. Order/negation-sensitive lexical
        # matching is opt-in and still lossy, never advertised as semantic equivalence.
        match = None
        # Bound character alignment work. Large chunks still get exact deduplication;
        # skipping an expensive approximate comparison retains more context safely.
        if near and len(doc.content) <= 1024:
            tokens = terms(doc.content)
            for old in kept:
                if old.source != doc.source or old.format != doc.format:
                    continue
                if len(old.content) > 1024 or comparisons_left <= 0:
                    continue
                other = terms(old.content)
                similarity = len(tokens & other) / max(1, len(tokens | other))
                comparisons_left -= 1
                if (
                    similarity >= 0.95
                    and SequenceMatcher(None, old.content, doc.content, autojunk=False).ratio()
                    >= 0.98
                ):
                    match = old
                    break
        if match:
            decisions.append(
                Decision(
                    document_id=doc.id,
                    action="remove_near",
                    reason="opt_in_lexical_near_duplicate",
                    retained_document_id=match.id,
                )
            )
        else:
            kept.append(doc)
            exact[key] = doc
    return kept, decisions
