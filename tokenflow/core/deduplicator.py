import re

from tokenflow.models import Decision, Document

HORIZONTAL_WHITESPACE = re.compile(r"[ \t]+")


def whitespace_key(doc: Document) -> tuple[str, str, str]:
    # Preserve word order, case, punctuation, and every line/paragraph boundary.
    return doc.source, doc.format, HORIZONTAL_WHITESPACE.sub(" ", doc.content)


def deduplicate(
    docs: tuple[Document, ...], protected: dict[str, str], *, near: bool
) -> tuple[list[Document], list[Decision]]:
    kept: list[Document] = []
    decisions: list[Decision] = []
    exact: dict[tuple[str, str, str], Document] = {}
    whitespace: dict[tuple[str, str, str], Document] = {}
    for doc in docs:
        key = (doc.source, doc.format, doc.content)
        previous = exact.get(key)
        if doc.id in protected:
            kept.append(doc)
            exact.setdefault(key, doc)
            if near:
                whitespace.setdefault(whitespace_key(doc), doc)
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
        # High lexical similarity is insufficient: a tiny edit can reverse a fact.
        # Only horizontal whitespace may differ, and callers still opt into loss.
        match = whitespace.get(whitespace_key(doc)) if near else None
        if match:
            decisions.append(
                Decision(
                    document_id=doc.id,
                    action="remove_near",
                    reason="opt_in_horizontal_whitespace_duplicate",
                    retained_document_id=match.id,
                )
            )
        else:
            kept.append(doc)
            exact[key] = doc
            if near:
                whitespace.setdefault(whitespace_key(doc), doc)
    return kept, decisions
