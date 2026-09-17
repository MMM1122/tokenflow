import re

from tokenflow.core.relevance import RelevanceScorer


def extract_paragraphs(text: str, scorer: RelevanceScorer) -> str:
    """Select original paragraphs; no generated facts or claim of losslessness."""
    paragraphs = re.split(r"\n\s*\n", text)
    if len(paragraphs) < 3:
        return text
    scores = [scorer.score(p) for p in paragraphs]
    peak = max(scores)
    # With no signal, leave selection to the explicit budget stage.
    if peak == 0:
        return text
    selected = [p for i, p in enumerate(paragraphs) if i == 0 or scores[i] >= peak * 0.5]
    if len(selected) == len(paragraphs):
        return text
    return "\n\n".join(selected)
