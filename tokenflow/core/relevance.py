import math
import re
from collections import Counter

STOPWORDS = frozenset(
    "a an the is are was were be to of in on for with and or this that it as by from "
    "what how please tell explain me about using use document answer".split()
)


def terms(text: str) -> frozenset[str]:
    latin = re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*", text.lower())
    chinese = re.findall(r"[\u3400-\u9fff]+", text)
    # Character bigrams keep Chinese useful without pretending this is semantic search.
    cjk = [run[i : i + 2] for run in chinese for i in range(max(1, len(run) - 1))]
    return frozenset([t for t in latin if t not in STOPWORDS] + cjk)


class RelevanceScorer:
    def __init__(self, query: str, corpus: list[str]):
        self.query = terms(query)
        df = Counter(t for text in corpus for t in terms(text))
        self.idf = {t: 1 + math.log((len(corpus) + 1) / (df[t] + 1)) for t in self.query}

    def score(self, text: str) -> float:
        tokens = terms(text)
        denominator = sum(self.idf.values())
        if not denominator:
            return 0.0
        return sum(self.idf[t] for t in self.query & tokens) / denominator


def importance_score(relevance: float, recency: float, importance: float) -> float:
    return 0.65 * relevance + 0.10 * recency + 0.25 * importance
