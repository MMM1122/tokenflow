import re

from tokenflow.models import Document, OptimizationRequest

SENSITIVE = re.compile(
    r"\d|\b(?:no|not|never|neither|nor|must|shall|required|only|except|unless|without|budget|"
    r"prefer|avoid|allergic|allergy|password|secret|deadline|cannot|[a-z]+n['\u2019]t)\b|"
    r"\u4e0d|\u65e0|\u6ca1|\u5fc5\u987b|\u53ea\u80fd|\u7981\u6b62|\u9884\u7b97|\u504f\u597d|\u8fc7\u654f|\u5bc6\u7801|\u622a\u6b62|\u9664\u975e|```|https?://|"
    r"^\s*(?:def |class |import |SELECT |\{|\[)|\|.+\|",
    re.IGNORECASE | re.MULTILINE,
)


def sensitive(text: str) -> bool:
    return bool(SENSITIVE.search(text))


def protection_reason(doc: Document, request: OptimizationRequest) -> str | None:
    if doc.protected:
        return "caller_protected"
    if doc.format != "prose":
        return "structured_format"
    # Check all instructions/history, not just the latest query (e.g. 'use that source').
    references = "\n".join(
        [request.instructions, request.query, *(m.content for m in request.history)]
    )
    if doc.source.casefold() in references.casefold():
        return "source_referenced"
    if sensitive(doc.content):
        return "constraint_numeric_or_code_heuristic"
    return None
