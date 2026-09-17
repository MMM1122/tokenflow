"""Inspect benchmark isolation without exposing request text or claiming provenance."""

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from tokenflow.core.tokenizer import render_messages
from tokenflow.datasets import load_cases
from tokenflow.models import OptimizationRequest
from tokenflow.run_store import digest


def audit_dataset(dataset: Path) -> dict:
    cases = load_cases(dataset)
    by_input, by_family = defaultdict(list), defaultdict(list)
    for case in cases:
        request = OptimizationRequest.model_validate(case["request"])
        # Match actual baseline messages, not incidental IDs or omitted defaults.
        by_input[digest(render_messages(request))].append(case)
        by_family[case["family"]].append(case)

    duplicates, conflicts = [], []
    for members in by_input.values():
        if len(members) < 2:
            continue
        group = {
            "case_ids": sorted(case["id"] for case in members),
            "families": sorted({case["family"] for case in members}),
            "splits": sorted({case["split"] for case in members}),
        }
        duplicates.append(group)
        expected = {term.casefold() for case in members for term in case["answer_contains"]}
        forbidden = {
            term.casefold() for case in members for term in case.get("answer_forbidden", [])
        }
        if any(bad in good for bad in forbidden for good in expected):
            conflicts.append(group["case_ids"])
    duplicates.sort(key=lambda group: group["case_ids"])
    family_overlap = [
        {"family": family, "splits": sorted({case["split"] for case in members})}
        for family, members in sorted(by_family.items())
        if len({case["split"] for case in members}) > 1
    ]
    cross_family = [group for group in duplicates if len(group["families"]) > 1]
    cross_split = [group for group in duplicates if len(group["splits"]) > 1]
    blockers = []
    if cross_family:
        blockers.append("identical_baseline_inputs_assigned_to_different_families")
    if cross_split or family_overlap:
        blockers.append("development_holdout_isolation_requires_review")
    if conflicts:
        blockers.append("conflicting_answer_checks_for_identical_inputs")
    if len(by_family) < 2:
        blockers.append("fewer_than_two_declared_families")
    return {
        "format_version": 1,
        "kind": "dataset_readiness_audit",
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "cases": len(cases),
        "declared_families": len(by_family),
        "unique_baseline_inputs": len(by_input),
        "cases_by_category": dict(sorted(Counter(c["category"] for c in cases).items())),
        "cases_by_split": dict(sorted(Counter(c["split"] for c in cases).items())),
        "family_sizes": {family: len(members) for family, members in sorted(by_family.items())},
        "duplicate_baseline_groups": duplicates,
        "cross_family_duplicate_groups": cross_family,
        "cross_split_duplicate_groups": cross_split,
        "families_spanning_splits": family_overlap,
        "conflicting_answer_check_groups": sorted(conflicts),
        "cases_without_required_context_checks": sorted(
            case["id"] for case in cases if not case["required_context"]
        ),
        "readiness_findings": blockers,
        "mechanical_checks_passed": not blockers,
        "independent_provenance_verified": None,
        "product_gate": "not_evaluated",
        "limitations": [
            "Declared families are labels, not proof of independent conversations.",
            "Exact matching cannot detect paraphrases, shared origins, or all split leakage.",
            "Consent, redaction, provenance, and holdout integrity require human review.",
            "Reports omit request text but retain potentially sensitive case and family IDs.",
            "No model calls were made; passing these checks does not establish product quality.",
        ],
    }
