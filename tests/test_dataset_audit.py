import copy
import json
from pathlib import Path

from tokenflow.cli import main
from tokenflow.dataset_audit import audit_dataset

DATASET = Path(__file__).resolve().parents[1] / "benchmarks" / "synthetic-v1.jsonl"


def case(case_id, family, query="PRIVATE_QUERY", split="holdout"):
    return {
        "id": case_id,
        "family": family,
        "category": "qa",
        "split": split,
        "request": {"query": query},
        "required_context": [],
        "answer_contains": ["PRIVATE_EXPECTED"],
    }


def audit(tmp_path, cases):
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in cases))
    return audit_dataset(path)


def test_synthetic_audit_does_not_claim_independence_or_quality():
    report = audit_dataset(DATASET)
    assert report["cases"] == 100
    assert report["declared_families"] == 20
    assert report["independent_provenance_verified"] is None
    assert report["product_gate"] == "not_evaluated"


def test_detects_duplicate_input_despite_different_ids_and_defaults(tmp_path):
    first = case("a", "first", split="dev")
    second = case("b", "second")
    second["request"].update(instructions="", input_budget=100, mode="balanced")
    report = audit(tmp_path, [first, second])
    assert report["unique_baseline_inputs"] == 1
    assert report["cross_family_duplicate_groups"][0]["case_ids"] == ["a", "b"]
    assert report["cross_split_duplicate_groups"]
    assert not report["mechanical_checks_passed"]
    assert "PRIVATE_QUERY" not in json.dumps(report)
    assert "PRIVATE_EXPECTED" not in json.dumps(report)


def test_family_split_leakage_with_distinct_inputs(tmp_path):
    report = audit(tmp_path, [case("a", "same", "one", "dev"), case("b", "same", "two")])
    assert report["families_spanning_splits"] == [{"family": "same", "splits": ["dev", "holdout"]}]
    assert not report["cross_split_duplicate_groups"]
    assert not report["mechanical_checks_passed"]


def test_conflicting_labels_across_identical_cases(tmp_path):
    first = case("a", "same")
    second = copy.deepcopy(first)
    second.update(id="b", answer_contains=["something else"], answer_forbidden=["private_expected"])
    report = audit(tmp_path, [first, second])
    assert report["conflicting_answer_check_groups"] == [["a", "b"]]


def test_related_variants_are_not_counted_as_independent_conversations(tmp_path):
    report = audit(
        tmp_path, [case("a", "same"), case("b", "same"), case("c", "other", "different")]
    )
    assert report["cases"] == 3
    assert report["declared_families"] == 2
    assert report["unique_baseline_inputs"] == 2
    assert report["duplicate_baseline_groups"]
    assert report["mechanical_checks_passed"]
    assert report["independent_provenance_verified"] is None


def test_audit_cli_has_no_model_requirement(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        "sys.argv", ["tokenflow", "audit-dataset", str(DATASET), "--output", str(output)]
    )
    main()
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads(output.read_text())
    assert report["cases"] == 100
