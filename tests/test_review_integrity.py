import json

import pytest

from tokenflow.evaluation import evaluate_reviews
from tokenflow.experiments import run_online
from tokenflow.llm.providers import ProviderResult
from tokenflow.run_store import run_lock


@pytest.fixture
def reviewed_run(tmp_path):
    # Test doubles exercise integrity, not real model quality or independent provenance.
    cases = [
        {
            "id": f"case-{i}",
            "family": f"family-{i}",
            "category": "qa",
            "split": "holdout",
            "request": {"query": f"Question {i}"},
            "required_context": [],
            "answer_contains": ["answer"],
        }
        for i in range(3)
    ]
    dataset = tmp_path / "test.jsonl"
    dataset.write_text("\n".join(json.dumps(case) for case in cases))

    class TestProvider:
        def generate(self, messages):
            return ProviderResult(
                text="test answer",
                model="test-only",
                input_tokens=10,
                output_tokens=2,
                latency_ms=1,
            )

    output = tmp_path / "run"
    run_online(dataset, output, TestProvider())
    reviews = read_lines(output / "blind-review.jsonl")
    for review in reviews:
        for side in ("left", "right"):
            review[f"{side}_scores"] = dict.fromkeys(
                ("correctness", "completeness", "instruction_adherence", "grounding"), 4
            )
            review[f"{side}_critical_violation"] = False
    write_lines(output / "blind-review.jsonl", reviews)
    return output


def read_lines(path):
    return list(map(json.loads, path.read_text().splitlines()))


def write_lines(path, rows):
    path.write_text("\n".join(map(json.dumps, rows)))


def test_complete_sample_still_cannot_pass_product_gate(reviewed_run):
    result = evaluate_reviews(reviewed_run)
    assert result["sample_quality_gate_passed"]
    assert (
        result["product_gate"] == "requires_independent_dataset_provenance_and_cost_latency_review"
    )
    assert len(result["dataset_sha256"]) == len(result["plan_sha256"]) == 64
    assert len(result["review_records_sha256"]) == 64


@pytest.mark.parametrize("field", ["left_answer", "right_answer", "request", "family"])
def test_edited_review_content_is_rejected(reviewed_run, field):
    path = reviewed_run / "blind-review.jsonl"
    rows = read_lines(path)
    rows[0][field] = "PRIVATE_ALTERED_CONTENT"
    write_lines(path, rows)
    with pytest.raises(ValueError, match="content differs") as error:
        evaluate_reviews(reviewed_run)
    assert "PRIVATE_ALTERED_CONTENT" not in str(error.value)


def test_swapped_blinding_key_is_rejected(reviewed_run):
    path = reviewed_run / "review-key.jsonl"
    keys = read_lines(path)
    keys[0]["left"], keys[0]["right"] = keys[0]["right"], keys[0]["left"]
    write_lines(path, keys)
    with pytest.raises(ValueError, match="arm mapping"):
        evaluate_reviews(reviewed_run)


@pytest.mark.parametrize("artifact", ["online.json", "manifest.json", "plan.json", "journal"])
def test_scoring_rejects_changed_source_artifacts(reviewed_run, artifact):
    if artifact == "journal":
        path = next((reviewed_run / "attempts").glob("*.completed.json"))
        data = json.loads(path.read_text())
        data["record"]["text"] = "changed"
    else:
        path = reviewed_run / artifact
        data = json.loads(path.read_text())
        if artifact == "plan.json":
            data["cases"][0]["request"]["query"] = "changed"
        else:
            data["cases"] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        evaluate_reviews(reviewed_run)


def test_review_reordering_preserves_scores_and_fingerprint(reviewed_run):
    original = evaluate_reviews(reviewed_run)
    for name in ("blind-review.jsonl", "review-key.jsonl"):
        path = reviewed_run / name
        write_lines(path, list(reversed(read_lines(path))))
    assert evaluate_reviews(reviewed_run) == original


def test_scoring_refuses_active_run_directory(reviewed_run):
    with run_lock(reviewed_run):
        with pytest.raises(ValueError, match="Another process"):
            evaluate_reviews(reviewed_run)
