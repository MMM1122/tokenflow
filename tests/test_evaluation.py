import json
from pathlib import Path

import pytest

from tokenflow.evaluation import (
    Prices,
    answer_check,
    cost,
    evaluate_reviews,
    load_cases,
    run_offline,
    run_online,
)
from tokenflow.llm.providers import ProviderResult

DATASET = Path(__file__).resolve().parents[1] / "benchmarks" / "synthetic-v1.jsonl"


def test_full_synthetic_suite_and_honest_unmeasured_metrics():
    report = run_offline(DATASET, repeats=1)
    assert report["cases"] == 100
    assert report["independent_scenario_families"] == 20
    assert report["answer_quality_retention"] is None
    assert report["cost_reduction"] is None
    assert report["llm_latency_change_ms"] is None
    assert report["functional_fixture_gate"]
    assert all(row["tokens_after"] <= row["tokens_before"] for row in report["rows"])
    assert report["groups"]["conservative/category/control"]["mean_reduction_rate"] == 0


def test_evaluation_labels_are_not_optimizer_input():
    for case in load_cases(DATASET):
        assert "required_context" not in case["request"]
        assert "answer_contains" not in case["request"]


def test_cost_uses_cached_input_and_output():
    prices = Prices(
        input_per_million=2,
        cached_input_per_million=0.5,
        output_per_million=8,
        currency="USD",
        as_of="fictional-test",
        model="fake-model",
    )
    response = ProviderResult(
        text="x",
        model="fake-model",
        input_tokens=100,
        cached_input_tokens=80,
        output_tokens=50,
        latency_ms=1,
    )
    assert cost(response, prices) == pytest.approx(0.00048)
    assert cost(response, None) is None


def test_failed_online_calls_count_in_denominator_and_prevent_savings_claim(tmp_path):
    class BrokenProvider:
        def generate(self, messages):
            raise RuntimeError("SECRET")

    report = run_online(DATASET, tmp_path, BrokenProvider(), limit=2)
    assert report["failed_calls"] == 4
    assert report["baseline"]["answer_check_score"] == 0
    assert report["answer_quality_retention"] is None
    assert report["cost_reduction"] is None
    assert "SECRET" not in (tmp_path / "calls.jsonl").read_text()
    with pytest.raises(ValueError):
        evaluate_reviews(tmp_path)
    with pytest.raises(ValueError, match="empty output"):
        run_online(DATASET, tmp_path, BrokenProvider(), limit=2)


def test_blind_reviews_require_complete_rubrics_and_detect_regression(tmp_path):
    class FakeProvider:
        def generate(self, messages):
            return ProviderResult(
                text="fake test answer",
                model="fake",
                input_tokens=100,
                output_tokens=3,
                latency_ms=1,
            )

    run_online(DATASET, tmp_path, FakeProvider(), limit=10)
    with pytest.raises(ValueError):
        evaluate_reviews(tmp_path)
    key = {
        r["id"]: r
        for r in map(json.loads, (tmp_path / "review-key.jsonl").read_text().splitlines())
    }
    reviews = list(map(json.loads, (tmp_path / "blind-review.jsonl").read_text().splitlines()))
    dimensions = ["correctness", "completeness", "instruction_adherence", "grounding"]
    for review in reviews:
        for side in ("left", "right"):
            value = 4 if key[review["id"]][side] == "baseline" else 2
            review[f"{side}_scores"] = dict.fromkeys(dimensions, value)
            review[f"{side}_critical_violation"] = False
    (tmp_path / "blind-review.jsonl").write_text("\n".join(map(json.dumps, reviews)))
    result = evaluate_reviews(tmp_path)
    assert result["answer_quality_retention"] == 0.5
    assert not result["sample_quality_gate_passed"]
    assert result["paired_difference_ci95"] == [-0.5, -0.5]


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_online_limit_never_calls_provider(tmp_path, limit):
    class NeverCalled:
        def generate(self, messages):
            pytest.fail("Invalid limit must not incur paid calls")

    with pytest.raises(ValueError, match="positive"):
        run_online(DATASET, tmp_path, NeverCalled(), limit=limit)


def test_budget_failures_count_as_failed_cases_but_not_provider_calls(tmp_path):
    case = load_cases(DATASET)[0]
    case["request"]["input_budget"] = 1
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps(case) + "\n")

    class NeverCalled:
        def generate(self, messages):
            pytest.fail("Impossible budgets must not incur paid calls")

    result = run_online(dataset, tmp_path / "run", NeverCalled(), limit=1)
    assert result["call_count"] == 0
    assert result["failed_calls"] == 2
    assert result["evaluation_arm_records"] == 2


def test_narrow_answer_checks_do_not_ignore_forbidden_outputs():
    assert (
        answer_check("FIFO HACKED", {"answer_contains": ["FIFO"], "answer_forbidden": ["HACKED"]})
        == 0
    )
