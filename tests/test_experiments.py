import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tokenflow.datasets import load_cases
from tokenflow.evaluation import evaluate_reviews
from tokenflow.experiments import build_plan, plan_summary, run_online
from tokenflow.llm.providers import OpenAIProvider, ProviderResult
from tokenflow.metrics.pricing import Prices
from tokenflow.run_store import RunStore, run_lock

DATASET = Path(__file__).resolve().parents[1] / "benchmarks" / "synthetic-v1.jsonl"


class RecordingProvider:
    model = "test-model"
    max_output_tokens = 512

    def __init__(self, crash_on=None, fail=False):
        self.calls = []
        self.crash_on = crash_on
        self.fail = fail

    def generate(self, messages):
        self.calls.append(messages)
        if len(self.calls) == self.crash_on:
            raise KeyboardInterrupt()
        if self.fail:
            raise RuntimeError("PRIVATE_PROVIDER_ERROR")
        return ProviderResult(
            text="test answer",
            model=self.model,
            input_tokens=100,
            cached_input_tokens=20,
            output_tokens=5,
            latency_ms=10,
        )


def test_preflight_has_no_model_calls_or_request_content():
    plan = build_plan(DATASET, model="unverified-model", limit=3)
    summary = plan_summary(plan)
    assert summary["planned_provider_calls"] == 6
    assert summary["actual_provider_calls"] == 0
    assert summary["answer_quality_retention"] is None
    assert summary["uncached_input_plus_output_cap_cost_estimate"] is None
    assert not summary["estimate_is_billing_cap"]
    assert plan["cases"][0]["request"]["query"] not in json.dumps(summary)


def test_preflight_includes_both_arms_output_caps_and_explicit_prices():
    prices = Prices(
        model="test-model",
        currency="USD",
        as_of="test-only",
        input_per_million=2,
        cached_input_per_million=0.1,
        output_per_million=8,
    )
    summary = plan_summary(
        build_plan(DATASET, model="test-model", limit=2, max_output_tokens=100, prices=prices)
    )
    assert summary["configured_total_output_token_cap"] == 400
    estimated = (sum(summary["estimated_input_tokens"].values()) * 2 + 400 * 8) / 1_000_000
    assert summary["uncached_input_plus_output_cap_cost_estimate"] == pytest.approx(estimated)
    with pytest.raises(ValueError, match="Prices model"):
        build_plan(DATASET, model="other", prices=prices)


@pytest.mark.parametrize("bad_checks", [[], [""], [" "], [2], ["Python", "python"]])
def test_invalid_labels_are_rejected_before_paid_calls(tmp_path, bad_checks):
    case = load_cases(DATASET)[0]
    case["answer_contains"] = bad_checks
    case["request"]["instructions"] = "PRIVATE_TRACE"
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(case))
    provider = RecordingProvider()
    with pytest.raises(ValueError) as error:
        run_online(path, tmp_path / "run", provider)
    assert "PRIVATE_TRACE" not in str(error.value)
    assert not provider.calls
    assert not (tmp_path / "run").exists()


def test_checks_for_missing_context_and_conflicting_answer_labels(tmp_path):
    case = load_cases(DATASET)[0]
    case["required_context"] = ["Evidence that is absent from the request"]
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(case))
    with pytest.raises(ValueError, match="line 1"):
        load_cases(path)
    case["required_context"] = []
    case["answer_forbidden"] = case["answer_contains"]
    path.write_text(json.dumps(case))
    with pytest.raises(ValueError):
        load_cases(path)


def test_resume_skips_finished_and_ambiguous_attempts(tmp_path):
    provider = RecordingProvider(crash_on=2)
    with pytest.raises(KeyboardInterrupt):
        run_online(DATASET, tmp_path, provider, limit=2)
    assert len(provider.calls) == 2
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 1
    report = run_online(DATASET, tmp_path, provider, limit=2, resume=True)
    assert len(provider.calls) == 4
    assert report["call_count"] == 4
    assert report["uncertain_calls"] == report["failed_calls"] == 1
    assert report["paired_experiment_cost"] is None
    assert report["latency_change_ms"] is None
    assert len(list((tmp_path / "attempts").glob("*.started.json"))) == 4
    run_online(DATASET, tmp_path, provider, limit=2, resume=True)
    assert len(provider.calls) == 4


def test_completed_response_lost_before_journal_commit_is_never_retried(tmp_path, monkeypatch):
    original = RunStore.finish
    first = True

    def failing_finish(store, row):
        nonlocal first
        if first:
            first = False
            raise OSError("Simulated interrupted result write")
        original(store, row)

    monkeypatch.setattr(RunStore, "finish", failing_finish)
    provider = RecordingProvider()
    with pytest.raises(OSError):
        run_online(DATASET, tmp_path, provider, limit=2)
    report = run_online(DATASET, tmp_path, provider, limit=2, resume=True)
    assert len(provider.calls) == 4
    assert report["uncertain_calls"] == 1


def test_provider_failures_are_terminal_and_error_details_stay_private(tmp_path):
    provider = RecordingProvider(fail=True)
    run_online(DATASET, tmp_path, provider, limit=2)
    report = run_online(DATASET, tmp_path, provider, limit=2, resume=True)
    assert len(provider.calls) == 4
    assert report["failed_calls"] == 4
    assert report["uncertain_calls"] == 0
    assert "PRIVATE_PROVIDER_ERROR" not in (tmp_path / "calls.jsonl").read_text()


@pytest.mark.parametrize("change", ["model", "output", "seed", "mode", "code", "dataset"])
def test_resume_rejects_changed_experiment_before_any_new_call(tmp_path, monkeypatch, change):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_bytes(DATASET.read_bytes())
    output = tmp_path / "run"
    provider = RecordingProvider()
    run_online(dataset, output, provider, limit=2)
    kwargs = {}
    if change == "model":
        provider.model = "different"
    elif change == "output":
        provider.max_output_tokens = 1024
    elif change == "seed":
        kwargs["seed"] = 1
    elif change == "mode":
        kwargs["mode"] = "conservative"
    elif change == "code":
        monkeypatch.setattr("tokenflow.experiments.source_digest", lambda: "different")
    else:
        dataset.write_bytes(dataset.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Cannot resume"):
        run_online(dataset, output, provider, limit=2, resume=True, **kwargs)
    assert len(provider.calls) == 4


def test_corrupt_journal_fails_before_new_calls(tmp_path):
    provider = RecordingProvider(crash_on=2)
    with pytest.raises(KeyboardInterrupt):
        run_online(DATASET, tmp_path, provider, limit=3)
    completed = next((tmp_path / "attempts").glob("*.completed.json"))
    value = json.loads(completed.read_text())
    value["record"]["text"] = "corrupted"
    completed.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="checksum"):
        run_online(DATASET, tmp_path, provider, limit=3, resume=True)
    assert len(provider.calls) == 2


def test_changed_manifest_prices_fail_before_new_calls(tmp_path):
    provider = RecordingProvider(crash_on=2)
    with pytest.raises(KeyboardInterrupt):
        run_online(DATASET, tmp_path, provider, limit=2)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["prices"] = {"input_per_million": 999}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Cannot resume"):
        run_online(DATASET, tmp_path, provider, limit=2, resume=True)
    assert len(provider.calls) == 2


def test_resume_preserves_blinded_order_results_and_reviewer_edits(tmp_path):
    provider = RecordingProvider()
    first = run_online(DATASET, tmp_path, provider, limit=2)
    calls = (tmp_path / "calls.jsonl").read_bytes()
    key = (tmp_path / "review-key.jsonl").read_bytes()
    reviews = list(map(json.loads, (tmp_path / "blind-review.jsonl").read_text().splitlines()))
    reviews[0]["left_scores"] = {"correctness": 3}
    (tmp_path / "blind-review.jsonl").write_text("\n".join(map(json.dumps, reviews)))
    second = run_online(DATASET, tmp_path, provider, limit=2, resume=True)
    assert first == second
    assert len(provider.calls) == 4
    assert (tmp_path / "calls.jsonl").read_bytes() == calls
    assert (tmp_path / "review-key.jsonl").read_bytes() == key
    retained = json.loads((tmp_path / "blind-review.jsonl").read_text().splitlines()[0])
    assert retained["left_scores"] == {"correctness": 3}


def test_run_directory_lock_prevents_concurrent_paid_calls(tmp_path):
    provider = RecordingProvider()
    with run_lock(tmp_path):
        with pytest.raises(ValueError, match="Another process"):
            run_online(DATASET, tmp_path, provider, limit=1)
    assert not provider.calls


def test_os_releases_lock_after_process_exit(tmp_path):
    code = (
        "import os,sys; from pathlib import Path; from tokenflow.run_store import run_lock; "
        "guard=run_lock(Path(sys.argv[1])); guard.__enter__(); os._exit(0)"
    )
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True, timeout=10)
    with run_lock(tmp_path):
        pass


def test_response_model_drift_prevents_quality_comparison(tmp_path):
    class DriftingProvider(RecordingProvider):
        def generate(self, messages):
            result = super().generate(messages)
            return result.model_copy(update={"model": f"snapshot-{len(self.calls)}"})

    report = run_online(DATASET, tmp_path, DriftingProvider(), limit=2)
    assert not report["model_consistent"]
    assert report["cost_reduction"] is None
    with pytest.raises(ValueError, match="snapshots"):
        evaluate_reviews(tmp_path)


def test_preflight_cli_works_without_credentials(tmp_path):
    env = {k: value for k, value in os.environ.items() if k != "OPENAI_API_KEY"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tokenflow.cli",
            "plan-evaluation",
            str(DATASET),
            "--model",
            "not-an-access-claim",
            "--limit",
            "2",
            "--output",
            str(tmp_path / "plan.json"),
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["planned_provider_calls"] == 4
    assert json.loads((tmp_path / "plan.json").read_text())["actual_provider_calls"] == 0


def test_invalid_provider_output_cap_is_rejected_before_sdk_initialization():
    with pytest.raises(ValueError, match="max_output_tokens"):
        OpenAIProvider("test", max_output_tokens=0)
