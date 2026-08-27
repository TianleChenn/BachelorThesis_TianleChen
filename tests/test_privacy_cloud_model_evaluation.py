import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from llm.model_clients import ModelCallResult, call_cloud_privacy_evaluation_model
from privacy.llm_privacy_assessor import build_privacy_assessment_messages
from privacy.llm_soft_gating_model import DEFAULT_MODEL_PATH, FEATURE_NAMES
from scripts import evaluate_privacy_cloud_models as evaluation


def assessment_json(*, blocked=False):
    return json.dumps({
        "privacy_risk_score": .31,
        "subject_scope": .22,
        "data_sensitivity": .43,
        "disclosure_level": .18,
        "analysis_type": "correlation_analysis",
        "blocked_request": blocked,
        "sensitive_categories": ["aggregate"],
        "explanation": "This is an aggregate request.",
        "confidence": .94,
    })


def sample(route="cloud"):
    return {"id": "sample-1", "prompt": "Assess only this original request.",
        "ground_truth_route": route, "privacy_level": "SECRET_GOLD_LEVEL",
        "reason": "SECRET_GOLD_REASON", "annotation": "SECRET_ANNOTATION"}


def caller(content, captured=None):
    def invoke(model_key, messages, *, max_tokens):
        if captured is not None: captured[model_key] = messages
        return ModelCallResult(content, model_key, f"actual-{model_key}", "test", True,
            False, False, None, input_tokens=10, output_tokens=20, total_tokens=30,
            latency_seconds=.1)
    return invoke


def test_model_neutral_fence_normalization_never_repairs_content():
    plain = '{"privacy_risk_score": 0.1}'
    assert evaluation.normalize_privacy_evaluation_response(plain) == plain
    assert evaluation.normalize_privacy_evaluation_response(f"```json\n{plain}\n```") == plain
    assert evaluation.normalize_privacy_evaluation_response(f"```\n{plain}\n```") == plain
    incomplete = "```json\n{\n  \"privacy_risk_score\": 0.1"
    assert evaluation.normalize_privacy_evaluation_response(incomplete) == incomplete
    with pytest.raises(ValueError):
        from privacy.llm_privacy_assessor import parse_privacy_assessment_response
        parse_privacy_assessment_response(evaluation.normalize_privacy_evaluation_response(incomplete))


def test_registry_and_locked_dataset_are_exactly_three_by_sixty():
    metadata, samples = evaluation.load_evaluation_samples()
    assert evaluation.MODEL_KEYS == ("gpt4_1", "gemini", "claude")
    assert len(samples) == metadata["sample_count"] == 60
    assert Counter(row["ground_truth_route"] for row in samples) == evaluation.EXPECTED_DISTRIBUTION
    assert len(samples) * len(evaluation.MODEL_KEYS) == 180


def test_all_models_receive_identical_shared_messages_without_ground_truth():
    captured = {}; row = sample()
    for key in evaluation.MODEL_KEYS:
        evaluation.evaluate_pair(row, key, caller=caller(assessment_json(), captured),
            gater=lambda features: {"cloud": .8, "collaboration": .1, "local_edge": .1})
    assert captured["gpt4_1"] == captured["gemini"] == captured["claude"]
    assert captured["gpt4_1"] == build_privacy_assessment_messages(row["prompt"])
    serialized = repr(captured["gpt4_1"])
    for hidden in ("SECRET_GOLD_LEVEL", "SECRET_GOLD_REASON", "SECRET_ANNOTATION", "ground_truth_route"):
        assert hidden not in serialized


def test_all_models_use_same_normalizer_and_output_budget():
    captured = {}
    def invoke(model_key, messages, *, max_tokens):
        captured[model_key] = max_tokens
        return ModelCallResult(f"```json\n{assessment_json()}\n```", model_key, model_key,
            "test", True, False, False, None)
    for key in evaluation.MODEL_KEYS:
        row = evaluation.evaluate_pair(sample("cloud"), key, caller=invoke,
            gater=lambda _features: {"cloud": .8, "collaboration": .1, "local_edge": .1})
        assert row["parse_success"] is True
        assert row["plain_json_compliant"] is False
        assert row["normalized_response"] == assessment_json()
    assert captured == {key: evaluation.PRIVACY_EVALUATION_MAX_TOKENS for key in evaluation.MODEL_KEYS}
    assert set(captured.values()) == {1500}


def test_four_features_are_ordered_and_shared_frozen_gater_selects_top_one():
    seen = []
    def gater(features):
        seen.append(features)
        return {"cloud": .1, "collaboration": .7, "local_edge": .2}
    row = evaluation.evaluate_pair(sample("collaboration"), "gpt4_1",
        caller=caller(assessment_json()), gater=gater)
    assert FEATURE_NAMES == ["privacy_risk_score", "subject_scope", "data_sensitivity", "disclosure_level"]
    assert seen == [[.31, .22, .43, .18]]
    assert row["predicted_route"] == "collaboration"
    assert row["soft_gating_skipped"] is False
    assert row["gating_model_path"] == str(DEFAULT_MODEL_PATH.resolve())


def test_blocked_request_skips_soft_gating():
    def forbidden(_features): raise AssertionError("Soft Gating must be skipped")
    row = evaluation.evaluate_pair(sample("blocked"), "claude",
        caller=caller(assessment_json(blocked=True)), gater=forbidden)
    assert row["predicted_route"] == "blocked"
    assert row["soft_gating_skipped"] is True
    assert row["soft_gating_probabilities"] is None
    assert row["route_correct"] is True


def synthetic_rows(*, api_success=True):
    rows = []
    routes = [route for route, count in evaluation.EXPECTED_DISTRIBUTION.items() for _ in range(count)]
    for key in evaluation.MODEL_KEYS:
        for index, route in enumerate(routes):
            rows.append({"sample_id": f"{index}", "model_key": key,
                "ground_truth_route": route, "predicted_route": route if api_success else None,
                "route_correct": api_success, "api_call_success": api_success,
                "parse_success": api_success,
                "blocked_request": route == "blocked" if api_success else None,
                **{name: .5 if api_success else None for name in FEATURE_NAMES}})
    return rows


def test_formal_metrics_keep_fixed_denominators_when_a_response_fails():
    dataset, _ = evaluation.load_evaluation_samples()
    overall, per_route, _, report = evaluation.build_results(synthetic_rows(), dataset)
    assert report["status"] == "complete" and report["total_model_generations"] == 180
    assert all(row["Samples"] == 60 and row["Non-Blocked Samples"] == 50
        and row["Blocked Ground Truth Samples"] == 10 for row in overall)
    assert all(row["Exact Route Accuracy"] == 1 for row in overall)
    assert {row["total"] for row in per_route if row["Ground Truth Route"] == "cloud"} == {5}
    rows = synthetic_rows(); rows[0].update(api_call_success=False, parse_success=False,
        route_correct=False, predicted_route=None, blocked_request=None,
        **{name: None for name in FEATURE_NAMES})
    overall, per_route, features, report = evaluation.build_results(rows, dataset)
    assert report["status"] == "complete"
    assert overall[0]["Exact Route Correct"] == 59
    assert overall[0]["Exact Route Accuracy"] == 59 / 60
    assert overall[0]["API Failures"] == 1
    assert overall[0]["Valid Privacy Assessments"] == 59
    assert {row["Ground Truth Route"]: row["total"] for row in per_route
        if row["model_key"] == "gpt4_1"} == evaluation.EXPECTED_DISTRIBUTION
    cloud = next(row for row in features
        if row["model"] == "GPT-4.1" and row["ground_truth_route"] == "cloud")
    assert cloud["total_samples"] == 5 and cloud["valid_feature_samples"] == 4
    assert cloud["privacy_risk_score_mean"] == .5


def test_feature_summary_rejects_none_nonfinite_and_empty_vectors_without_crashing():
    dataset, _ = evaluation.load_evaluation_samples()
    rows = synthetic_rows()
    blocked = [row for row in rows if row["model_key"] == "gpt4_1"
        and row["ground_truth_route"] == "blocked"]
    for row in blocked:
        row.update(parse_success=False, route_correct=False, predicted_route=None,
            **{name: None for name in FEATURE_NAMES})
    blocked[0][FEATURE_NAMES[0]] = float("nan")
    _, _, features, _ = evaluation.build_results(rows, dataset)
    summary = next(row for row in features
        if row["model"] == "GPT-4.1" and row["ground_truth_route"] == "blocked")
    assert summary["total_samples"] == 10
    assert summary["valid_feature_samples"] == 0
    assert all(summary[f"{name}_mean"] is None for name in FEATURE_NAMES)


def test_resume_reuses_every_checkpoint_pair_including_recorded_failures():
    prior = synthetic_rows()
    prior[0].update(api_call_success=False, parse_success=False)
    completed_pairs = evaluation.checkpoint_completed_pairs(prior)
    assert len(completed_pairs) == 180
    assert (prior[0]["sample_id"], prior[0]["model_key"]) in completed_pairs


def test_checkpoint_version_and_production_checkpoint_are_unchanged():
    evaluation.validate_checkpoint_version([{"privacy_prompt_version": evaluation.PROMPT_VERSION}])
    with pytest.raises(RuntimeError, match="use --fresh"):
        evaluation.validate_checkpoint_version([{"privacy_prompt_version": "old"}])
    checkpoint = Path("artifacts/prism_soft_gater_4d_llm_hard.pt")
    assert checkpoint.resolve() == DEFAULT_MODEL_PATH.resolve()
    script = Path("scripts/evaluate_privacy_cloud_models.py").read_text(encoding="utf-8")
    for forbidden in ("torch.save", ".backward(", "optimizer.step", "LLMPrivacySoftGater("):
        assert forbidden not in script
    assert "assess_privacy_with_llm" not in script


def test_api_failure_cannot_be_correct():
    def failed(_key, _messages, *, max_tokens):
        return ModelCallResult(None, "model", None, "test", False, True, False, "offline")
    row = evaluation.evaluate_pair(sample("cloud"), "gemini", caller=failed)
    assert row["api_call_success"] is False
    assert row["parse_success"] is False
    assert row["route_correct"] is False
    assert row["predicted_route"] is None


def test_parse_failure_preserves_exact_raw_response_and_transport_success():
    raw = "```json\n{not valid plain JSON}\n```"
    row = evaluation.evaluate_pair(sample("cloud"), "gemini", caller=caller(raw))
    assert row["api_call_success"] is True
    assert row["parse_success"] is False
    assert row["raw_model_response"] == raw
    assert row["error"].startswith("JSONDecodeError:")


@pytest.mark.parametrize("model_key,model", [
    ("gpt4_1", "gpt-4.1"),
    ("gemini", "gemini-3.5-flash"),
    ("claude", "claude-sonnet-5"),
])
def test_privacy_evaluation_never_sends_api_response_format(model_key, model):
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=assessment_json(), incomplete_details=None, status="completed",
        model=model, usage=None,
    )
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=assessment_json()), finish_reason="stop")],
        model=model, usage=None,
    )
    runtime = {"model":model, "provider":"test", "api_key":"secret", "base_url":None}
    with patch("llm.model_clients.get_cloud_codegen_evaluation_runtime", return_value=runtime), \
         patch("openai.OpenAI", return_value=client):
        result = call_cloud_privacy_evaluation_model(
            model_key, build_privacy_assessment_messages("Return a privacy assessment."),
        )
    assert result.success
    called = client.chat.completions.create.call_args
    assert "response_format" not in called.kwargs
    assert "temperature" not in called.kwargs
    assert called.kwargs["max_tokens"] == 1500
    assert "max_output_tokens" not in called.kwargs


def test_existing_codegen_call_keeps_automatic_json_response_format():
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'), finish_reason="stop")],
        model="gemini", usage=None,
    )
    with patch("openai.OpenAI", return_value=client):
        from llm.model_clients import _call
        result = _call([{"role":"user", "content":"Return JSON only."}], "gemini", "test",
            "secret", "https://example.invalid/v1", 0.0, 100)
    assert result.success
    assert client.chat.completions.create.call_args.kwargs["response_format"] == {"type":"json_object"}
    assert client.chat.completions.create.call_args.kwargs["temperature"] == 0.0
