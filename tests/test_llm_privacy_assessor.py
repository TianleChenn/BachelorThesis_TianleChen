import json

import pytest

from llm.model_clients import ModelCallResult
from privacy import llm_privacy_assessor as assessor


def response(**updates):
    payload = {
        "privacy_risk_score": 0.38,
        "subject_scope": 0.17,
        "data_sensitivity": 0.82,
        "disclosure_level": 0.46,
        "analysis_type": "descriptive_statistics",
        "blocked_request": False,
        "sensitive_categories": ["MEDICAL"],
        "explanation": "Aggregate sensitive analysis has moderate privacy risk.",
        "confidence": 0.91,
    }
    payload.update(updates)
    return json.dumps(payload)


def test_strict_json_parsing_and_metadata():
    result = assessor.parse_privacy_assessment_response(response(), actual_model="gpt-4.1-2025")
    assert result.risk_score == 0.38
    assert result.subject_scope == 0.17
    assert result.actual_model == "gpt-4.1-2025"


def test_obsolete_risk_level_field_is_ignored():
    result = assessor.parse_privacy_assessment_response(response(risk_level="anything"))
    assert "risk_level" not in result.to_dict()


@pytest.mark.parametrize("content", ["```json\n{}\n```", "not json", response(privacy_risk_score=2)])
def test_invalid_responses_are_rejected(content):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        assessor.parse_privacy_assessment_response(content)


def test_assessor_uses_mocked_model_and_cache(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(assessor, "CACHE_PATH", tmp_path / "cache.json")
    monkeypatch.setenv("PRIVACY_RISK_CACHE_ENABLED", "true")
    monkeypatch.setattr(assessor, "call_privacy_risk_model", lambda messages, **kwargs: (
        calls.append(messages) or ModelCallResult(response(), "gpt-4.1", "gpt-4.1", "openai_privacy_assessor", True, False, False, None)
    ))
    first = assessor.assess_privacy_with_llm("aggregate blood analysis")
    second = assessor.assess_privacy_with_llm(" aggregate   blood ANALYSIS ")
    assert first.to_dict() == second.to_dict()
    assert first.cache_used is False
    assert second.cache_used is True
    assert len(calls) == 1
    assert "Do not use predefined thresholds." in calls[0][0]["content"]
