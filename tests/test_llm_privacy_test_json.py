from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, call, patch

from privacy import prism_router
from privacy.llm_privacy_assessor import PrivacyAssessmentResult
from sports.service import _build_privacy_test


JSON_FIELDS = {
    "privacy_risk_score", "subject_scope", "data_sensitivity",
    "disclosure_level", "analysis_type",
    "blocked_request", "sensitive_categories", "explanation", "confidence",
}


def assessment(**updates):
    values = dict(privacy_risk_score=.55, subject_scope=.22,
        data_sensitivity=.81, disclosure_level=.47, analysis_type="descriptive_statistics",
        blocked_request=False, sensitive_categories=["MEDICAL"],
        explanation="Aggregate sensitive analysis.", confidence=.94,
        requested_model="gpt-4.1", actual_model="gpt-4.1", provider="mock",
        success=True, fallback_used=False, error=None)
    values.update(updates)
    return PrivacyAssessmentResult(**values)


def metadata(decision):
    return _build_privacy_test("Analyze aggregate blood data", prism_router.to_dict(decision))


def test_privacy_test_contains_validated_llm_json_and_exact_features(monkeypatch):
    monkeypatch.setattr(prism_router, "assess_privacy_with_llm", lambda prompt, **kwargs: assessment())
    monkeypatch.setattr(prism_router, "trained_soft_gating_features", lambda values: {
        "cloud": .1, "collaboration": .8, "local_edge": .1})
    monkeypatch.setattr(prism_router, "two_layer_ldp", lambda prompt, entities: ("protected", []))
    result = metadata(prism_router.prism_route("Analyze aggregate blood data"))
    assert {"input_prompt", "llm_generated_json", "gating_features",
        "gating_probabilities", "selected_route"} <= set(result)
    assert set(result["llm_generated_json"]) == JSON_FIELDS
    value = result["llm_generated_json"]
    assert result["gating_features"] == [float(value[name]) for name in [
        "privacy_risk_score", "subject_scope", "data_sensitivity", "disclosure_level"]]


def test_invalid_assessment_exposes_safe_fallback_json_only(monkeypatch):
    monkeypatch.setattr(prism_router, "assess_privacy_with_llm", lambda prompt: (_ for _ in ()).throw(ValueError("invalid raw model text")))
    result = metadata(prism_router.prism_route("request"))
    assert result["selected_route"] == "local_edge"
    assert result["fallback_used"] is True
    assert set(result["llm_generated_json"]) == JSON_FIELDS
    assert "raw_response" not in result


def test_llm_blocked_request_produces_blocked_route(monkeypatch):
    monkeypatch.setattr(prism_router, "assess_privacy_with_llm", lambda prompt, **kwargs: assessment(
        privacy_risk_score=.95, disclosure_level=.99,
        blocked_request=True))
    monkeypatch.setattr(prism_router, "trained_soft_gating_features", lambda values: {
        "cloud": .1, "collaboration": .1, "local_edge": .8})
    result = metadata(prism_router.prism_route("export raw data"))
    assert result["selected_route"] == "blocked"
    assert result["blocked_request"] is True


def test_frontend_keeps_complete_assessment_backend_but_hides_raw_json():
    source = Path("frontend.py").read_text(encoding="utf-8")
    section = source.split("def render_privacy_test", 1)[1].split("\ndef _format_llm_decimal", 1)[0]
    assert 'assessment = privacy_test.get("llm_generated_json") or {}' in section
    assert "st.json(" not in section
    assert "sensitive_categories" not in section
    assert "confidence" not in section
    for label in ("Privacy Risk", "Subject Scope", "Data Sensitivity", "Disclosure Level",
            "Blocked", "Assessment Explanation"):
        assert label in section
    assert "Analysis Type" not in section
    assert "Privacy Assessor Output" in section
    assert "Four-dimensional Soft Gating" in section
    assert "Privacy Assessment Dimensions" not in section
    assert "## Privacy Test – LLM Assessment" not in source
    assert "The JSON below is the structured privacy assessment returned by the LLM after schema validation." not in source
    assert "Entity Weights" not in source
    assert "Raw Risk Score Calculation" not in source
    assert "Bounded Risk Score" not in source
    assert "predefined weights" not in section.lower()


def test_clean_assessment_renderer_does_not_mutate_underlying_dictionary():
    import frontend
    assessment_value = {
        "privacy_risk_score": .13, "subject_scope": .08,
        "data_sensitivity": .22, "disclosure_level": .18,
        "analysis_type": "logistic_regression", "blocked_request": False,
        "sensitive_categories": ["MEDICAL"],
        "explanation": "Aggregate logistic regression request.", "confidence": .94,
    }
    privacy_test = {"assessment_success":True, "llm_generated_json":assessment_value,
        "gating_features":[.13,.08,.22,.18],
        "gating_probabilities":{"cloud":.7,"collaboration":.2,"local_edge":.1},
        "selected_route":"cloud"}
    original = deepcopy(privacy_test)
    output_columns = [Mock() for _ in range(4)]
    streamlit = Mock(); streamlit.columns.return_value = output_columns
    with patch.object(frontend, "st", streamlit):
        frontend.render_privacy_test(privacy_test)
    assert privacy_test == original
    assert [column.metric.call_args for column in output_columns] == [
        call("Cloud", "70.0%"), call("Collaboration", "20.0%"),
        call("Local Edge", "10.0%"), call("Selected Route", "Cloud")]
    assert streamlit.dataframe.call_count == 2
    assessor_table = streamlit.dataframe.call_args_list[0].args[0]
    assert assessor_table.to_dict(orient="records") == [{
        "Privacy Risk":"0.13", "Subject Scope":"0.08",
        "Data Sensitivity":"0.22", "Disclosure Level":"0.18", "Blocked":"No"}]
    streamlit.json.assert_not_called()


def test_blocked_assessment_stops_before_soft_gating_ui():
    import frontend
    privacy_test = {"assessment_success":True, "llm_generated_json":{
        "privacy_risk_score":.9,"subject_scope":.8,"data_sensitivity":.9,
        "disclosure_level":.9,"blocked_request":True,"explanation":"Blocked."},
        "gating_features":[.9,.8,.9,.9], "gating_probabilities":None,
        "selected_route":"blocked"}
    streamlit = Mock()
    with patch.object(frontend,"st",streamlit):
        frontend.render_privacy_test(privacy_test)
    streamlit.info.assert_called_once_with(
        "Soft Gating was not executed because the Privacy Assessor blocked this request.")
    assert streamlit.dataframe.call_count == 1
    assert list(streamlit.dataframe.call_args.args[0].columns) == [
        "Privacy Risk", "Subject Scope", "Data Sensitivity", "Disclosure Level", "Blocked"]
