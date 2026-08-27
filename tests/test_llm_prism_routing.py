from privacy import prism_router
from privacy.llm_privacy_assessor import PrivacyAssessmentResult


def assessment(**updates):
    values = dict(privacy_risk_score=.4, subject_scope=.18,
        data_sensitivity=.79, disclosure_level=.43, analysis_type="descriptive_statistics",
        blocked_request=False, sensitive_categories=["MEDICAL"], explanation="Moderate aggregate risk.",
        confidence=.9, requested_model="gpt-4.1", actual_model="gpt-4.1", provider="mock",
        success=True, fallback_used=False, error=None)
    values.update(updates)
    return PrivacyAssessmentResult(**values)


def test_llm_features_feed_existing_soft_gater(monkeypatch):
    captured = []
    monkeypatch.setattr(prism_router, "assess_privacy_with_llm", lambda prompt, **kwargs: assessment())
    monkeypatch.setattr(prism_router, "trained_soft_gating_features", lambda values: (
        captured.append(values) or {"cloud": .1, "collaboration": .8, "local_edge": .1}
    ))
    monkeypatch.setattr(prism_router, "two_layer_ldp", lambda prompt, entities: ("protected", []))
    decision = prism_router.prism_route("aggregate blood analysis")
    assert captured == [[.4, .18, .79, .43]]
    assert decision.route == "collaboration"
    assert decision.cloud_prompt == "protected"
    assert decision.gating_source == "llm_risk_score_plus_soft_gating"


def test_soft_gating_top1_is_not_overridden_by_privacy_thresholds(monkeypatch):
    monkeypatch.setattr(prism_router, "assess_privacy_with_llm", lambda prompt, **kwargs: assessment(
        privacy_risk_score=.8, subject_scope=.95,
        data_sensitivity=.9, disclosure_level=.8, blocked_request=False))
    monkeypatch.setattr(prism_router, "trained_soft_gating_features", lambda values: {"cloud": .9, "collaboration": .05, "local_edge": .05})
    decision = prism_router.prism_route("request")
    assert decision.route == "cloud"
    assert decision.probabilities["cloud"] == .9
    assert decision.cloud_prompt == "request"


def test_assessor_failure_fails_safe_without_blocking(monkeypatch):
    monkeypatch.setattr(prism_router, "assess_privacy_with_llm", lambda prompt: (_ for _ in ()).throw(RuntimeError("offline")))
    decision = prism_router.prism_route("request")
    assert decision.route == "local_edge"
    assert decision.blocked is False
    assert decision.risk_score is None
    assert decision.gating_source == "safe_local_fallback"
    assert decision.gating_skipped is True
    assert decision.gating_features is None
    assert decision.probabilities is None
