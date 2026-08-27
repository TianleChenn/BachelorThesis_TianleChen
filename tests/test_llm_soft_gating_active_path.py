from privacy import prism_router
from privacy.llm_privacy_assessor import PrivacyAssessmentResult


def _assessment(**updates):
    values = dict(
        privacy_risk_score=.4, subject_scope=.2, data_sensitivity=.5,
        disclosure_level=.3, analysis_type="descriptive_statistics",
        blocked_request=False, sensitive_categories=[], explanation="Assessment.",
        confidence=.9, requested_model="test", actual_model="test", provider="test",
        success=True, fallback_used=False, error=None,
    )
    values.update(updates)
    return PrivacyAssessmentResult(**values)


def test_production_router_records_active_4d_model(monkeypatch):
    monkeypatch.setattr(prism_router, "assess_privacy_with_llm", lambda *a, **k: _assessment())
    monkeypatch.setattr(prism_router, "trained_soft_gating_features", lambda values: {
        "cloud": .7, "collaboration": .2, "local_edge": .1,
    })
    decision = prism_router.prism_route("aggregate request")
    assert decision.gating_model_name == "LLM-based 4D Soft Gating"
    assert decision.gating_input_dim == 4
    assert decision.gating_features == [.4, .2, .5, .3]
    assert decision.gating_model_path.endswith("prism_soft_gater_4d_llm_hard.pt")


def test_blocked_request_skips_soft_gating(monkeypatch):
    monkeypatch.setattr(prism_router, "assess_privacy_with_llm",
                        lambda *a, **k: _assessment(blocked_request=True))
    monkeypatch.setattr(prism_router, "trained_soft_gating_features",
                        lambda values: (_ for _ in ()).throw(AssertionError("must skip")))
    decision = prism_router.prism_route("blocked request")
    assert decision.route == "blocked"
    assert decision.gating_skipped is True
    assert decision.probabilities is None


def test_assessment_failure_stays_local_and_skips_gating(monkeypatch):
    monkeypatch.setattr(prism_router, "assess_privacy_with_llm",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    decision = prism_router.prism_route("request")
    assert decision.route == "local_edge"
    assert decision.gating_skipped is True
    assert decision.probabilities is None
