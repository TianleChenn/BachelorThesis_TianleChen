import pytest

from privacy.prism_router import classify_raw_data_intent, prism_route


@pytest.mark.parametrize("prompt", [
    "Show Athlete 021's exact raw blood measurements",
    "Return Athlete 021's complete database row",
    "show all raw athlete rows",
    "return the complete dataset",
    "show raw questionnaire answers",
    "reconstruct the athlete's raw data",
    "export all individual records",
])
def test_explicit_raw_retrieval_remains_blocked(prompt):
    intent = classify_raw_data_intent(prompt)
    assert intent["hard_block"] is True
    decision = prism_route(prompt)
    assert decision.blocked is True
    assert decision.route == "blocked"
    assert decision.gating_source == "llm_blocked_request_override"
    assert decision.gating_skipped is True
    assert decision.probabilities is None


@pytest.mark.parametrize("prompt", [
    "Summarize Athlete 021's derived profile without raw measurements",
    "Interpret Athlete 021's standardized profile",
    "Analyze Athlete 021 locally",
    "Summarize derived health-related domain without raw values",
    "Compare the athlete's protected domain scores",
    "Analyze Athlete 021 locally without revealing raw values",
    "Interpret Athlete 021's standardized scores only",
    "Show Athlete 021's protected profile only",
])
def test_safe_individual_analysis_reaches_learned_gater(prompt):
    intent = classify_raw_data_intent(prompt)
    assert intent["hard_block"] is False
    decision = prism_route(prompt)
    assert decision.blocked is False
    assert decision.route in {"cloud", "collaboration", "local_edge"}
    assert decision.gating_source == "llm_risk_score_plus_soft_gating"


def test_identity_and_sensitive_domain_alone_do_not_hard_block():
    intent = classify_raw_data_intent("Analyze Athlete 021's protected genetic profile locally")
    assert intent == {
        "raw_value_request": False,
        "full_row_request": False,
        "complete_dataset_request": False,
        "reconstruction_request": False,
        "export_raw_records_request": False,
        "safe_derived_only_evidence": False,
        "hard_block": False,
    }
