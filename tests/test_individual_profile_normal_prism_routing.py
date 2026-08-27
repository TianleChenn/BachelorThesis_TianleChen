from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from llm.analysis_request_contracts import build_request_contract,render_request_contract
from llm.code_generator import CodeGenerationResult
from privacy.prism_router import detect_entities, prism_route, sensitivity_profile, two_layer_ldp
from sports.service import handle_user_request


PROMPT = "Generate a protected standardized individual athlete profile for Athlete_127."
FULL_PROMPT = (
    "Generate a protected standardized individual athlete profile for Athlete_127. "
    "Include the eight standardized domains, strongest and weakest domains, the profile "
    "figure, and whether the profile matches the paper's three-domain group-level pattern. "
    "Do not expose the athlete identifier or raw measurements in the final result. "
    "Do not produce individual status or future performance forecasts."
)


def test_prompt_contains_real_id_without_preselected_route_language():
    assert "Athlete_127" in PROMPT
    assert "anonymous" not in PROMPT.lower()
    assert "locally" not in PROMPT.lower()
    assert not any(term in PROMPT.lower() for term in ("cloud", "collaboration", "local edge"))


@pytest.mark.parametrize("athlete_id", ["Athlete_001", "Athlete_127", "Athlete_300", "athlete_127", "ATHLETE_127", "Athlete-1024"])
def test_athlete_id_variants_are_detected_and_protected(athlete_id):
    entities = detect_entities(f"Generate an individual profile for {athlete_id}.")
    assert any(entity.category == "ATHLETE_ID" and entity.protect for entity in entities)


def test_individual_profile_flags_are_detected_without_aggregate_override():
    _, _, _, flags = sensitivity_profile(PROMPT)
    assert flags["individual_analysis_present"] is True
    assert flags["aggregate_analysis_present"] is False
    _, _, _, full_flags = sensitivity_profile(FULL_PROMPT)
    assert full_flags["individual_analysis_present"] is True
    assert full_flags["aggregate_analysis_present"] is False
    assert full_flags["has_raw_request"] is False
    assert full_flags["has_hard_block"] is False


def test_soft_gating_selects_route_and_branch_metadata_is_consistent():
    decision = prism_route(PROMPT)
    assert set(decision.probabilities) == {"cloud", "collaboration", "local_edge"}
    assert decision.route in {"cloud", "collaboration", "local_edge", "blocked"}
    if decision.route == "collaboration":
        assert decision.cloud_prompt is not None
        assert decision.privacy_method == "llm_privacy_assessment"
        assert decision.cloud_payload_type == "two_layer_ldp_perturbed_prompt"
        assert decision.ldp_audit
    elif decision.route == "local_edge":
        assert decision.cloud_prompt is None
        assert decision.privacy_method == "llm_privacy_assessment"
        assert decision.ldp_audit == []
    elif decision.route == "cloud":
        assert decision.cloud_prompt == PROMPT
        assert decision.privacy_method == "llm_privacy_assessment"
        assert decision.cloud_payload_type == "original_prompt"
        assert decision.ldp_audit == []


def test_ldp_changes_only_protected_entities_and_preserves_task_semantics():
    entities = detect_entities(PROMPT)
    perturbed, audit = two_layer_ldp(PROMPT, entities, seed=2024)
    assert "standardized individual athlete profile" in perturbed.lower()
    assert any(row["entity_category"] == "ATHLETE_ID" for row in audit)
    assert not any(row["entity_category"] == "ANALYSIS_ACTION" for row in audit)
    analysis_entity = next(entity for entity in entities if entity.category == "ANALYSIS_ACTION")
    assert analysis_entity.protect is False


def test_service_uses_current_subject_and_returns_anonymous_eight_domain_result():
    generated = CodeGenerationResult(
        code=render_request_contract(build_request_contract("individual_profile")),
        code_source="test_restricted_contract",
        action="individual_profile",
        group=None,
        explanation="Test trusted subject execution.",
        requested_analysis="individual_profile",
        structure_validation_passed=True,
        request_match_passed=True,
        generator_target="local",
    )
    with patch("sports.service.generate_code", return_value=generated):
        response = handle_user_request(
            PROMPT,
            use_openai=False,
            requested_analysis="individual_profile",
            private_local_context={"CURRENT_SUBJECT": "Athlete_127"},
        )
    serialized = json.dumps(response.get("result"), default=str)
    assert response["allowed"] is True
    assert "Athlete_127" not in serialized
    assert "Athlete_" not in serialized
    assert len(response["result"]["table"]) == 8
    assert response["result"]["analysis"] == "Anonymous Athlete Profile"
    assert response["privacy_test"]["input_prompt"] == PROMPT
    if response["llm_result"]["privacy_route"] == "local_edge":
        assert response["model_decision"]["selected_tier"] == "local"
        assert response["pipeline_audit"]["local_edge_generator_used"] is True


def test_frontend_marks_cost_router_not_applicable_for_local_edge():
    source = open("frontend.py", encoding="utf-8").read()
    assert '_render_full_value_card("Router Status", "Not Applicable")' in source
    assert "Cost-aware Cloud/Local Router is " in source
    assert "bypassed completely and no cloud model is called" in source
    assert "evaluation_only_router_prediction" not in source
