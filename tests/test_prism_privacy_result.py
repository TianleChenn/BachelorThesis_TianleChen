from unittest.mock import patch

from sports.service import handle_user_request


def _prediction(*args, **kwargs):
    return {"p_cloud": .2, "cloud_model_probability":.2, "threshold": .6,
            "selected_tier": "local", "selected_model":"local_ministral",
            "execution_model":"Ministral-3-8B-Local", "router_name":"athlete_cloud_local_router",
            "model_version": "test"}


def test_prism_result_uses_actual_cloud_prompt():
    prompt="Explain blood micronutrients using aggregate regression."
    with patch("llm.athlete_cloud_local_router.AthleteCloudLocalRouter.predict", side_effect=_prediction):
        response = handle_user_request(prompt, use_openai=False)
    result = response["prism_privacy_result"]
    assert "cloud_prompt" not in response["privacy_decision"]
    assert result["prompt_after_prism"] is not None
    assert result["prompt_after_prism"] != prompt


def test_gating_probabilities_are_valid():
    with patch("llm.athlete_cloud_local_router.AthleteCloudLocalRouter.predict", side_effect=_prediction):
        result = handle_user_request(
            "What analyses are available?", use_openai=False
        )["prism_privacy_result"]
    gating = result["soft_gating"]
    assert all(0.0 <= value <= 1.0 for value in gating.values())
    assert abs(sum(gating.values()) - 1.0) <= 0.01


def test_local_route_has_no_cloud_prompt():
    result = handle_user_request(
        "Analyze the exact protected profile for Athlete_003.",
        use_openai=False,
    )["prism_privacy_result"]
    assert result["route"] in {"local_edge", "blocked"}
    assert result["prompt_after_prism"] is None


def test_semantic_privacy_score_is_bounded():
    result = handle_user_request(
        "Explain my blood and genetic information.", use_openai=False
    )["prism_privacy_result"]
    assert 0.0 <= result["privacy_risk_score"] <= 1.0
    assert result["privacy_assessment_method"] == "llm_semantic_assessment"
