from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sports.service import _build_pipeline_model_audit, _select_model_decision


def _decision(tier):
    model = "gemini-3.5-flash" if tier == "cloud" else "Ministral-3-8B-Local"
    return SimpleNamespace(selected_tier=tier,
        selected_model="cloud_gemini" if tier == "cloud" else "local_ministral",
        execution_model=model)


def test_cloud_classifies_original_prompt_and_requires_a_tier():
    privacy = SimpleNamespace(route="cloud", cloud_prompt="original cloud prompt")
    with patch("sports.service.route_cloud_local", return_value=_decision("cloud")) as router:
        result = _select_model_decision("original cloud prompt", privacy)
    router.assert_called_once_with("original cloud prompt", "cloud",
                                   router_prompt_source="original_prompt_local_classifier")
    assert result.selected_tier == "cloud"


def test_collaboration_classifier_uses_original_prompt_locally():
    privacy = SimpleNamespace(route="collaboration", cloud_prompt="protected LDP prompt")
    with patch("sports.service.route_cloud_local", return_value=_decision("local")) as router:
        result = _select_model_decision("original private prompt", privacy)
    router.assert_called_once_with("original private prompt", "collaboration",
                                   router_prompt_source="original_prompt_local_classifier")
    assert result.selected_tier == "local"


@pytest.mark.parametrize("route,expected", [("local_edge", "local"), ("blocked", "none")])
def test_local_and_blocked_bypass_cost_aware_classifier(route, expected):
    privacy = SimpleNamespace(route=route, cloud_prompt=None)
    with patch("sports.service.route_cloud_local") as router:
        decision = _select_model_decision("private", privacy)
    router.assert_not_called()
    assert decision.selected_tier == expected
    assert decision.threshold is None
    assert decision.cloud_model_probability is None
    assert decision.router_name == ("privacy_forced_local" if route == "local_edge" else "not_applicable")


def test_cloud_and_collaboration_audit_are_explicit():
    for route, tier, model in (("cloud", "cloud", "gemini-3.5-flash"),
                               ("collaboration", "local", "Ministral-3-8B-Local")):
        pipeline = _build_pipeline_model_audit(
            {"route":route}, {"selected_model":f"{tier}_model", "selected_tier":tier,
                              "execution_model":model},
            {"actual_model":model, "provider":"openai_compatible"})
        assert pipeline["cost_aware_router_applicable"] is True
        assert pipeline["cost_aware_selected_tier"] == tier
        assert pipeline["cost_aware_selected_model"] == model
