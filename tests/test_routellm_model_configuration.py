from pathlib import Path

from llm.model_config import (
    LOCAL_EDGE_GENERATOR_MODEL,
    ROUTELLM_STRONG_MODEL,
    ROUTELLM_WEAK_MODEL,
)
from privacy.routellm_router import route_model
from scripts.athlete_router_evaluation_common import get_athlete_router_evaluation_models


class _AthleteRouter:
    def __init__(self, tier):
        self.tier = tier

    def predict(self, *args, **kwargs):
        return {
            "p_strong": .8 if self.tier == "strong" else .2,
            "threshold": .6,
            "selected_tier": self.tier,
            "model_version": "test",
        }


def test_canonical_route_models_are_used_by_backend_and_environment():
    assert ROUTELLM_STRONG_MODEL == "gpt-4.1"
    assert ROUTELLM_WEAK_MODEL == "Ministral-3-8B"
    env = Path(".env").read_text(encoding="utf-8")
    assert f"LLM_STRONG_MODEL={ROUTELLM_STRONG_MODEL}" in env
    configured = get_athlete_router_evaluation_models()
    assert configured.strong_model == ROUTELLM_STRONG_MODEL
    assert configured.weak_model == ROUTELLM_WEAK_MODEL
    assert configured.weak_model_id == "ministral-8b-latest"
    assert configured.judge_model == "gpt-4o-mini"


def test_route_decision_exposes_selected_tier_and_execution_model():
    strong = route_model("safe", "cloud", privacy_prompt="safe", athlete_router=_AthleteRouter("strong"))
    weak = route_model("safe", "cloud", privacy_prompt="safe", athlete_router=_AthleteRouter("weak"))
    assert strong.selected_tier == "strong"
    assert strong.execution_model == strong.execution_strong_model == ROUTELLM_STRONG_MODEL
    assert weak.selected_tier == "weak"
    assert weak.execution_model == weak.execution_weak_model == ROUTELLM_WEAK_MODEL


def test_local_model_is_not_the_weak_cloud_model():
    local = route_model("private", "local_edge")
    assert local.selected_tier == "none"
    assert local.execution_model == LOCAL_EDGE_GENERATOR_MODEL
    assert local.execution_weak_model == ROUTELLM_WEAK_MODEL
    assert local.execution_weak_model != LOCAL_EDGE_GENERATOR_MODEL
