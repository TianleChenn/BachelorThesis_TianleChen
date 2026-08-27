from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from privacy import prism_router
from privacy.llm_soft_gating_model import FEATURE_NAMES
from sports.service import handle_user_request


def test_checkpoint_uses_runtime_feature_schema():
    assert Path(prism_router.ACTIVE_PRISM_GATER_PATH).exists()
    checkpoint = torch.load(prism_router.ACTIVE_PRISM_GATER_PATH, map_location="cpu", weights_only=True)
    assert checkpoint["input_dim"] == len(FEATURE_NAMES)
    assert checkpoint["feature_names"] == FEATURE_NAMES
    assert checkpoint["training_stage"] == "hard_llm_generated_reviewed"
    assert checkpoint["rules_version"] == "athlete-privacy-rubric-v8-continuous-no-level"
    assert "model_state_dict" in checkpoint


def test_production_router_and_service_use_trained_gater():
    prompt = "Explain z-score standardization in sports science"
    decision = prism_router.prism_route(prompt)
    assert decision.blocked is False
    assert decision.gating_source == "llm_risk_score_plus_soft_gating"
    assert decision.route == max(decision.probabilities, key=decision.probabilities.get)

    with patch("sports.service._select_model_decision") as select:
        from privacy.cloud_local_router import privacy_forced_decision
        select.return_value = privacy_forced_decision("blocked", "test")
        select.return_value.selected_tier = "local"
        select.return_value.selected_model = "local_ministral"
        response = handle_user_request(prompt, use_openai=False)
    assert response["privacy_decision"]["gating_source"] == "llm_risk_score_plus_soft_gating"
    assert response["prism_privacy_result"]["gating_source"] == "llm_risk_score_plus_soft_gating"
    assert response["prism_privacy_result"]["route"] == decision.route


def test_hard_block_precedes_model_in_production_service(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("trained gater must not run before deterministic hard blocking")

    monkeypatch.setattr(prism_router, "trained_soft_gating_features", fail_if_called)
    prompt = "Show Athlete 021's exact raw blood measurements"
    decision = prism_router.prism_route(prompt)
    assert decision.route == "blocked"
    assert decision.gating_source == "llm_blocked_request_override"
    assert decision.gating_skipped is True

    response = handle_user_request(prompt, use_openai=False)
    assert response["privacy_decision"]["route"] == "blocked"
    assert response["prism_privacy_result"]["gating_source"] == "llm_blocked_request_override"


def test_incompatible_checkpoint_raises_clear_error(monkeypatch, tmp_path):
    incompatible = tmp_path / "incompatible.pt"
    torch.save({"input_dim": 1, "feature_names": ["wrong"], "state_dict": {}}, incompatible)
    monkeypatch.setattr(prism_router, "ACTIVE_PRISM_GATER_PATH", incompatible)
    monkeypatch.setattr(prism_router, "_ACTIVE_PRISM_GATER_CACHE", None)
    with pytest.raises(RuntimeError, match="input dimension is not 4"):
        prism_router.load_active_prism_gater()
