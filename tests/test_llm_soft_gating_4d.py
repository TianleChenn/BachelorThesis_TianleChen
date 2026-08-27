from types import SimpleNamespace

import pytest
import torch

from privacy.llm_soft_gating_model import (
    FEATURE_NAMES, INPUT_DIM, LLMPrivacySoftGater, build_llm_gating_features,
    predict_llm_privacy_route, softmax_probabilities,
)


def test_llm_soft_gater_has_exact_four_dimensional_schema():
    assert INPUT_DIM == 4
    assert FEATURE_NAMES == [
        "privacy_risk_score", "subject_scope", "data_sensitivity", "disclosure_level"
    ]
    assert LLMPrivacySoftGater()(torch.zeros(2, 4)).shape == (2, 3)


def test_features_remain_continuous_and_validated():
    assessment = SimpleNamespace(
        privacy_risk_score=.5372, subject_scope=.3184,
        data_sensitivity=.7921, disclosure_level=.4768,
    )
    assert build_llm_gating_features(assessment) == [.5372, .3184, .7921, .4768]
    assessment.disclosure_level = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_llm_gating_features(assessment)


def test_softmax_and_prediction_cover_three_routes():
    model = LLMPrivacySoftGater()
    route, probabilities = predict_llm_privacy_route([.1, .2, .3, .4], model)
    assert route in {"cloud", "collaboration", "local_edge"}
    assert set(probabilities) == {"cloud", "collaboration", "local_edge"}
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert sum(softmax_probabilities(torch.tensor([1.0, 2.0, 3.0])).values()) == pytest.approx(1.0)


def test_model_can_save_and_reload(tmp_path):
    from privacy.llm_soft_gating_model import load_llm_privacy_soft_gater
    path = tmp_path / "model.pt"
    model = LLMPrivacySoftGater()
    torch.save({"model_state_dict": model.state_dict(), "input_dim": 4,
                "feature_names": FEATURE_NAMES}, path)
    assert load_llm_privacy_soft_gater(path).linear.in_features == 4
