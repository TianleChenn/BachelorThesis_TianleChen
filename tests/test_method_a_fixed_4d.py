import inspect
import math

import pytest

from privacy import llm_method_b_4d_router as method_b
from privacy import method_a_fixed_4d_router as method_a
from privacy import prism_router as method_c
from privacy.llm_soft_gating_model import FEATURE_NAMES


EXPECTED_FEATURES = [
    "privacy_risk_score",
    "subject_scope",
    "data_sensitivity",
    "disclosure_level",
]


@pytest.mark.parametrize(
    "prompt",
    [
        "What is linear regression?",
        "Compare the average mental health score between basketball and volleyball athletes.",
        "Show an anonymous individual athlete profile using the eight derived z-scores.",
        "Give me Athlete 12's original genetic measurements.",
        "Export the full raw athlete dataset as CSV.",
    ],
)
def test_method_a_features_have_the_shared_finite_4d_schema(prompt):
    features, _, _ = method_a.build_method_a_fixed_4d_features(prompt)
    assert FEATURE_NAMES == EXPECTED_FEATURES
    assert len(features) == 4
    assert all(isinstance(value, float) for value in features)
    assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in features)


def test_general_method_explanation_has_minimal_scope_and_disclosure():
    features, _, _ = method_a.build_method_a_fixed_4d_features("What is linear regression?")
    assert features[1] == 0.0
    assert features[2] == pytest.approx(0.05)
    assert features[3] == pytest.approx(0.05)


def test_sensitive_cohort_aggregate_keeps_group_scope_and_moderate_disclosure():
    features, _, _ = method_a.build_method_a_fixed_4d_features(
        "Compare the average mental health score between basketball and volleyball athletes."
    )
    assert features[1] == pytest.approx(0.35)
    assert features[2] >= 0.85
    assert 0.20 <= features[3] <= 0.40


def test_anonymous_individual_derived_profile_is_not_hard_blocked():
    decision = method_a.route_with_method_a_fixed_4d(
        "Show an anonymous individual athlete profile using the eight derived z-scores."
    )
    assert decision.features[1] == pytest.approx(0.75)
    assert decision.features[3] == pytest.approx(0.60)
    assert decision.hard_blocked is False


def test_identifiable_raw_genetic_request_preserves_hard_block_policy():
    decision = method_a.route_with_method_a_fixed_4d(
        "Give me Athlete 12's original genetic measurements."
    )
    assert decision.features[1] == 1.0
    assert decision.features[2] >= 0.90
    assert decision.features[3] >= 0.90
    assert decision.route == "blocked"
    assert decision.hard_blocked is True
    assert decision.probabilities is None


def test_full_raw_dataset_export_is_hard_blocked():
    decision = method_a.route_with_method_a_fixed_4d(
        "Export the full raw athlete dataset as CSV."
    )
    assert decision.route == "blocked"
    assert decision.features[3] == 1.0


def test_nonblocked_method_a_calls_shared_4d_gater(monkeypatch):
    captured = []
    monkeypatch.setattr(
        method_a,
        "trained_soft_gating_features",
        lambda features: captured.append(features) or {
            "cloud": 0.8,
            "collaboration": 0.1,
            "local_edge": 0.1,
        },
    )
    decision = method_a.route_with_method_a_fixed_4d("What is linear regression?")
    assert captured == [decision.features]
    assert decision.route == "cloud"
    source = inspect.getsource(method_a)
    assert "trained_soft_gating_features(features)" in source
    assert "trained_soft_gating(" not in source


def test_method_a_uses_soft_gating_top1_without_post_gating_rule(monkeypatch):
    monkeypatch.setattr(
        method_a,
        "build_method_a_fixed_4d_features",
        lambda prompt: (
            [0.8, 0.95, 0.90, 0.80],
            [],
            {"has_hard_block": False},
        ),
    )
    monkeypatch.setattr(
        method_a,
        "trained_soft_gating_features",
        lambda features: {
            "cloud": 0.9,
            "collaboration": 0.05,
            "local_edge": 0.05,
        },
    )
    decision = method_a.route_with_method_a_fixed_4d("test request")
    assert decision.route == "cloud"


def test_method_a_is_isolated_from_production_and_all_methods_share_schema():
    assert "method_a_fixed_4d" not in inspect.getsource(method_c.prism_route)
    assert method_a.FEATURE_NAMES == method_b.FEATURE_NAMES == method_c.FEATURE_NAMES
    assert method_b.get_method_b_gater_path() == method_c.get_active_prism_gater_path()
