from pathlib import Path

import pytest

from privacy.prism_router import prism_route, to_dict
from sports.service import _build_privacy_test
from ui.cohort_prompts import build_dashboard_prompt


def _metadata(prompt):
    return _build_privacy_test(prompt, to_dict(prism_route(prompt)))


def test_privacy_test_uses_llm_assessment_and_four_features():
    metadata = _metadata(build_dashboard_prompt("table1", "table tennis athletes"))
    assert metadata["assessment_method"] == "LLM-based Privacy Assessment"
    assert 0 <= metadata["privacy_risk_score"] <= 1
    assert len(metadata["gating_features"]) == metadata["gating_input_dim"] == 4
    assert sum(metadata["gating_probabilities"].values()) == pytest.approx(1.0)
    assert "risk_terms" not in metadata
    assert "detected_entities" not in metadata


def test_individual_subject_is_not_exposed_as_detected_entity():
    metadata = _metadata("Generate a protected standardized individual athlete profile for Athlete_127.")
    assert "detected_entities" not in metadata
    assert metadata["selected_route"] in {"cloud", "collaboration", "local_edge", "blocked"}


def test_frontend_uses_semantic_assessment_ui():
    source = Path("frontend.py").read_text(encoding="utf-8")
    assert "## Privacy-aware Router Result" in source
    assert "### Privacy Assessor Output" in source
    assert "### Four-dimensional Soft Gating" in source
    assert "#### Soft Gating Input" in source
    assert "#### Soft Gating Output" in source
    assert "Privacy-test details are unavailable for this request." in source
    assert "Privacy Risk Level" not in source
    assert "Input Privacy Level" not in source
    assessment_section = source.split(
        "def show_prism_privacy_result", 1
    )[1].split("def render_privacy_test", 1)[0]
    privacy_test_section = source.split("def render_privacy_test", 1)[1]
    assert "Soft-Gating Probabilities" not in assessment_section
    assert "probability_columns" not in assessment_section
    assert "Four-Dimensional Soft-Gating Input" not in privacy_test_section
    assert "Privacy Assessment Dimensions" not in privacy_test_section
    assert "Active Soft-Gating Model" not in privacy_test_section
    assert "**Training Dataset:**" not in privacy_test_section
    assert "**Simulation Stage:**" not in privacy_test_section
    assert "**Training Stage:**" not in privacy_test_section
    assert "**Features Source:**" not in privacy_test_section
    assert "**Independent Evaluation:**" not in privacy_test_section
    assert "**Group Split:**" not in privacy_test_section
    assert 'privacy_test.get("gating_probabilities")' in privacy_test_section
    assert 'output_columns=st.columns(4)' in privacy_test_section
    assert 'metric("Assessment Confidence"' not in assessment_section
    assert '_format_probability_percent(probabilities.get("cloud",0.0))' in source
    assert '_format_probability_percent(probabilities.get("collaboration",0.0))' in source
    assert '_format_probability_percent(probabilities.get("local_edge",0.0))' in source
    assert 'return f"{bounded * 100:.1f}%"' in source
