from __future__ import annotations

import matplotlib.figure
import numpy as np

from sports.config import DOMAIN_ORDER, PREDICTORS
from sports.restricted_analysis_api import RestrictedAnalysisAPI


def _result():
    return RestrictedAnalysisAPI(subject_reference="Athlete_003").individual_profile(
        subject_token="CURRENT_SUBJECT",
        variables=list(DOMAIN_ORDER),
        reference_group="all",
        output_mode="standardized_profile",
    )


def test_anonymous_profile_includes_the_complete_noise_experiment():
    result = _result()
    utility = result["noise_utility"]

    assert utility["analysis_key"] == "individual_profile"
    assert utility["noise_distribution"] == "uniform"
    assert utility["noise_range"] == [-0.50, 0.50]
    assert utility["repetitions"] == 50
    assert utility["base_random_seed"] == 2026
    assert utility["independent_repetitions"] is True
    assert utility["cumulative_noise"] is False
    assert utility["perturbed_columns"] == list(PREDICTORS)
    assert utility["outcome_perturbed"] is False
    assert utility["controls_perturbed"] is False

    mean_result = utility["mean_perturbed_result"]
    assert mean_result["title"] == "Mean Perturbed Anonymous Athlete Profile"
    assert isinstance(mean_result["figure"], matplotlib.figure.Figure)
    assert [row["domain_key"] for row in mean_result["table"]] == list(DOMAIN_ORDER)

    metric = utility["average_difference"]
    assert metric["label"] == "Average Anonymous Profile Difference"
    assert metric["display_unit"] == "z-score units"
    assert np.isfinite(metric["value"])
    assert metric["value"] >= 0.0
    assert isinstance(utility["stability_figure"], matplotlib.figure.Figure)


def test_profile_noise_uses_one_fixed_anonymous_subject_without_exposing_it():
    result = _result()
    utility = result["noise_utility"]
    invariants = utility["selection_invariants"]

    assert invariants["same_anonymous_subject"] is True
    assert invariants["same_profile_order"] is True
    assert invariants["subject_identifier_exposed"] is False
    assert "Athlete_003" not in repr(result)
    assert "CURRENT_SUBJECT" not in repr(result)

    original = np.asarray(
        [row["z_score"] for row in result["table"]], dtype=float
    )
    baseline = np.asarray(
        utility["internal_summary"]["baseline_profile"], dtype=float
    )
    np.testing.assert_allclose(original, baseline)


def test_profile_stability_figure_has_original_mean_and_one_sd():
    utility = _result()["noise_utility"]
    axis = utility["stability_figure"].axes[0]
    labels = set(axis.get_legend_handles_labels()[1])
    assert "Original" in labels
    assert "Mean perturbed" in labels
    assert "Mean perturbed ± 1 SD" in labels
    assert axis.get_title() == "Anonymous Profile Stability After Numerical Perturbation"


def test_mean_perturbed_profile_figure_remains_anonymous():
    figure = _result()["noise_utility"]["mean_perturbed_result"]["figure"]
    axis = figure.axes[0]
    labels = set(axis.get_legend_handles_labels()[1])
    assert "Overall Mean (z = 0)" in labels
    assert "Elite Mean Profile" in labels
    assert "Mean Perturbed Anonymous Profile" in labels
    public_text = " ".join(
        [axis.get_title(), *[text.get_text() for text in axis.texts]]
    )
    assert "Athlete_" not in public_text

