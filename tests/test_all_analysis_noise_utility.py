from __future__ import annotations

import math

import matplotlib.figure
import matplotlib.patches
import pandas as pd
import pytest

from sports.analysis import load_data, run_table1, run_table2
from sports.analysis_noise_utility import evaluate_analysis_noise_utility
from sports.config import PREDICTORS


ANALYSES = (
    "table1",
    "table2",
    "figure1",
    "figure2",
    "correlation",
    "variance_analysis",
)

EXPECTED_DIFFERENCE_LABELS = {
    "table1": "Average Logistic Coefficient Difference",
    "table2": "Average Standardized Coefficient Difference",
    "figure1": "Average Figure 1 Coefficient Difference",
    "figure2": "Average Profile Difference",
    "correlation": "Average Correlation Difference",
    "variance_analysis": "Average Variance Difference",
}

EXPECTED_MEAN_TITLES = {
    "table1": "Mean Perturbed Table 1-style Logistic Regression",
    "table2": "Mean Perturbed Table 2-style Linear Regression",
    "figure1": "Mean Perturbed Figure 1",
    "figure2": "Mean Perturbed Figure 2",
    "correlation": "Mean Perturbed Correlation Result",
    "variance_analysis": "Mean Perturbed Variance Analysis",
}


def _legend_labels(figure: matplotlib.figure.Figure) -> set[str]:
    labels: set[str] = set()
    for axis in figure.axes:
        labels.update(label for label in axis.get_legend_handles_labels()[1] if label)
    return labels


def _table_rows(mean_result: dict) -> list[dict]:
    table = mean_result.get("table")
    if isinstance(table, pd.DataFrame):
        return table.to_dict(orient="records")
    if isinstance(table, dict):
        table = table.get("rows") or table.get("table")
    return list(table or mean_result.get("rows") or [])


def _semantic_flag(utility: dict, name: str):
    """Read a behavioral invariant without prescribing its metadata container."""
    if name in utility:
        return utility[name]
    for value in utility.values():
        if isinstance(value, dict) and name in value:
            return value[name]
    return None


@pytest.fixture(scope="module")
def noise_results():
    dataframe = load_data().copy()
    original = dataframe.copy(deep=True)
    results = {}
    for key in ANALYSES:
        options = {"variables": list(PREDICTORS), "repetitions": 3}
        if key == "figure2":
            options["max_athletes"] = 20
        if key == "figure1":
            options["variance_iterations"] = 100
        if key == "variance_analysis":
            options["iterations"] = 100
        results[key] = evaluate_analysis_noise_utility(key, dataframe, **options)
    pd.testing.assert_frame_equal(dataframe, original)
    return results


def test_all_dashboard_noise_evaluators_share_the_public_three_part_schema(noise_results):
    for key, utility in noise_results.items():
        assert utility["analysis_key"] == key
        assert utility["noise_distribution"] == "uniform"
        assert utility["noise_range"] == [-0.50, 0.50]
        assert utility["repetitions"] == 3
        assert utility["independent_repetitions"] is True
        assert utility["cumulative_noise"] is False
        assert utility["perturbed_columns"] == list(PREDICTORS)
        assert utility["outcome_perturbed"] is False
        assert utility["controls_perturbed"] is False

        mean_result = utility["mean_perturbed_result"]
        difference = utility["average_difference"]
        assert mean_result["title"] == EXPECTED_MEAN_TITLES[key]
        assert difference["label"] == EXPECTED_DIFFERENCE_LABELS[key]
        assert math.isfinite(float(difference["value"]))
        assert float(difference["value"]) >= 0.0
        assert difference["explanation"]
        assert isinstance(utility["stability_figure"], matplotlib.figure.Figure)


def test_each_stability_figure_compares_original_mean_and_one_sd(noise_results):
    for utility in noise_results.values():
        labels = _legend_labels(utility["stability_figure"])
        assert "Original" in labels
        assert "Mean perturbed" in labels
        assert any("1 SD" in label for label in labels)


def test_average_variance_difference_is_exclusive_to_variance(noise_results):
    labels = {
        key: utility["average_difference"]["label"]
        for key, utility in noise_results.items()
    }
    assert labels["variance_analysis"] == "Average Variance Difference"
    assert all(
        label != "Average Variance Difference"
        for key, label in labels.items()
        if key != "variance_analysis"
    )


def test_mean_perturbed_regression_tables_match_the_original_public_shape(noise_results):
    baseline_by_analysis = {
        "table1": run_table1(dataframe=load_data())["rows"],
        "table2": run_table2(dataframe=load_data())["rows"],
    }
    for key, baseline_rows in baseline_by_analysis.items():
        mean_rows = _table_rows(noise_results[key]["mean_perturbed_result"])
        assert mean_rows
        assert set(mean_rows[0]) >= set(baseline_rows[0])
        baseline_keys = {(row.get("model"), row.get("variable")) for row in baseline_rows}
        mean_keys = {(row.get("model"), row.get("variable")) for row in mean_rows}
        assert mean_keys == baseline_keys


def test_mean_perturbed_figure1_is_a_network_not_a_coefficient_bar_chart(noise_results):
    figure = noise_results["figure1"]["mean_perturbed_result"]["figure"]
    assert isinstance(figure, matplotlib.figure.Figure)
    assert any(
        isinstance(patch, matplotlib.patches.Circle)
        for axis in figure.axes
        for patch in axis.patches
    )


def test_mean_perturbed_correlation_matches_the_original_28_row_table_type(noise_results):
    rows = _table_rows(noise_results["correlation"]["mean_perturbed_result"])
    expected_pairs = [
        (first, second)
        for index, first in enumerate(PREDICTORS)
        for second in PREDICTORS[index + 1 :]
    ]
    assert len(rows) == 28
    assert [(row["variable_1"], row["variable_2"]) for row in rows] == expected_pairs
    assert all("correlation" in row for row in rows)


def test_figure2_keeps_one_anonymous_selection_across_noise_runs(noise_results):
    utility = noise_results["figure2"]
    assert _semantic_flag(utility, "same_selected_profiles") is True
    public_text = " ".join(
        text.get_text()
        for axis in utility["mean_perturbed_result"]["figure"].axes
        for text in axis.texts
    )
    assert "Athlete_" not in public_text
    assert "athlete_id" not in repr(utility["mean_perturbed_result"]).lower()


def test_variance_and_figure1_reuse_one_fixed_sampling_plan(noise_results):
    assert _semantic_flag(noise_results["variance_analysis"], "fixed_sampling_plan") is True
    assert _semantic_flag(noise_results["figure1"], "fixed_sampling_plan") is True
