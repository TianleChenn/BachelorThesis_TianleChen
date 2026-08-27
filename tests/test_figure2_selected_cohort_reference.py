import numpy as np
import pandas as pd

from sports.analysis import load_data
from sports.config import DOMAIN_ORDER
from sports.filters import apply_analysis_filters
from sports.restricted_analysis_api import RestrictedAnalysisAPI


def _expected(filters):
    full = load_data()
    numeric = full[DOMAIN_ORDER].apply(pd.to_numeric, errors="coerce")
    full_z = (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, 1)
    cohort = apply_analysis_filters(full, filters)
    return full_z.loc[cohort.index, DOMAIN_ORDER].mean(), len(cohort)


def _result(filters, maximum=20):
    return RestrictedAnalysisAPI().figure2(
        variables=list(DOMAIN_ORDER), filters=filters, max_athletes=maximum,
        reference_group="selected_cohort",
    )


def _reference_line(result):
    return next(line for line in result["figure"].axes[0].lines
                if line.get_label() == "Selected Cohort Mean Profile")


def test_figure2_reference_uses_complete_table_tennis_cohort_not_sample():
    filters = {"sport": "table tennis"}
    expected, cohort_size = _expected(filters)
    result = _result(filters, maximum=10)
    plotted = np.asarray(_reference_line(result).get_ydata(), dtype=float)

    assert cohort_size > result["shown_profiles"]
    assert result["cohort_size"] == cohort_size
    assert np.allclose(plotted, expected.to_numpy(dtype=float))
    displayed_lines = result["figure"].axes[0].lines[:result["shown_profiles"]]
    displayed_mean = np.mean([line.get_ydata() for line in displayed_lines], axis=0)
    assert not np.allclose(plotted, displayed_mean)
    assert "Table Tennis" in result["figure"].axes[0].get_title()


def test_female_and_all_athlete_references_use_their_exact_cohorts():
    female = _result({"sex": "female"}, maximum=20)
    female_expected, female_size = _expected({"sex": "female"})
    all_result = _result({}, maximum=20)
    all_expected, all_size = _expected({})

    assert female["cohort_size"] == female_size
    assert np.allclose(_reference_line(female).get_ydata(), female_expected)
    assert all_result["cohort_size"] == all_size
    assert np.allclose(_reference_line(all_result).get_ydata(), all_expected)
    assert np.allclose(all_expected, np.zeros(8), atol=1e-12)
    assert not np.allclose(female_expected, all_expected)


def test_reference_metadata_anonymity_precision_and_individual_distinction():
    result = _result({"sport": "table tennis"}, maximum=10)
    assert result["reference_type"] == "selected_cohort_mean"
    assert result["reference_label"] == "Selected Cohort Mean Profile"
    assert all(isinstance(value, float) and np.isfinite(value) for value in result["reference_profile"].values())
    assert any(value != round(value, 2) for value in result["reference_profile"].values())
    assert "Athlete_" not in repr({key: value for key, value in result.items() if key != "figure"})
    assert all(row["anonymous_profile_label"].startswith("Profile ") for row in result["table"])

    individual = RestrictedAnalysisAPI(subject_reference="Athlete_001").individual_profile(
        subject_token="CURRENT_SUBJECT", variables=list(DOMAIN_ORDER),
        reference_group="all", output_mode="standardized_profile",
    )
    assert individual["reference_type"] == "elite_mean"
    assert individual["reference_label"] == "Elite Mean Profile"
    selected_values = np.asarray(list(result["reference_profile"].values()))
    assert not np.allclose(selected_values, np.asarray(individual["elite_mean_profile"]))
