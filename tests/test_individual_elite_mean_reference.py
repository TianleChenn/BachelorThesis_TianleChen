import numpy as np

from sports.config import DOMAIN_ORDER
from sports.restricted_analysis_api import RestrictedAnalysisAPI


def _profile(subject):
    return RestrictedAnalysisAPI(subject_reference=subject).individual_profile(
        subject_token="CURRENT_SUBJECT",
        variables=list(DOMAIN_ORDER),
        reference_group="all",
        output_mode="standardized_profile",
    )


def test_individual_figure_uses_shared_nonflat_elite_mean_reference():
    first = _profile("Athlete_001")
    second = _profile("Athlete_002")

    first_lines = {line.get_label(): line for line in first["figure"].axes[0].lines}
    second_lines = {line.get_label(): line for line in second["figure"].axes[0].lines}
    first_reference = np.asarray(first_lines["Elite Mean Profile"].get_ydata(), dtype=float)
    second_reference = np.asarray(second_lines["Elite Mean Profile"].get_ydata(), dtype=float)
    first_profile = np.asarray(first_lines["Anonymous Profile"].get_ydata(), dtype=float)
    second_profile = np.asarray(second_lines["Anonymous Profile"].get_ydata(), dtype=float)

    assert len(first_reference) == 8
    assert not np.allclose(first_reference, np.zeros(8))
    assert np.ptp(first_reference) > 0
    assert np.allclose(first_reference, second_reference)
    assert not np.allclose(first_profile, second_profile)
    assert first["elite_reference_count"] == second["elite_reference_count"]
    assert first["elite_mean_profile"] == second["elite_mean_profile"]

    axis = first["figure"].axes[0]
    legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]
    assert legend_labels == ["Overall Mean (z = 0)", "Elite Mean Profile", "Anonymous Profile"]
    zero_line = first_lines["Overall Mean (z = 0)"]
    assert zero_line.get_linestyle() == "-"
    assert np.allclose(zero_line.get_ydata(), [0.0, 0.0])
    assert axis.get_title() == "Individual Standardized Profile\nAnonymous Profile vs Elite Mean Profile"
