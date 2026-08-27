import pytest

from sports.config import PREDICTORS
from sports.restricted_analysis_api import RestrictedAnalysisAPI


def test_table2_and_correlation_api_results_contain_noise_utility():
    api = RestrictedAnalysisAPI()
    table2 = api.table2(variables=PREDICTORS)
    assert "noise_utility" in table2
    correlation = api.correlation(variables=PREDICTORS[:2], visualization=False)
    assert correlation["noise_utility"]["analysis_key"] == "correlation"


def test_statistically_insufficient_table2_cohort_is_not_privacy_blocked():
    api = RestrictedAnalysisAPI()
    api._filtered = lambda filters: api._frame().head(5)
    with pytest.raises(
        ValueError,
        match="All four Table 2 models are required for utility evaluation",
    ):
        api.table2(variables=PREDICTORS)
