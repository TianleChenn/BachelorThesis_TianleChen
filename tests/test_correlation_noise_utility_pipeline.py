from unittest.mock import patch

from privacy.numerical_perturbation import BASE_RANDOM_SEED, NOISE_HIGH, NOISE_LOW, NOISE_REPETITIONS
from sports.config import PREDICTORS
from sports.restricted_analysis_api import RestrictedAnalysisAPI


def test_correlation_attaches_existing_noise_utility_with_current_repetitions():
    utility = {"analysis_key":"correlation", "repetitions":NOISE_REPETITIONS}
    with patch("sports.restricted_analysis_api.evaluate_analysis_noise_utility", return_value=utility) as evaluate:
        result = RestrictedAnalysisAPI().correlation(
            variables=list(PREDICTORS), method="pearson", visualization=False)
    assert result["noise_utility"] is utility
    assert result["noise_utility"]["analysis_key"] == "correlation"
    assert result["noise_utility"]["repetitions"] == 50
    args, kwargs = evaluate.call_args
    assert args[0] == "correlation"
    assert kwargs["variables"] == list(PREDICTORS)
    assert kwargs["method"] == "pearson"


def test_correlation_never_returns_original_heatmap_even_when_visualization_is_true():
    utility = {"analysis_key":"correlation", "repetitions":NOISE_REPETITIONS}
    with patch("sports.restricted_analysis_api.evaluate_analysis_noise_utility", return_value=utility):
        result = RestrictedAnalysisAPI().correlation(
            variables=list(PREDICTORS), method="pearson", visualization=True)
    assert "figure" not in result
    assert result["table"]
    assert result["rows"] == result["table"]
    assert result["noise_utility"] is utility


def test_controlled_perturbation_constants_are_unchanged():
    assert NOISE_LOW == -0.50
    assert NOISE_HIGH == 0.50
    assert NOISE_REPETITIONS == 50
    assert BASE_RANDOM_SEED == 2026
