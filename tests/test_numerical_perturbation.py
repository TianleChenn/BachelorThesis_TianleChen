import pandas as pd
import numpy as np

from privacy.numerical_perturbation import NOISE_HIGH, NOISE_LOW, perturb_standardized_predictors
from sports.analysis import load_data
from sports.config import PREDICTORS


def test_perturbation_changes_only_standardized_predictors_and_is_reproducible():
    original = load_data().head(20)
    first = perturb_standardized_predictors(original, seed=2026)
    second = perturb_standardized_predictors(original, seed=2026)
    pd.testing.assert_frame_equal(first, second)
    assert not first[PREDICTORS].equals(original[PREDICTORS])
    differences=first[PREDICTORS].to_numpy(dtype=float)-original[PREDICTORS].to_numpy(dtype=float)
    assert NOISE_LOW == -0.50
    assert NOISE_HIGH == 0.50
    assert np.all(differences >= NOISE_LOW)
    assert np.all(differences <= NOISE_HIGH)
    untouched = [column for column in original.columns if column not in PREDICTORS]
    pd.testing.assert_frame_equal(first[untouched], original[untouched])
    pd.testing.assert_frame_equal(original, load_data().head(20))


def test_each_seed_starts_from_original_not_previous_result():
    original = load_data().head(20)
    first = perturb_standardized_predictors(original, seed=2026)
    second = perturb_standardized_predictors(original, seed=2027)
    assert not first[PREDICTORS].equals(second[PREDICTORS])
    assert first["expertise_value"].equals(second["expertise_value"])
    for column in ["age","sex","athlete_id"]:
        assert first[column].equals(original[column])
