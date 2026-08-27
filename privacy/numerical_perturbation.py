"""Local numerical perturbation used by the Table 2 utility experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sports.config import PREDICTORS

NOISE_DISTRIBUTION = "uniform"
NOISE_LOW = -0.50
NOISE_HIGH = 0.50
NOISE_REPETITIONS = 50
BASE_RANDOM_SEED = 2026
NOISE_EXPERIMENT_VERSION = "uniform-0.50-v2"


def create_uniformly_perturbed_dataframe(
    dataframe: pd.DataFrame,
    *,
    seed: int,
    lower_bound: float = NOISE_LOW,
    upper_bound: float = NOISE_HIGH,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return a deep copy with independent uniform noise on the eight domains."""
    columns = list(PREDICTORS) if columns is None else list(columns)
    if lower_bound >= upper_bound:
        raise ValueError("Noise lower bound must be less than its upper bound.")
    if any(name not in PREDICTORS for name in columns):
        raise ValueError("Only the eight standardized predictor columns may be perturbed.")
    missing = [name for name in columns if name not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing standardized predictor columns: {missing}")

    result = dataframe.copy(deep=True)
    numeric = result[columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        invalid = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"Predictor columns contain missing or non-numeric values: {invalid}")

    rng = np.random.default_rng(seed)
    noise = rng.uniform(lower_bound, upper_bound, size=numeric.shape)
    result.loc[:, columns] = numeric.to_numpy(dtype=float) + noise
    return result


def perturb_standardized_predictors(dataframe: pd.DataFrame, *, seed: int, low: float = NOISE_LOW, high: float = NOISE_HIGH) -> pd.DataFrame:
    """Backward-compatible internal name for the same full-precision operation."""
    return create_uniformly_perturbed_dataframe(
        dataframe, seed=seed, lower_bound=low, upper_bound=high
    )
