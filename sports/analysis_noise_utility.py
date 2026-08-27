"""Controlled numerical perturbation for dashboard and anonymous-profile analyses."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from privacy.numerical_perturbation import (
    BASE_RANDOM_SEED,
    NOISE_DISTRIBUTION,
    NOISE_HIGH,
    NOISE_LOW,
    NOISE_REPETITIONS,
    create_uniformly_perturbed_dataframe,
)
from .analysis import run_table1
from .config import DISPLAY_NAMES, DOMAIN_ORDER, PREDICTORS
from .filters import apply_analysis_filters
from .noise_figures import (
    create_correlation_stability_figure,
    create_figure1_stability_figure,
    create_figure2_stability_figure,
    create_individual_profile_stability_figure,
    create_mean_perturbed_correlation_figure,
    create_mean_perturbed_figure1,
    create_mean_perturbed_figure2,
    create_mean_perturbed_table1_figure,
    create_mean_perturbed_variance_figure,
    create_table1_stability_figure,
    create_variance_stability_figure,
)


EXPERIMENT_VERSION = "uniform-0.50-all-analyses-v2"
ALLOWED = {
    "table1",
    "table2",
    "figure1",
    "figure2",
    "correlation",
    "variance_analysis",
    "individual_profile",
}

_ANALYSIS_TITLES = {
    "table1": "Logistic Regression Noise Stability Evaluation",
    "table2": "Linear Regression Noise Stability Evaluation",
    "figure1": "Figure 1 Noise Stability Evaluation",
    "figure2": "Figure 2 Noise Stability Evaluation",
    "correlation": "Correlation Noise Stability Evaluation",
    "variance_analysis": "Variance Noise Stability Evaluation",
    "individual_profile": "Anonymous Profile Noise Stability Evaluation",
}


def _noise_payload(
    *,
    analysis_key: str,
    mean_perturbed_result: dict,
    average_difference: dict,
    stability_figure,
    repetitions: int,
    successful_runs: int,
    base_seed: int,
    technical_details: dict | None = None,
    internal_summary: dict | None = None,
    legacy_noisy_figure=None,
    selection_invariants: dict | None = None,
    sampling_invariants: dict | None = None,
) -> dict:
    """Build the common public schema and preserve legacy keys for old sessions."""
    mean_result = {
        "title": str(mean_perturbed_result["title"]),
        "table": mean_perturbed_result.get("table"),
        "figure": mean_perturbed_result.get("figure"),
    }
    metric = {
        "label": str(average_difference["label"]),
        "value": float(average_difference["value"]),
        "display_unit": str(average_difference.get("display_unit") or ""),
        "explanation": str(average_difference["explanation"]),
    }
    legacy_figure = mean_result["figure"] or legacy_noisy_figure
    if legacy_figure is None:
        raise ValueError("Every noise utility requires a mean-perturbed figure payload.")

    return {
        "analysis": _ANALYSIS_TITLES[analysis_key],
        "analysis_key": analysis_key,
        "noise_distribution": NOISE_DISTRIBUTION,
        "noise_range": [NOISE_LOW, NOISE_HIGH],
        "repetitions": int(repetitions),
        "attempted_runs": int(repetitions),
        "successful_runs": int(successful_runs),
        "failed_runs": int(repetitions - successful_runs),
        "independent_repetitions": True,
        "cumulative_noise": False,
        "base_random_seed": int(base_seed),
        "experiment_version": EXPERIMENT_VERSION,
        "perturbed_columns": list(PREDICTORS),
        "outcome_perturbed": False,
        "controls_perturbed": False,
        "raw_rows_exposed": False,
        "noise_matrix_exposed": False,
        "mean_perturbed_result": mean_result,
        "average_difference": metric,
        "stability_figure": stability_figure,
        # Compatibility fields are deliberately retained so existing stored
        # responses and older callers keep working. The new renderer consumes
        # the three fields immediately above.
        "primary_metric": {
            "label": metric["label"],
            "value": metric["value"],
            "baseline_value": None,
            "mean_noisy_value": None,
            "display_kind": (
                "z_score_units" if metric["display_unit"] else "four_decimals"
            ),
            "explanation": metric["explanation"],
        },
        "noisy_result_title": mean_result["title"],
        "noisy_result_caption": (
            "Mean result from independent perturbations using "
            "Uniform(-0.50, 0.50)."
        ),
        "noisy_result_figure": legacy_figure,
        "stability_title": stability_figure.axes[0].get_title(),
        "stability_caption": (
            "Original, mean perturbed, and mean perturbed ± 1 SD across "
            f"{int(repetitions)} independent runs."
        ),
        "technical_details": dict(technical_details or {}),
        "internal_summary": dict(internal_summary or {}),
        "selection_invariants": dict(selection_invariants or {}),
        "sampling_invariants": dict(sampling_invariants or {}),
    }


def _perturbed_runs(
    dataframe: pd.DataFrame,
    repetitions: int,
    low: float,
    high: float,
    seed: int,
) -> Iterable[pd.DataFrame]:
    """Always perturb a fresh copy of the same original dataframe."""
    for index in range(repetitions):
        yield create_uniformly_perturbed_dataframe(
            dataframe,
            seed=seed + index,
            lower_bound=low,
            upper_bound=high,
        )


def _table1_vector(result: dict, field: str, model_name: str) -> np.ndarray:
    primary_rows = {
        row["variable"]: row
        for row in result["rows"]
        if row.get("model") == model_name
    }
    return np.asarray(
        [primary_rows[DISPLAY_NAMES[predictor]][field] for predictor in PREDICTORS],
        dtype=float,
    )


def _mean_table1_rows(results: list[dict]) -> list[dict]:
    """Return a Table 1-shaped mean table across successful noisy fits."""
    numeric_fields = (
        "B",
        "SE",
        "Wald",
        "p",
        "odds_ratio",
        "or_ci_low",
        "or_ci_high",
    )
    indexed_results = [
        {
            (row.get("model"), row.get("variable")): row
            for row in result.get("rows", [])
        }
        for result in results
    ]
    rows = []
    for template in results[0]["rows"]:
        key = (template.get("model"), template.get("variable"))
        matching = [indexed[key] for indexed in indexed_results if key in indexed]
        if not matching:
            continue
        row = {"model": key[0], "variable": key[1]}
        for field in numeric_fields:
            values = [item.get(field) for item in matching]
            values = [float(value) for value in values if value is not None]
            row[field] = float(np.mean(values)) if values else None
        if row["or_ci_low"] is not None and row["or_ci_high"] is not None:
            row["OR 95% CI"] = f"[{row['or_ci_low']}, {row['or_ci_high']}]"
        rows.append(row)
    return rows


def evaluate_table1_noise_utility(
    dataframe,
    repetitions,
    low,
    high,
    seed,
    **_,
):
    baseline_result = run_table1(dataframe=dataframe)
    converged_models = [
        model["model"]
        for model in baseline_result.get("model_stats", [])
        if isinstance(model, dict) and model.get("converged") is True
    ]
    if not converged_models:
        raise ValueError("No Table 1 model converged for numerical perturbation evaluation.")
    evaluated_model = converged_models[-1]
    baseline_vector = _table1_vector(baseline_result, "B", evaluated_model)
    successful_results = []
    vectors = []
    for perturbed in _perturbed_runs(dataframe, repetitions, low, high, seed):
        try:
            result = run_table1(dataframe=perturbed)
            evaluated_stats = next(
                (
                    model for model in result.get("model_stats", [])
                    if model.get("model") == evaluated_model
                ),
                {},
            )
            if not evaluated_stats.get("converged", False):
                continue
            vector = _table1_vector(result, "B", evaluated_model)
            if np.isfinite(vector).all():
                successful_results.append(result)
                vectors.append(vector)
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    if not vectors:
        raise RuntimeError("No noisy Table 1 regression completed successfully.")

    array = np.asarray(vectors, dtype=float)
    mean = array.mean(axis=0)
    standard_deviation = (
        array.std(axis=0, ddof=1) if len(array) > 1 else np.zeros_like(mean)
    )
    per_run_rmse = np.sqrt(np.mean((array - baseline_vector) ** 2, axis=1))
    mean_rows = _mean_table1_rows(successful_results)
    legacy_figure = create_mean_perturbed_table1_figure(mean, standard_deviation)
    stability_figure = create_table1_stability_figure(
        baseline_vector, mean, standard_deviation
    )
    return _noise_payload(
        analysis_key="table1",
        mean_perturbed_result={
            "title": "Mean Perturbed Table 1-style Logistic Regression",
            "table": mean_rows,
            "figure": None,
        },
        average_difference={
            "label": "Average Logistic Coefficient Difference",
            "value": float(per_run_rmse.mean()),
            "display_unit": "",
            "explanation": (
                "This value shows the average change in the logistic regression "
                "coefficients after noise was added. A smaller value means that "
                "the coefficients remained more stable."
            ),
        },
        stability_figure=stability_figure,
        repetitions=repetitions,
        successful_runs=len(vectors),
        base_seed=seed,
        technical_details={
            "evaluated_model": evaluated_model,
            "coefficient_rmse_sd": float(
                per_run_rmse.std(ddof=1) if len(per_run_rmse) > 1 else 0.0
            ),
            "successful_noisy_models": len(vectors),
        },
        internal_summary={
            "baseline_vector": baseline_vector,
            "mean_vector": mean,
            "standard_deviation": standard_deviation,
        },
        legacy_noisy_figure=legacy_figure,
    )


def _correlation(
    dataframe: pd.DataFrame, variables: list[str], method: str = "pearson"
) -> np.ndarray:
    return (
        dataframe[variables]
        .apply(pd.to_numeric, errors="coerce")
        .corr(method=method)
        .to_numpy(dtype=float)
    )


def evaluate_correlation_noise_utility(
    dataframe,
    repetitions,
    low,
    high,
    seed,
    variables=None,
    method="pearson",
    **_,
):
    variables = list(variables or PREDICTORS)
    baseline_matrix = _correlation(dataframe, variables, method)
    upper = np.triu_indices(len(variables), 1)
    matrices = np.asarray(
        [
            _correlation(perturbed, variables, method)
            for perturbed in _perturbed_runs(dataframe, repetitions, low, high, seed)
        ],
        dtype=float,
    )
    vectors = matrices[:, upper[0], upper[1]]
    baseline_vector = baseline_matrix[upper]
    mean_vector = vectors.mean(axis=0)
    standard_deviation = (
        vectors.std(axis=0, ddof=1)
        if repetitions > 1
        else np.zeros_like(mean_vector)
    )
    per_run_rmse = np.sqrt(np.mean((vectors - baseline_vector) ** 2, axis=1))
    mean_matrix = matrices.mean(axis=0)
    np.fill_diagonal(mean_matrix, 1.0)
    mean_rows = []
    pair_index = 0
    for first_index, first in enumerate(variables):
        for second in variables[first_index + 1 :]:
            mean_rows.append(
                {
                    "variable_1": first,
                    "variable_2": second,
                    "correlation": float(mean_vector[pair_index]),
                    "correlation_sd": float(standard_deviation[pair_index]),
                    "method": method,
                    "n": int(len(dataframe)),
                }
            )
            pair_index += 1
    mean_figure = create_mean_perturbed_correlation_figure(
        mean_matrix, variables=variables
    )
    stability_figure = create_correlation_stability_figure(
        baseline_vector,
        mean_vector,
        standard_deviation,
        variables=variables,
    )
    return _noise_payload(
        analysis_key="correlation",
        mean_perturbed_result={
            "title": "Mean Perturbed Correlation Result",
            "table": mean_rows,
            # The current original correlation result is a 28-row table. Keep
            # the mean result the same public type; the heatmap remains in the
            # compatibility field for callers that already render it.
            "figure": None,
        },
        average_difference={
            "label": "Average Correlation Difference",
            "value": float(per_run_rmse.mean()),
            "display_unit": "",
            "explanation": (
                "This value shows the average change in the pairwise correlations "
                "after noise was added. A smaller value means that the correlation "
                "structure remained more stable."
            ),
        },
        stability_figure=stability_figure,
        repetitions=repetitions,
        successful_runs=repetitions,
        base_seed=seed,
        technical_details={
            "correlation_rmse_sd": float(
                per_run_rmse.std(ddof=1) if repetitions > 1 else 0.0
            ),
            "pair_count": int(len(mean_vector)),
            "sign_agreement": float(
                np.mean(np.sign(mean_vector) == np.sign(baseline_vector))
            ),
        },
        internal_summary={
            "mean_matrix": mean_matrix,
            "baseline_vector": baseline_vector,
            "mean_vector": mean_vector,
            "standard_deviation": standard_deviation,
        },
        legacy_noisy_figure=mean_figure,
    )


def _elite_membership(dataframe: pd.DataFrame) -> pd.Series:
    if "elite_status" in dataframe.columns:
        values = dataframe["elite_status"]
        if pd.api.types.is_numeric_dtype(values):
            return pd.to_numeric(values, errors="coerce").eq(1)
        return values.astype(str).str.casefold().isin({"elite", "1", "true", "yes"})
    return pd.to_numeric(dataframe["expertise_value"], errors="coerce").ge(13)


def _variance_sample_plan(
    dataframe: pd.DataFrame,
    *,
    iterations: int = 1000,
    sample_size: int = 22,
):
    elite_mask = _elite_membership(dataframe)
    elite_indexes = dataframe.index[elite_mask].tolist()
    semi_indexes = np.asarray(dataframe.index[~elite_mask])
    size = min(sample_size, len(elite_indexes), len(semi_indexes))
    if size < 2:
        raise ValueError("Variance evaluation requires at least two athletes per group.")
    elite_indexes = elite_indexes[:size]
    rng = np.random.default_rng(42)
    semi_draws = [
        rng.choice(semi_indexes, size=size, replace=False).tolist()
        for _ in range(iterations)
    ]
    return elite_indexes, semi_draws


def _variance_components(
    dataframe: pd.DataFrame,
    plan,
    *,
    standardize: bool = False,
) -> dict[str, np.ndarray]:
    numeric = dataframe[PREDICTORS].apply(pd.to_numeric, errors="coerce")
    if standardize:
        numeric = (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, 1)
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    elite_indexes, semi_draws = plan
    elite_values = numeric.loc[elite_indexes].to_numpy(dtype=float)
    elite_variance = np.var(elite_values, axis=0, ddof=1)
    semi_variances = np.asarray(
        [
            np.var(numeric.loc[indexes].to_numpy(dtype=float), axis=0, ddof=1)
            for indexes in semi_draws
        ],
        dtype=float,
    )
    return {
        "elite": elite_variance,
        "semi_mean": semi_variances.mean(axis=0),
        "semi_min": semi_variances.min(axis=0),
        "semi_max": semi_variances.max(axis=0),
    }


def _variance_vector(components: dict[str, np.ndarray]) -> np.ndarray:
    return np.vstack([components["elite"], components["semi_mean"]])


def evaluate_variance_noise_utility(
    dataframe,
    repetitions,
    low,
    high,
    seed,
    iterations=1000,
    **_,
):
    plan = _variance_sample_plan(dataframe, iterations=iterations)
    baseline_components = _variance_components(dataframe, plan)
    baseline = _variance_vector(baseline_components)
    run_components = [
        _variance_components(perturbed, plan)
        for perturbed in _perturbed_runs(dataframe, repetitions, low, high, seed)
    ]
    runs = np.asarray([_variance_vector(item) for item in run_components], dtype=float)
    mean = runs.mean(axis=0)
    standard_deviation = (
        runs.std(axis=0, ddof=1) if repetitions > 1 else np.zeros_like(mean)
    )
    per_run_rmse = np.sqrt(np.mean((runs - baseline) ** 2, axis=(1, 2)))
    mean_rows = [
        {
            "variable": variable,
            "elite_variance": float(mean[0, index]),
            "semi_elite_variance": float(mean[1, index]),
            "iterations": int(iterations),
        }
        for index, variable in enumerate(PREDICTORS)
    ]
    mean_figure = create_mean_perturbed_variance_figure(mean, standard_deviation)
    stability_figure = create_variance_stability_figure(
        baseline, mean, standard_deviation
    )
    return _noise_payload(
        analysis_key="variance_analysis",
        mean_perturbed_result={
            "title": "Mean Perturbed Variance Analysis",
            "table": mean_rows,
            "figure": mean_figure,
        },
        average_difference={
            "label": "Average Variance Difference",
            "value": float(per_run_rmse.mean()),
            "display_unit": "",
            "explanation": (
                "This value shows the average change in the variance results after "
                "noise was added. A smaller value means that the variance analysis "
                "remained more stable."
            ),
        },
        stability_figure=stability_figure,
        repetitions=repetitions,
        successful_runs=repetitions,
        base_seed=seed,
        technical_details={
            "variance_rmse_sd": float(
                per_run_rmse.std(ddof=1) if repetitions > 1 else 0.0
            ),
            "fixed_group_membership": True,
            "fixed_sampling_plan": True,
        },
        internal_summary={
            "baseline": baseline,
            "mean": mean,
            "standard_deviation": standard_deviation,
        },
        sampling_invariants={"fixed_sampling_plan": True},
    )


def _standardized_betas(dataframe: pd.DataFrame) -> np.ndarray:
    import statsmodels.api as sm

    predictors = dataframe[PREDICTORS].apply(pd.to_numeric, errors="coerce")
    outcome = pd.to_numeric(dataframe["expertise_value"], errors="coerce")
    valid = outcome.notna() & predictors.notna().all(axis=1)
    predictors = predictors.loc[valid]
    outcome = outcome.loc[valid]
    model = sm.OLS(outcome, sm.add_constant(predictors)).fit()
    outcome_standard_deviation = float(outcome.std(ddof=0))
    if not np.isfinite(outcome_standard_deviation) or outcome_standard_deviation == 0:
        outcome_standard_deviation = 1.0
    return np.asarray(
        [
            float(model.params[predictor])
            * float(predictors[predictor].std(ddof=0))
            / outcome_standard_deviation
            for predictor in PREDICTORS
        ],
        dtype=float,
    )


def _mean_variance_summary(run_components: list[dict[str, np.ndarray]]) -> dict:
    return {
        predictor: {
            key: float(
                np.mean([components[key][index] for components in run_components])
            )
            for key in ("elite", "semi_mean", "semi_min", "semi_max")
        }
        for index, predictor in enumerate(PREDICTORS)
    }


def evaluate_figure1_noise_utility(
    dataframe,
    repetitions,
    low,
    high,
    seed,
    variance_iterations=1000,
    correlation_threshold=0.15,
    **_,
):
    plan = _variance_sample_plan(dataframe, iterations=variance_iterations)
    baseline_beta = _standardized_betas(dataframe)
    beta_runs = []
    correlation_runs = []
    variance_runs = []
    for perturbed in _perturbed_runs(dataframe, repetitions, low, high, seed):
        beta_runs.append(_standardized_betas(perturbed))
        correlation_runs.append(_correlation(perturbed, list(PREDICTORS)))
        variance_runs.append(_variance_components(perturbed, plan, standardize=True))

    beta_array = np.asarray(beta_runs, dtype=float)
    mean_beta = beta_array.mean(axis=0)
    beta_standard_deviation = (
        beta_array.std(axis=0, ddof=1)
        if repetitions > 1
        else np.zeros_like(mean_beta)
    )
    per_run_rmse = np.sqrt(
        np.mean((beta_array - baseline_beta) ** 2, axis=1)
    )
    mean_correlation = np.asarray(correlation_runs, dtype=float).mean(axis=0)
    mean_variance_summary = _mean_variance_summary(variance_runs)
    mean_table = [
        {
            "variable": DISPLAY_NAMES.get(predictor, predictor),
            "beta": float(mean_beta[index]),
            "elite_variance": mean_variance_summary[predictor]["elite"],
            "semi_elite_variance_mean": mean_variance_summary[predictor]["semi_mean"],
            "semi_elite_variance_min": mean_variance_summary[predictor]["semi_min"],
            "semi_elite_variance_max": mean_variance_summary[predictor]["semi_max"],
        }
        for index, predictor in enumerate(PREDICTORS)
    ]
    mean_figure = create_mean_perturbed_figure1(
        mean_beta,
        mean_correlation,
        mean_variance_summary,
        variables=PREDICTORS,
        correlation_threshold=correlation_threshold,
    )
    stability_figure = create_figure1_stability_figure(
        baseline_beta, mean_beta, beta_standard_deviation
    )
    return _noise_payload(
        analysis_key="figure1",
        mean_perturbed_result={
            "title": "Mean Perturbed Figure 1",
            "table": mean_table,
            "figure": mean_figure,
        },
        average_difference={
            "label": "Average Figure 1 Coefficient Difference",
            "value": float(per_run_rmse.mean()),
            "display_unit": "",
            "explanation": (
                "This value shows the average change in the standardized "
                "coefficients used in Figure 1 after noise was added."
            ),
        },
        stability_figure=stability_figure,
        repetitions=repetitions,
        successful_runs=repetitions,
        base_seed=seed,
        technical_details={
            "figure1_coefficient_rmse_sd": float(
                per_run_rmse.std(ddof=1) if repetitions > 1 else 0.0
            ),
            "fixed_sampling_plan": True,
        },
        internal_summary={
            "baseline_beta": baseline_beta,
            "mean_beta": mean_beta,
            "standard_deviation": beta_standard_deviation,
            "mean_correlation": mean_correlation,
        },
        sampling_invariants={"fixed_sampling_plan": True},
    )


def _full_source_profiles(
    dataframe: pd.DataFrame,
    *,
    variables: list[str],
    cohort_indexes,
    selected_indexes,
) -> tuple[np.ndarray, np.ndarray]:
    numeric = dataframe[variables].apply(pd.to_numeric, errors="coerce")
    z_scores = (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, 1)
    z_scores = z_scores.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    selected_profiles = z_scores.loc[selected_indexes, variables].to_numpy(dtype=float)
    cohort_mean = z_scores.loc[cohort_indexes, variables].mean(axis=0).to_numpy(dtype=float)
    return selected_profiles, cohort_mean


def evaluate_figure2_noise_utility(
    dataframe,
    repetitions,
    low,
    high,
    seed,
    variables=None,
    filters=None,
    max_athletes=50,
    **_,
):
    variables = list(variables or PREDICTORS)
    cohort = apply_analysis_filters(dataframe, filters)
    if cohort.empty:
        raise ValueError("Figure 2 noise evaluation requires a non-empty cohort.")
    count = (
        len(cohort)
        if max_athletes is None
        else min(int(max_athletes), len(cohort))
    )
    selected = (
        cohort.sample(frac=1.0, random_state=2024)
        if max_athletes is None
        else cohort.sample(n=count, random_state=2024)
    )
    selected_indexes = selected.index.tolist()
    cohort_indexes = cohort.index.tolist()
    baseline_profiles, baseline_cohort_mean = _full_source_profiles(
        dataframe,
        variables=variables,
        cohort_indexes=cohort_indexes,
        selected_indexes=selected_indexes,
    )
    profile_runs = []
    cohort_mean_runs = []
    for perturbed in _perturbed_runs(dataframe, repetitions, low, high, seed):
        profiles, cohort_mean = _full_source_profiles(
            perturbed,
            variables=variables,
            cohort_indexes=cohort_indexes,
            selected_indexes=selected_indexes,
        )
        profile_runs.append(profiles)
        cohort_mean_runs.append(cohort_mean)
    profile_array = np.asarray(profile_runs, dtype=float)
    cohort_mean_array = np.asarray(cohort_mean_runs, dtype=float)
    mean_profiles = profile_array.mean(axis=0)
    mean_cohort_profile = cohort_mean_array.mean(axis=0)
    cohort_profile_standard_deviation = (
        cohort_mean_array.std(axis=0, ddof=1)
        if repetitions > 1
        else np.zeros_like(mean_cohort_profile)
    )
    per_run_rmse = np.sqrt(
        np.mean((profile_array - baseline_profiles) ** 2, axis=(1, 2))
    )
    pattern_indexes = [
        variables.index(name)
        for name in (
            "basic_cognitive_function",
            "lower_body_dynamics",
            "blood_micronutrients",
        )
        if name in variables
    ]
    mean_table = [
        {
            "anonymous_profile_label": f"Profile {index:02d}",
            "pattern_match": bool(
                pattern_indexes
                and all(profile[position] > 0 for position in pattern_indexes)
            ),
            "pattern_description": (
                "Matches the paper's three-domain group-level pattern"
                if pattern_indexes
                and all(profile[position] > 0 for position in pattern_indexes)
                else "Does not match the paper's three-domain group-level pattern"
            ),
        }
        for index, profile in enumerate(mean_profiles, start=1)
    ]
    group_label = (
        " / ".join(
            str(value).replace("_", " ").title()
            for value in (filters or {}).values()
        )
        or "All athletes"
    )
    mean_figure = create_mean_perturbed_figure2(
        mean_profiles,
        selected_dataframe=selected,
        cohort_dataframe=cohort,
        reference_profile=mean_cohort_profile,
        variables=variables,
        group_label=f"Mean perturbed {group_label}",
    )
    stability_figure = create_figure2_stability_figure(
        baseline_cohort_mean,
        mean_cohort_profile,
        cohort_profile_standard_deviation,
    )
    return _noise_payload(
        analysis_key="figure2",
        mean_perturbed_result={
            "title": "Mean Perturbed Figure 2",
            "table": mean_table,
            "figure": mean_figure,
        },
        average_difference={
            "label": "Average Profile Difference",
            "value": float(per_run_rmse.mean()),
            "display_unit": "z-score units",
            "explanation": (
                "This value shows the average change in the standardized athlete "
                "profiles after noise was added."
            ),
        },
        stability_figure=stability_figure,
        repetitions=repetitions,
        successful_runs=repetitions,
        base_seed=seed,
        technical_details={
            "profile_rmse_sd": float(
                per_run_rmse.std(ddof=1) if repetitions > 1 else 0.0
            ),
            "anonymous_profile_count": int(count),
            "same_selected_profiles": True,
            "full_source_standardization": True,
        },
        internal_summary={
            "baseline_cohort_profile": baseline_cohort_mean,
            "mean_cohort_profile": mean_cohort_profile,
            "cohort_profile_standard_deviation": cohort_profile_standard_deviation,
            "domain_mean_absolute_change": np.mean(
                np.abs(mean_profiles - baseline_profiles), axis=0
            ),
        },
        selection_invariants={
            "same_selected_profiles": True,
            "same_profile_order": True,
            "same_filters": True,
            "full_source_standardization": True,
        },
    )


def _individual_profile_components(
    dataframe: pd.DataFrame,
    *,
    subject_index,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one clipped anonymous profile and the elite reference profile."""
    if subject_index not in dataframe.index:
        raise ValueError("The fixed anonymous subject is unavailable.")
    numeric = dataframe[DOMAIN_ORDER].apply(pd.to_numeric, errors="coerce")
    means = numeric.mean(axis=0)
    standard_deviations = numeric.std(axis=0, ddof=0).replace(0, 1)
    selected = numeric.loc[subject_index]
    if isinstance(selected, pd.DataFrame):
        selected = selected.iloc[0]
    profile = ((selected - means) / standard_deviations).clip(-3.0, 3.0)
    elite_mask = pd.to_numeric(
        dataframe["expertise_value"], errors="coerce"
    ).ge(13)
    elite_profiles = (numeric.loc[elite_mask] - means) / standard_deviations
    elite_mean = elite_profiles.mean(axis=0)
    return (
        profile.loc[DOMAIN_ORDER].to_numpy(dtype=float),
        elite_mean.loc[DOMAIN_ORDER].to_numpy(dtype=float),
    )


def _profile_interpretation(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "Unavailable"
    if value >= 1.5:
        return "Well above average"
    if value >= 0.5:
        return "Above average"
    if value > -0.5:
        return "Near average"
    if value > -1.5:
        return "Below average"
    return "Well below average"


def _anonymous_profile_rows(values: np.ndarray) -> list[dict]:
    rows = []
    for domain, raw_value in zip(DOMAIN_ORDER, values):
        value = None if not np.isfinite(raw_value) else float(raw_value)
        rows.append(
            {
                "variable": DISPLAY_NAMES.get(
                    domain, domain.replace("_", " ").title()
                ),
                "domain_key": domain,
                "z_score": value,
                "interpretation": _profile_interpretation(value),
            }
        )
    return rows


def evaluate_individual_profile_noise_utility(
    dataframe,
    repetitions,
    low,
    high,
    seed,
    *,
    subject_index=None,
    variables=None,
    **_,
):
    """Evaluate one fixed anonymous profile without exposing its identifier."""
    variables = list(variables or DOMAIN_ORDER)
    if variables != list(DOMAIN_ORDER):
        raise ValueError(
            "Anonymous profile noise evaluation requires the eight canonical domains."
        )
    if subject_index is None:
        raise ValueError("A fixed anonymous subject is required for noise evaluation.")

    baseline_profile, _ = _individual_profile_components(
        dataframe, subject_index=subject_index
    )
    profile_runs = []
    elite_reference_runs = []
    for perturbed in _perturbed_runs(dataframe, repetitions, low, high, seed):
        profile, elite_reference = _individual_profile_components(
            perturbed, subject_index=subject_index
        )
        profile_runs.append(profile)
        elite_reference_runs.append(elite_reference)

    profile_array = np.asarray(profile_runs, dtype=float)
    elite_reference_array = np.asarray(elite_reference_runs, dtype=float)
    if not np.isfinite(profile_array).all():
        raise ValueError(
            "Anonymous profile noise evaluation produced unavailable domain values."
        )
    mean_profile = profile_array.mean(axis=0)
    profile_standard_deviation = (
        profile_array.std(axis=0, ddof=1)
        if repetitions > 1
        else np.zeros_like(mean_profile)
    )
    mean_elite_reference = np.nanmean(elite_reference_array, axis=0)
    per_run_rmse = np.sqrt(
        np.mean((profile_array - baseline_profile) ** 2, axis=1)
    )
    mean_rows = _anonymous_profile_rows(mean_profile)

    from .figures import generate_individual_profile_line_figure

    mean_figure = generate_individual_profile_line_figure(
        mean_rows,
        profile_label="Mean Perturbed Anonymous Profile",
        elite_mean_profile=[float(value) for value in mean_elite_reference],
    )
    stability_figure = create_individual_profile_stability_figure(
        baseline_profile,
        mean_profile,
        profile_standard_deviation,
    )
    return _noise_payload(
        analysis_key="individual_profile",
        mean_perturbed_result={
            "title": "Mean Perturbed Anonymous Athlete Profile",
            "table": mean_rows,
            "figure": mean_figure,
        },
        average_difference={
            "label": "Average Anonymous Profile Difference",
            "value": float(per_run_rmse.mean()),
            "display_unit": "z-score units",
            "explanation": (
                "This value shows the average change in the same anonymous "
                "athlete profile after noise was added. A smaller value means "
                "that the profile remained more stable."
            ),
        },
        stability_figure=stability_figure,
        repetitions=repetitions,
        successful_runs=repetitions,
        base_seed=seed,
        technical_details={
            "profile_rmse_sd": float(
                per_run_rmse.std(ddof=1) if repetitions > 1 else 0.0
            ),
            "same_anonymous_subject": True,
            "subject_identifier_exposed": False,
            "full_source_standardization": True,
        },
        internal_summary={
            "baseline_profile": baseline_profile,
            "mean_profile": mean_profile,
            "profile_standard_deviation": profile_standard_deviation,
            "mean_elite_reference": mean_elite_reference,
        },
        selection_invariants={
            "same_anonymous_subject": True,
            "same_profile_order": True,
            "subject_identifier_exposed": False,
            "full_source_standardization": True,
        },
    )


def evaluate_analysis_noise_utility(
    analysis_key,
    dataframe,
    *,
    filters=None,
    variables=None,
    repetitions=NOISE_REPETITIONS,
    noise_low=NOISE_LOW,
    noise_high=NOISE_HIGH,
    base_seed=BASE_RANDOM_SEED,
    **options,
):
    if analysis_key not in ALLOWED:
        raise ValueError(f"Unsupported noise analysis: {analysis_key}")
    if analysis_key == "table2":
        from .regression_noise_utility import evaluate_table2_noise_utility

        return evaluate_table2_noise_utility(
            dataframe,
            repetitions=repetitions,
            noise_low=noise_low,
            noise_high=noise_high,
            base_seed=base_seed,
            group=options.get("group", "all"),
        )
    evaluator = {
        "table1": evaluate_table1_noise_utility,
        "figure1": evaluate_figure1_noise_utility,
        "figure2": evaluate_figure2_noise_utility,
        "correlation": evaluate_correlation_noise_utility,
        "variance_analysis": evaluate_variance_noise_utility,
        "individual_profile": evaluate_individual_profile_noise_utility,
    }[analysis_key]
    return evaluator(
        dataframe,
        repetitions,
        noise_low,
        noise_high,
        base_seed,
        filters=filters,
        variables=variables,
        **options,
    )
