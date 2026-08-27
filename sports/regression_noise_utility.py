"""Table 2-only numerical perturbation utility evaluation."""

from __future__ import annotations

import numpy as np

from privacy.numerical_perturbation import (
    BASE_RANDOM_SEED, NOISE_DISTRIBUTION, NOISE_EXPERIMENT_VERSION, NOISE_HIGH, NOISE_LOW,
    NOISE_REPETITIONS, create_uniformly_perturbed_dataframe,
)
from .analysis import fit_table2_models_raw
from .config import DISPLAY_NAMES, PREDICTORS


def _primary(raw: dict) -> dict:
    return next(model for model in raw["models"] if model["model"].startswith("d)"))


def _mean_perturbed_table2_rows(noisy_models: dict[str, list[dict]], group: str) -> list[dict]:
    """Aggregate the 50 raw Table 2 fits into one public-schema mean table."""
    rows = []
    for model_name, runs in noisy_models.items():
        if not runs:
            continue
        predictor_order = runs[0]["predictor_order"]
        for variable in predictor_order:
            coefficients = [run["coefficients"][variable] for run in runs]
            beta = np.asarray([item["coefficient"] for item in coefficients], dtype=float)
            standardized = np.asarray([
                np.nan if item["standardized_beta"] is None else item["standardized_beta"]
                for item in coefficients
            ], dtype=float)
            standard_error = np.asarray([item["standard_error"] for item in coefficients], dtype=float)
            t_values = np.asarray([item["t_value"] for item in coefficients], dtype=float)
            p_values = np.asarray([item["p_value"] for item in coefficients], dtype=float)
            robust_types = [run["robust_standard_error_type"] for run in runs]
            robust_type = robust_types[0] if len(set(robust_types)) == 1 else "mixed"
            rows.append({
                "group": group,
                "model": model_name,
                "variable": "Intercept" if variable == "const" else DISPLAY_NAMES.get(variable, variable.replace("_", " ").title()),
                "beta_nature": float(np.mean(beta)),
                "standardized_beta": None if np.isnan(standardized).all() else float(np.nanmean(standardized)),
                "SE": float(np.mean(standard_error)),
                "t": float(np.mean(t_values)),
                "p": float(np.mean(p_values)),
                "robust_standard_error_type": robust_type,
                "beta_nature_std": float(np.std(beta, ddof=1)),
                "standardized_beta_std": None if np.isnan(standardized).all() else float(np.nanstd(standardized, ddof=1)),
                "SE_std": float(np.std(standard_error, ddof=1)),
                "t_std": float(np.std(t_values, ddof=1)),
                "p_std": float(np.std(p_values, ddof=1)),
            })
    return rows


def evaluate_table2_noise_utility(
    dataframe,
    *,
    repetitions: int = NOISE_REPETITIONS,
    noise_low: float = NOISE_LOW,
    noise_high: float = NOISE_HIGH,
    base_seed: int = BASE_RANDOM_SEED,
    group: str = "all",
) -> dict:
    """Compare Table 2 with 50 independently perturbed copies of one cohort."""
    baseline_raw = fit_table2_models_raw(dataframe.copy(deep=True), group=group)
    if len(baseline_raw["models"]) != 4:
        raise ValueError("All four Table 2 models are required for utility evaluation.")

    noisy_models = {model["model"]: [] for model in baseline_raw["models"]}
    for repetition in range(repetitions):
        perturbed = create_uniformly_perturbed_dataframe(
            dataframe.copy(deep=True), seed=base_seed + repetition,
            lower_bound=noise_low, upper_bound=noise_high,
        )
        fitted = fit_table2_models_raw(perturbed, group=group)
        for model in fitted["models"]:
            noisy_models[model["model"]].append(model)

    model_summaries = []
    for baseline in baseline_raw["models"]:
        runs = noisy_models[baseline["model"]]
        r2 = np.asarray([run["r_squared"] for run in runs], dtype=float)
        adjusted = np.asarray([run["adjusted_r_squared"] for run in runs], dtype=float)
        model_summaries.append({
            "model": baseline["model"],
            "baseline_r_squared": baseline["r_squared"],
            "mean_noisy_r_squared": float(r2.mean()),
            "std_noisy_r_squared": float(r2.std(ddof=1)),
            "mean_delta_r_squared": float(r2.mean() - baseline["r_squared"]),
            "baseline_adjusted_r_squared": baseline["adjusted_r_squared"],
            "mean_noisy_adjusted_r_squared": float(adjusted.mean()),
            "std_noisy_adjusted_r_squared": float(adjusted.std(ddof=1)),
            "mean_delta_adjusted_r_squared": float(adjusted.mean() - baseline["adjusted_r_squared"]),
        })

    baseline = _primary(baseline_raw)
    runs = noisy_models[baseline["model"]]
    coefficient_rows = []
    baseline_betas, noisy_means = [], []
    sign_matches, significance_matches = [], []
    for predictor in PREDICTORS:
        base = baseline["coefficients"][predictor]
        values = np.asarray(
            [run["coefficients"][predictor]["standardized_coefficient"] for run in runs], dtype=float
        )
        pvalues = np.asarray([run["coefficients"][predictor]["p_value"] for run in runs], dtype=float)
        mean = float(values.mean())
        sign_agreement = float(np.mean(np.sign(values) == np.sign(base["standardized_coefficient"])))
        p_agreement = float(np.mean((pvalues < 0.05) == (base["p_value"] < 0.05)))
        baseline_betas.append(base["standardized_coefficient"])
        noisy_means.append(mean)
        sign_matches.append(sign_agreement)
        significance_matches.append(p_agreement)
        coefficient_rows.append({
            "variable": predictor,
            "predictor_label": DISPLAY_NAMES.get(predictor, predictor),
            "baseline_standardized_beta": base["standardized_coefficient"],
            "mean_noisy_standardized_beta": mean,
            "std_noisy_standardized_beta": float(values.std(ddof=1)),
            "min_noisy_standardized_beta": float(values.min()),
            "max_noisy_standardized_beta": float(values.max()),
            "mean_difference": float(mean - base["standardized_coefficient"]),
            "sign_agreement_rate": sign_agreement,
            "significance_agreement_rate": p_agreement,
        })
    baseline_vector = np.asarray(baseline_betas, dtype=float)
    coefficient_rmse_per_run = [
        float(np.sqrt(np.mean((np.asarray([
            run["coefficients"][predictor]["standardized_coefficient"] for predictor in PREDICTORS
        ], dtype=float) - baseline_vector) ** 2))) for run in runs
    ]
    sign_agreement_per_run = [float(np.mean([
        np.sign(run["coefficients"][predictor]["standardized_coefficient"])
        == np.sign(baseline["coefficients"][predictor]["standardized_coefficient"])
        for predictor in PREDICTORS
    ])) for run in runs]
    significance_agreement_per_run = [float(np.mean([
        (run["coefficients"][predictor]["p_value"] < 0.05)
        == (baseline["coefficients"][predictor]["p_value"] < 0.05)
        for predictor in PREDICTORS
    ])) for run in runs]
    primary_model_summary = next(row for row in model_summaries if row["model"] == baseline["model"])
    primary_summary = {
        **primary_model_summary,
        "coefficient_rmse_per_run": coefficient_rmse_per_run,
        "mean_coefficient_rmse": float(np.mean(coefficient_rmse_per_run)),
        "std_coefficient_rmse": float(np.std(coefficient_rmse_per_run, ddof=1)),
        "sign_agreement_per_run": sign_agreement_per_run,
        "mean_sign_agreement": float(np.mean(sign_agreement_per_run)),
        "significance_agreement_per_run": significance_agreement_per_run,
        "mean_significance_agreement": float(np.mean(significance_agreement_per_run)),
    }

    from .noise_figures import create_mean_perturbed_table2_figure_from_arrays, create_table2_stability_figure
    mean_perturbed_rows = _mean_perturbed_table2_rows(noisy_models, group)
    noisy_only_figure=create_mean_perturbed_table2_figure_from_arrays(np.asarray(noisy_means),np.asarray([row["std_noisy_standardized_beta"] for row in coefficient_rows]))
    stability_figure=create_table2_stability_figure(np.asarray(baseline_betas),np.asarray(noisy_means),np.asarray([row["std_noisy_standardized_beta"] for row in coefficient_rows]))
    average_difference = {
        "label": "Average Standardized Coefficient Difference",
        "value": float(primary_summary["mean_coefficient_rmse"]),
        "display_unit": "",
        "explanation": (
            "This value shows the average change in the standardized linear "
            "regression coefficients after noise was added. A smaller value "
            "means that the regression coefficients remained more stable."
        ),
    }
    mean_perturbed_result = {
        "title": "Mean Perturbed Table 2-style Linear Regression",
        "table": mean_perturbed_rows,
        "figure": None,
    }
    return {
        "analysis": "Linear Regression Noise Stability Evaluation",
        "analysis_key":"table2",
        "noise_distribution": NOISE_DISTRIBUTION,
        "noise_range": [float(noise_low), float(noise_high)],
        "repetitions": int(repetitions),
        "independent_repetitions": True,
        "cumulative_noise": False,
        "base_random_seed": int(base_seed),
        "experiment_version": "uniform-0.50-all-analyses-v2",
        "attempted_runs":int(repetitions),"successful_runs":int(repetitions),"failed_runs":0,
        "controls_perturbed":False,
        "perturbed_columns": list(PREDICTORS),
        "outcome_perturbed": False,
        "primary_model": baseline["model"],
        "primary_summary": primary_summary,
        "coefficient_summary": coefficient_rows,
        "model_summary": model_summaries,
        "mean_perturbed_table2": {
            "analysis": "Mean Perturbed Table 2-style Linear Regression",
            "repetitions": int(repetitions),
            "noise_distribution": NOISE_DISTRIBUTION,
            "noise_range": [float(noise_low), float(noise_high)],
            "rows": mean_perturbed_rows,
            "variability_available": True,
        },
        "raw_rows_exposed": False,
        "noise_matrix_exposed": False,
        "privacy_interpretation": "This is a controlled numerical perturbation experiment and is not presented as formal differential privacy.",
        "mean_perturbed_result":mean_perturbed_result,
        "average_difference":average_difference,
        "primary_metric":{"label":average_difference["label"],"value":average_difference["value"],"baseline_value":None,"mean_noisy_value":None,"display_kind":"four_decimals","explanation":average_difference["explanation"]},
        "noisy_result_title":mean_perturbed_result["title"],"noisy_result_caption":"Mean result from independent perturbations using Uniform(-0.50, 0.50).","noisy_result_figure":noisy_only_figure,
        "stability_title":"Standardized Coefficient Stability After Numerical Perturbation","stability_caption":"Original, mean perturbed, and mean perturbed ± 1 SD across independent runs.","stability_figure":stability_figure,
        "technical_details":{"coefficient_rmse":primary_summary["mean_coefficient_rmse"],"sign_agreement":primary_summary["mean_sign_agreement"],"significance_agreement":primary_summary["mean_significance_agreement"]},
        "internal_summary":{"model_summary":model_summaries,"coefficient_summary":coefficient_rows},
        "figure": stability_figure,
    }
