"""Shared adapters for the controlled numerical-perturbation benchmark.

This module intentionally delegates data loading, perturbation, regression,
correlation, variance sampling, and profile construction to the production
analysis modules. It only extracts fixed vectors and computes evaluation metrics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from privacy.numerical_perturbation import create_uniformly_perturbed_dataframe
from sports.analysis import fit_table2_models_raw, load_data, run_table1
from sports.analysis_noise_utility import (
    _correlation,
    _full_source_profiles,
    _standardized_betas,
    _variance_components,
    _variance_sample_plan,
    _variance_vector,
)
from sports.config import DISPLAY_NAMES, PREDICTORS

ANALYSES = (
    "Logistic Regression",
    "Multiple Linear Regression",
    "Figure 1",
    "Figure 2",
    "Correlation Analysis",
    "Variance Analysis",
)
NETWORK_THRESHOLD = 0.15
METRIC_FIELDS = (
    "sign_agreement", "significance_agreement", "top3_overlap", "rank_score",
    "edge_jaccard", "group_order_agreement", "highest_domain_retention",
    "category_agreement", "top2_overlap", "bottom2_overlap", "r_squared_change",
    "regression_sign_agreement", "regression_top3_overlap",
    "variance_rank_agreement", "regression_rank_agreement",
)


@dataclass(frozen=True)
class AnalysisResult:
    vector: np.ndarray
    details: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkContext:
    original: pd.DataFrame
    baselines: dict[str, AnalysisResult]
    variance_plan: Any
    figure2_cohort_indexes: list
    figure2_selected_indexes: list
    variance_iterations: int


def load_benchmark_data() -> pd.DataFrame:
    dataframe = load_data().copy(deep=True)
    missing = [column for column in PREDICTORS if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Dataset is missing standardized domains: {missing}")
    numeric = dataframe[PREDICTORS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("The eight standardized domain columns must be complete and numeric")
    return dataframe


def perturb(dataframe: pd.DataFrame, *, amplitude: float, seed: int) -> pd.DataFrame:
    if not math.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("Noise amplitude must be a positive finite number")
    return create_uniformly_perturbed_dataframe(
        dataframe.copy(deep=True), seed=seed, lower_bound=-amplitude,
        upper_bound=amplitude, columns=list(PREDICTORS))


def _primary_table1(dataframe: pd.DataFrame) -> AnalysisResult:
    result = run_table1(dataframe=dataframe)
    stats = result.get("model_stats") or []
    if not stats or not stats[-1].get("converged", False):
        raise RuntimeError("Primary logistic regression did not converge")
    rows = {row.get("variable"): row for row in result.get("rows", [])
            if str(row.get("model", "")).startswith("d)")}
    selected = [rows[DISPLAY_NAMES[predictor]] for predictor in PREDICTORS]
    vector = np.asarray([row["B"] for row in selected], dtype=float)
    pvalues = np.asarray([row.get("p", np.nan) for row in selected], dtype=float)
    return AnalysisResult(vector, {"pvalues": pvalues})


def _primary_table2(dataframe: pd.DataFrame) -> AnalysisResult:
    raw = fit_table2_models_raw(dataframe.copy(deep=True), group="all")
    primary = next(model for model in raw["models"] if str(model["model"]).startswith("d)"))
    vector = np.asarray([primary["coefficients"][name]["standardized_coefficient"]
                         for name in PREDICTORS], dtype=float)
    pvalues = np.asarray([primary["coefficients"][name]["p_value"] for name in PREDICTORS], dtype=float)
    return AnalysisResult(vector, {"pvalues": pvalues, "r_squared": float(primary["r_squared"])})


def _correlation_result(dataframe: pd.DataFrame) -> AnalysisResult:
    matrix = _correlation(dataframe, list(PREDICTORS), "pearson")
    upper = np.triu_indices(len(PREDICTORS), 1)
    return AnalysisResult(np.asarray(matrix[upper], dtype=float), {"matrix": matrix})


def _variance_result(dataframe: pd.DataFrame, plan, *, standardize: bool = False) -> AnalysisResult:
    components = _variance_components(dataframe, plan, standardize=standardize)
    vector = _variance_vector(components).reshape(-1)
    return AnalysisResult(vector, {"components": components})


def _figure1_result(dataframe: pd.DataFrame, plan) -> AnalysisResult:
    regression = _standardized_betas(dataframe)
    correlation = _correlation_result(dataframe)
    variance = _variance_result(dataframe, plan, standardize=True)
    vector = np.concatenate([regression, correlation.vector, variance.vector])
    return AnalysisResult(vector, {
        "regression": regression, "correlation": correlation,
        "variance": variance,
    })


def _figure2_result(dataframe: pd.DataFrame, cohort_indexes: list, selected_indexes: list) -> AnalysisResult:
    profiles, cohort_mean = _full_source_profiles(
        dataframe, variables=list(PREDICTORS), cohort_indexes=cohort_indexes,
        selected_indexes=selected_indexes)
    return AnalysisResult(profiles.reshape(-1), {
        "profiles": profiles, "cohort_mean": cohort_mean,
    })


def build_context(*, variance_iterations: int) -> BenchmarkContext:
    original = load_benchmark_data()
    plan = _variance_sample_plan(original, iterations=variance_iterations)
    cohort_indexes = original.index.tolist()
    selected_indexes = original.sample(
        n=min(50, len(original)), random_state=2024).index.tolist()
    baselines = {
        "Logistic Regression": _primary_table1(original),
        "Multiple Linear Regression": _primary_table2(original),
        "Correlation Analysis": _correlation_result(original),
        "Variance Analysis": _variance_result(original, plan),
        "Figure 1": _figure1_result(original, plan),
        "Figure 2": _figure2_result(original, cohort_indexes, selected_indexes),
    }
    return BenchmarkContext(original, baselines, plan, cohort_indexes,
                            selected_indexes, variance_iterations)


def analyze_dataframe(name: str, dataframe: pd.DataFrame, context: BenchmarkContext) -> AnalysisResult:
    if name == "Logistic Regression":
        return _primary_table1(dataframe)
    if name == "Multiple Linear Regression":
        return _primary_table2(dataframe)
    if name == "Correlation Analysis":
        return _correlation_result(dataframe)
    if name == "Variance Analysis":
        return _variance_result(dataframe, context.variance_plan)
    if name == "Figure 1":
        return _figure1_result(dataframe, context.variance_plan)
    if name == "Figure 2":
        return _figure2_result(dataframe, context.figure2_cohort_indexes,
                               context.figure2_selected_indexes)
    raise ValueError(f"Unsupported analysis: {name}")


def _agreement(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.asarray(left) == np.asarray(right)))


def _top_overlap(left: np.ndarray, right: np.ndarray, count: int) -> float:
    a = set(np.argsort(np.abs(left))[-count:])
    b = set(np.argsort(np.abs(right))[-count:])
    return len(a & b) / count


def _rank_score(left: np.ndarray, right: np.ndarray) -> float:
    a = pd.Series(np.abs(left)).rank(method="average")
    b = pd.Series(np.abs(right)).rank(method="average")
    rho = float(a.corr(b, method="spearman"))
    return (rho + 1.0) / 2.0 if math.isfinite(rho) else float("nan")


def _regression_metrics(original: AnalysisResult, perturbed: AnalysisResult) -> dict:
    metrics = {
        "sign_agreement": _agreement(np.sign(original.vector), np.sign(perturbed.vector)),
        "top3_overlap": _top_overlap(original.vector, perturbed.vector, 3),
        "rank_score": _rank_score(original.vector, perturbed.vector),
    }
    original_p = np.asarray(original.details.get("pvalues", []), dtype=float)
    perturbed_p = np.asarray(perturbed.details.get("pvalues", []), dtype=float)
    metrics["significance_agreement"] = (
        _agreement(original_p < .05, perturbed_p < .05)
        if len(original_p) == len(original.vector) and np.isfinite(original_p).all()
        and np.isfinite(perturbed_p).all() else float("nan"))
    if "r_squared" in original.details:
        metrics["r_squared_change"] = abs(
            perturbed.details["r_squared"] - original.details["r_squared"])
    return metrics


def _edge_set(vector: np.ndarray) -> set[int]:
    return set(np.flatnonzero(np.abs(vector) > NETWORK_THRESHOLD).tolist())


def _correlation_metrics(original: AnalysisResult, perturbed: AnalysisResult) -> dict:
    first, second = _edge_set(original.vector), _edge_set(perturbed.vector)
    union = first | second
    return {
        "sign_agreement": _agreement(np.sign(original.vector), np.sign(perturbed.vector)),
        "top3_overlap": _top_overlap(original.vector, perturbed.vector, 3),
        "rank_score": _rank_score(original.vector, perturbed.vector),
        "edge_jaccard": len(first & second) / len(union) if union else 1.0,
    }


def _variance_metrics(original: AnalysisResult, perturbed: AnalysisResult) -> dict:
    base = original.vector.reshape(2, len(PREDICTORS))
    noisy = perturbed.vector.reshape(2, len(PREDICTORS))
    ranks = [_rank_score(base[index], noisy[index]) for index in range(2)]
    return {
        "rank_score": float(np.nanmean(ranks)),
        "group_order_agreement": _agreement(base[0] > base[1], noisy[0] > noisy[1]),
        "highest_domain_retention": float(np.mean([
            int(np.argmax(base[index]) == np.argmax(noisy[index])) for index in range(2)])),
    }


def _figure2_metrics(original: AnalysisResult, perturbed: AnalysisResult) -> dict:
    # RMSE uses every fixed anonymous profile value, matching the existing
    # Figure 2 utility target. Structural domain ordering uses the eight-value
    # cohort profile so top/bottom overlap remains a domain-level statement.
    base = np.asarray(original.details["cohort_mean"], dtype=float)
    noisy = np.asarray(perturbed.details["cohort_mean"], dtype=float)
    categories = lambda values: np.where(values > 1, 1, np.where(values < -1, -1, 0))
    return {
        "rank_score": _rank_score(base, noisy),
        "top2_overlap": _top_overlap(base, noisy, 2),
        "bottom2_overlap": len(set(np.argsort(base)[:2]) & set(np.argsort(noisy)[:2])) / 2,
        "category_agreement": _agreement(categories(base), categories(noisy)),
    }


def comparison_metrics(name: str, original: AnalysisResult, perturbed: AnalysisResult) -> dict:
    if original.vector.shape != perturbed.vector.shape or not np.isfinite(perturbed.vector).all():
        raise ValueError("Analysis returned an invalid or incompatible numerical vector")
    difference = perturbed.vector - original.vector
    metrics = {field: float("nan") for field in METRIC_FIELDS}
    metrics.update({
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "relative_error": float(np.linalg.norm(difference) /
                                (np.linalg.norm(original.vector) + 1e-12)),
    })
    if name in {"Logistic Regression", "Multiple Linear Regression"}:
        metrics.update(_regression_metrics(original, perturbed))
    elif name == "Correlation Analysis":
        metrics.update(_correlation_metrics(original, perturbed))
    elif name == "Variance Analysis":
        metrics.update(_variance_metrics(original, perturbed))
    elif name == "Figure 2":
        metrics.update(_figure2_metrics(original, perturbed))
    elif name == "Figure 1":
        regression_original = AnalysisResult(original.details["regression"], {})
        regression_noisy = AnalysisResult(perturbed.details["regression"], {})
        reg = _regression_metrics(regression_original, regression_noisy)
        corr = _correlation_metrics(original.details["correlation"], perturbed.details["correlation"])
        variance = _variance_metrics(original.details["variance"], perturbed.details["variance"])
        metrics.update({
            "regression_sign_agreement": reg["sign_agreement"],
            "regression_top3_overlap": reg["top3_overlap"],
            "regression_rank_agreement": reg["rank_score"],
            "edge_jaccard": corr["edge_jaccard"],
            "variance_rank_agreement": variance["rank_score"],
            "group_order_agreement": variance["group_order_agreement"],
        })
    applicable = [metrics[field] for field in METRIC_FIELDS
                  if field != "r_squared_change" and math.isfinite(metrics[field])]
    metrics["structure_score"] = float(np.mean(applicable)) if applicable else float("nan")
    return metrics


def evaluate_one(name: str, perturbed: pd.DataFrame, context: BenchmarkContext) -> dict:
    try:
        result = analyze_dataframe(name, perturbed, context)
        return {"success": True, "error": "", **comparison_metrics(
            name, context.baselines[name], result)}
    except Exception as exc:
        return {"success": False, "error": f"{type(exc).__name__}: {exc}",
                "rmse": float("nan"), "relative_error": float("nan"),
                "structure_score": float("nan"),
                **{field: float("nan") for field in METRIC_FIELDS}}
