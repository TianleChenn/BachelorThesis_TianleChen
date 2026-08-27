"""Figures for the controlled numerical-perturbation dashboard experiment."""

from __future__ import annotations

from . import matplotlib_backend as _matplotlib_backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DISPLAY_NAMES, PREDICTORS


def _labels(variables=None) -> list[str]:
    return [DISPLAY_NAMES.get(name, name) for name in (variables or PREDICTORS)]


def _coefficient_noisy(mean, standard_deviation, title, *, variables=None):
    """Legacy noisy-only coefficient view retained for payload compatibility."""
    mean = np.asarray(mean, dtype=float)
    standard_deviation = np.asarray(standard_deviation, dtype=float)
    y = np.arange(len(mean))
    fig, ax = plt.subplots(figsize=(10.5, 6.1))
    ax.barh(
        y,
        mean,
        xerr=standard_deviation,
        color="#4c78a8",
        alpha=0.82,
        capsize=4,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(_labels(variables))
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Mean coefficient after noise")
    fig.tight_layout()
    return fig


def _stability_line(
    original,
    perturbed_mean,
    perturbed_standard_deviation,
    title,
    *,
    labels,
    ylabel,
):
    """Draw the common Original/Mean perturbed/mean ± 1 SD line view."""
    original = np.asarray(original, dtype=float)
    perturbed_mean = np.asarray(perturbed_mean, dtype=float)
    perturbed_standard_deviation = np.asarray(
        perturbed_standard_deviation, dtype=float
    )
    if not (
        original.shape
        == perturbed_mean.shape
        == perturbed_standard_deviation.shape
    ):
        raise ValueError("Stability vectors must have identical shapes.")

    x = np.arange(len(original))
    width = max(11.0, min(19.0, 0.34 * len(original) + 7.0))
    fig, ax = plt.subplots(figsize=(width, 5.8))
    ax.plot(
        x,
        original,
        "o-",
        color="#b42318",
        linewidth=2,
        markersize=4,
        label="Original",
    )
    ax.plot(
        x,
        perturbed_mean,
        "o-",
        color="#175cd3",
        linewidth=2,
        markersize=4,
        label="Mean perturbed",
    )
    ax.fill_between(
        x,
        perturbed_mean - perturbed_standard_deviation,
        perturbed_mean + perturbed_standard_deviation,
        color="#84adff",
        alpha=0.30,
        label="Mean perturbed ± 1 SD",
    )
    ax.axhline(0, color="black", linewidth=1, alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def create_mean_perturbed_table1_figure(mean, standard_deviation):
    return _coefficient_noisy(
        mean,
        standard_deviation,
        "Mean Perturbed Table 1-style Logistic Regression",
    )


def create_table1_stability_figure(original, mean, standard_deviation):
    return _stability_line(
        original,
        mean,
        standard_deviation,
        "Logistic Coefficient Stability After Numerical Perturbation",
        labels=_labels(),
        ylabel="B coefficient",
    )


def create_mean_perturbed_table2_figure_from_arrays(mean, standard_deviation):
    return _coefficient_noisy(
        mean,
        standard_deviation,
        "Mean Perturbed Table 2-style Linear Regression",
    )


def create_table2_stability_figure(original, mean, standard_deviation):
    return _stability_line(
        original,
        mean,
        standard_deviation,
        "Standardized Coefficient Stability After Numerical Perturbation",
        labels=_labels(),
        ylabel="Standardized coefficient (Model d)",
    )


def create_figure1_stability_figure(original, mean, standard_deviation):
    return _stability_line(
        original,
        mean,
        standard_deviation,
        "Figure 1 Coefficient Stability After Numerical Perturbation",
        labels=_labels(),
        ylabel="Standardized beta",
    )


def create_mean_perturbed_correlation_figure(matrix, *, variables=None):
    matrix = np.asarray(matrix, dtype=float)
    names = _labels(variables)
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_yticklabels(names)
    ax.set_title("Mean Perturbed Correlation Result")
    fig.colorbar(image, ax=ax, label="Mean Pearson correlation")
    fig.tight_layout()
    return fig


def _correlation_pair_labels(variables=None):
    names = _labels(variables)
    return [
        f"{names[first]} / {names[second]}"
        for first in range(len(names))
        for second in range(first + 1, len(names))
    ]


def create_correlation_stability_figure(
    original, mean, standard_deviation, *, variables=None
):
    return _stability_line(
        original,
        mean,
        standard_deviation,
        "Correlation Stability After Numerical Perturbation",
        labels=_correlation_pair_labels(variables),
        ylabel="Pairwise correlation",
    )


def _variance_group_bars(values, standard_deviation=None, *, title):
    values = np.asarray(values, dtype=float)
    standard_deviation = (
        None
        if standard_deviation is None
        else np.asarray(standard_deviation, dtype=float)
    )
    x = np.arange(values.shape[1])
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    labels = ("Elite variance", "Semi-elite sampled mean")
    colors = ("#2f6f9f", "#8a8a8a")
    for index, label in enumerate(labels):
        errors = None if standard_deviation is None else standard_deviation[index]
        ax.bar(
            x + (index - 0.5) * width,
            values[index],
            width,
            yerr=errors,
            label=label,
            color=colors[index],
            alpha=0.88 if index == 0 else 0.62,
            capsize=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(_labels(), rotation=35, ha="right")
    ax.set_ylabel("Variance of standardized domain score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def create_mean_perturbed_variance_figure(mean, standard_deviation):
    return _variance_group_bars(
        mean,
        standard_deviation,
        title="Mean Perturbed Variance Analysis",
    )


def create_variance_stability_figure(original, mean, standard_deviation=None):
    original = np.asarray(original, dtype=float)
    mean = np.asarray(mean, dtype=float)
    if original.ndim == 1:
        # Backward compatibility for earlier aggregate-domain payloads.
        standard_deviation = (
            np.zeros_like(mean)
            if standard_deviation is None
            else np.asarray(standard_deviation, dtype=float)
        )
        labels = _labels()
    else:
        standard_deviation = (
            np.zeros_like(mean)
            if standard_deviation is None
            else np.asarray(standard_deviation, dtype=float)
        )
        group_names = ("Elite", "Semi-elite")
        labels = [
            f"{group} – {domain}"
            for group in group_names
            for domain in _labels()
        ]
        original = original.reshape(-1)
        mean = mean.reshape(-1)
        standard_deviation = standard_deviation.reshape(-1)
    return _stability_line(
        original,
        mean,
        standard_deviation,
        "Variance Stability After Numerical Perturbation",
        labels=labels,
        ylabel="Variance",
    )


def create_mean_perturbed_figure1(
    mean_beta,
    mean_correlation,
    mean_variance,
    *,
    variables=None,
    correlation_threshold=0.15,
):
    """Render the mean components with the original Figure 1 network logic."""
    from .figures import create_figure1_from_components

    selected_variables = list(variables or PREDICTORS)
    if isinstance(mean_variance, dict):
        variance_summary = mean_variance
    else:
        variance_array = np.asarray(mean_variance, dtype=float)
        variance_summary = {
            variable: {
                "elite": float(variance_array[0, index]),
                "semi_mean": float(variance_array[1, index]),
                "semi_min": float(variance_array[1, index]),
                "semi_max": float(variance_array[1, index]),
            }
            for index, variable in enumerate(selected_variables)
        }
    return create_figure1_from_components(
        standardized_betas=np.asarray(mean_beta, dtype=float),
        correlation_matrix=np.asarray(mean_correlation, dtype=float),
        variance_summary=variance_summary,
        variables=selected_variables,
        correlation_threshold=correlation_threshold,
        title=(
            "Mean Perturbed Figure 1\n"
            "Regression effects, predictor correlations, and elite vs semi-elite variance"
        ),
    )


def create_mean_perturbed_figure2(
    mean_profiles,
    *,
    selected_dataframe=None,
    cohort_dataframe=None,
    reference_profile=None,
    variables=None,
    group_label="Selected cohort",
):
    """Render mean noisy profiles with the original Figure 2 profile function."""
    selected_variables = list(variables or PREDICTORS)
    mean_profiles = np.asarray(mean_profiles, dtype=float)
    if selected_dataframe is not None and cohort_dataframe is not None:
        from .figures import create_figure2

        profile_frame = pd.DataFrame(
            mean_profiles,
            index=selected_dataframe.index,
            columns=selected_variables,
        )
        reference = pd.Series(
            np.asarray(reference_profile, dtype=float),
            index=selected_variables,
        )
        return create_figure2(
            dataframe=cohort_dataframe,
            selected_dataframe=selected_dataframe,
            standardized_profiles=profile_frame,
            reference_profile=reference,
            reference_label="Mean Perturbed Selected Cohort Mean Profile",
            variables=selected_variables,
            group_label=group_label,
            max_athletes=len(selected_dataframe),
        )

    # Compatibility fallback for callers that only have the aggregate array.
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(selected_variables))
    for row in mean_profiles:
        ax.plot(x, row, alpha=0.28)
    ax.set_xticks(x)
    ax.set_xticklabels(_labels(selected_variables), rotation=35, ha="right")
    ax.axhline(0, color="black")
    ax.set_ylabel("z-score")
    ax.set_title("Mean Perturbed Figure 2")
    fig.tight_layout()
    return fig


def create_figure2_stability_figure(original, mean, standard_deviation):
    return _stability_line(
        original,
        mean,
        standard_deviation,
        "Figure 2 Profile Stability After Numerical Perturbation",
        labels=_labels(),
        ylabel="Mean z-score profile",
    )


def create_individual_profile_stability_figure(
    original, mean, standard_deviation
):
    """Compare one fixed anonymous profile across independent noise runs."""
    return _stability_line(
        original,
        mean,
        standard_deviation,
        "Anonymous Profile Stability After Numerical Perturbation",
        labels=_labels(),
        ylabel="z-score",
    )
