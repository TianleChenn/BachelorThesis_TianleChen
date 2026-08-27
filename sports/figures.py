from __future__ import annotations

import numpy as np
import pandas as pd
from . import matplotlib_backend as _matplotlib_backend
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

from .config import DISPLAY_NAMES, DOMAIN_LABELS, DOMAIN_ORDER, PREDICTORS


def _figure_label(predictor: str) -> str:
    return DISPLAY_NAMES.get(predictor, predictor.replace("_", " ").title()).replace(" ", "\n")


def _load_data() -> pd.DataFrame:
    """Use the authoritative protected-dataset loader."""
    from .analysis import load_data

    return load_data()


def _elite_mask(df: pd.DataFrame) -> pd.Series:
    """
    Detect elite athletes robustly.

    Supports:
    - elite_status = 1 / 0
    - elite_status = elite / semi_elite
    - expertise_value >= 13
    """
    if "elite_status" in df.columns:
        col = df["elite_status"]

        if pd.api.types.is_numeric_dtype(col):
            return col.astype(int) == 1

        return col.astype(str).str.lower().isin(["elite", "1", "true", "yes"])

    if "expertise_value" in df.columns:
        return df["expertise_value"] >= 13

    raise ValueError("Dataset must contain either elite_status or expertise_value.")


def _normalize_filter_text(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _ensure_national_team_column(df: pd.DataFrame) -> pd.DataFrame:
    if "national_team" in df.columns:
        return df
    df = df.copy()
    if "age" in df.columns:
        age = pd.to_numeric(df["age"], errors="coerce")
        df["national_team"] = np.where(
            age < 20,
            "junior_national_team",
            "senior_national_team",
        )
    else:
        df["national_team"] = "senior_national_team"
    return df


def _filter_group_for_plot(df: pd.DataFrame, group: str = "all") -> pd.DataFrame:
    group = _normalize_filter_text(group or "all")
    df = _ensure_national_team_column(df)

    if group in {"all", "full", "complete", "all_athletes"}:
        return df.copy()

    mask = _elite_mask(df)
    if group in {"elite", "top", "super"}:
        return df[mask].copy()
    if group in {"semi_elite", "semi", "non_elite", "lower"}:
        return df[~mask].copy()

    if ":" not in group:
        return df.copy()

    category, value = group.split(":", 1)
    value = _normalize_filter_text(value)

    if category == "sport" and "sport" in df.columns:
        return df[df["sport"].map(_normalize_filter_text) == value].copy()
    if category == "sex" and "sex" in df.columns:
        return df[df["sex"].map(_normalize_filter_text) == value].copy()
    if category == "national_team" and "national_team" in df.columns:
        return df[df["national_team"].map(_normalize_filter_text) == value].copy()
    if category == "age_group" and "age" in df.columns:
        age = pd.to_numeric(df["age"], errors="coerce")
        if value in {"under_20", "u20", "junior"}:
            return df[age < 20].copy()
        if value in {"20_and_above", "20_plus", "above_20", "senior"}:
            return df[age >= 20].copy()

    return df.copy()


def _zscore(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    values = df[columns].copy()
    return (values - values.mean()) / values.std(ddof=0).replace(0, 1)


def _standardized_linear_betas(df: pd.DataFrame, predictors: list[str]) -> dict[str, float]:
    """
    Compute standardized beta coefficients for expertise_value.

    This mimics the beta values shown in the Nature paper Table 2.
    Only aggregate coefficients are returned. No raw rows are exposed.
    """
    if "expertise_value" not in df.columns:
        return {p: 0.0 for p in predictors}

    data = df[predictors + ["expertise_value"]].dropna().copy()
    if len(data) < len(predictors) + 2:
        return {p: 0.0 for p in predictors}

    x = data[predictors].copy()
    x_std = x.std(ddof=0).replace(0, 1)
    x_z = (x - x.mean()) / x_std

    y = data["expertise_value"].astype(float)
    y_std = y.std(ddof=0)
    if not np.isfinite(y_std) or y_std == 0:
        y_std = 1.0
    y_z = (y - y.mean()) / y_std

    x_design = np.column_stack([np.ones(len(x_z)), x_z.to_numpy(dtype=float)])
    coef = np.linalg.lstsq(x_design, y_z.to_numpy(dtype=float), rcond=None)[0]

    betas = coef[1:]
    return {p: float(b) for p, b in zip(predictors, betas)}


def _sampled_variance_summary(
    z_scores: pd.DataFrame,
    elite_mask: pd.Series,
    predictors: list[str],
    iterations: int = 1000,
    sample_size: int = 22,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """
    Compute the variance summary used inside each Figure 1 node.

    This returns aggregate statistics only.
    """
    elite_mask = elite_mask.reindex(z_scores.index).fillna(False)

    elite_df = z_scores.loc[elite_mask, predictors]
    semi_df = z_scores.loc[~elite_mask, predictors]

    if len(elite_df) < 2 or len(semi_df) < 2:
        raise ValueError("Not enough elite or semi-elite athletes for variance comparison.")

    sample_size = min(sample_size, len(elite_df), len(semi_df))
    rng = np.random.default_rng(seed)

    elite_used = elite_df.head(sample_size)
    elite_var = elite_used.var(ddof=1)

    sampled_values = {p: [] for p in predictors}
    semi_indices = semi_df.index.to_numpy()

    for _ in range(iterations):
        draw_idx = rng.choice(semi_indices, size=sample_size, replace=False)
        draw = semi_df.loc[draw_idx, predictors]
        draw_var = draw.var(ddof=1)

        for p in predictors:
            sampled_values[p].append(float(draw_var[p]))

    summary = {}
    for p in predictors:
        values = np.array(sampled_values[p], dtype=float)
        summary[p] = {
            "elite": float(elite_var[p]),
            "semi_mean": float(np.mean(values)),
            "semi_min": float(np.min(values)),
            "semi_max": float(np.max(values)),
        }

    return summary


def _correlation_mds_positions(correlation: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Produce initial two-dimensional positions from absolute correlations.

    Stronger absolute correlations result in smaller distances. These are
    initial positions only; collision removal is applied afterwards.
    """
    variables = list(correlation.columns)
    matrix = correlation.loc[variables, variables].to_numpy(dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.clip(matrix, -1.0, 1.0)
    distance = 1.0 - np.abs(matrix)
    np.fill_diagonal(distance, 0.0)
    n = len(variables)
    centering = np.eye(n) - np.ones((n, n), dtype=float) / n
    gram = -0.5 * centering @ np.square(distance) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive_values = np.clip(eigenvalues[:2], 0.0, None)
    coordinates = eigenvectors[:, :2] * np.sqrt(positive_values)
    if coordinates.shape[1] < 2:
        coordinates = np.pad(coordinates, ((0, 0), (0, 2 - coordinates.shape[1])))
    scale = float(np.max(np.linalg.norm(coordinates, axis=1)))
    if not np.isfinite(scale) or scale < 1e-8:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        coordinates = np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        coordinates = coordinates / scale
    coordinates *= 2.3
    return {
        variable: coordinates[index].astype(float)
        for index, variable in enumerate(variables)
    }


def _relax_node_positions(
    initial_positions: dict[str, np.ndarray],
    radii: dict[str, float],
    minimum_gap: float = 0.30,
    iterations: int = 500,
) -> dict[str, tuple[float, float]]:
    """
    Move overlapping circles apart while preserving the MDS layout.
    """
    names = list(initial_positions)
    original = {
        name: np.asarray(initial_positions[name], dtype=float).copy()
        for name in names
    }
    current = {name: position.copy() for name, position in original.items()}

    for _ in range(iterations):
        movement = {name: np.zeros(2, dtype=float) for name in names}
        maximum_overlap = 0.0

        for first_index, first_name in enumerate(names):
            for second_name in names[first_index + 1:]:
                first_position = current[first_name]
                second_position = current[second_name]
                difference = second_position - first_position
                distance = float(np.linalg.norm(difference))
                required_distance = radii[first_name] + radii[second_name] + minimum_gap
                overlap = required_distance - distance
                if overlap <= 0:
                    continue

                maximum_overlap = max(maximum_overlap, overlap)
                if distance < 1e-8:
                    deterministic_angle = (names.index(first_name) + names.index(second_name)) * 1.618
                    direction = np.array(
                        [np.cos(deterministic_angle), np.sin(deterministic_angle)],
                        dtype=float,
                    )
                else:
                    direction = difference / distance

                displacement = direction * overlap * 0.52
                movement[first_name] -= displacement
                movement[second_name] += displacement

        for name in names:
            movement[name] += (original[name] - current[name]) * 0.012

        for name in names:
            step = movement[name]
            step_length = float(np.linalg.norm(step))
            if step_length > 0.18:
                step = step / step_length * 0.18
            current[name] += step

        center = np.mean(np.stack(list(current.values())), axis=0)
        for name in names:
            current[name] -= center

        if maximum_overlap < 0.01:
            break

    return {
        name: (float(current[name][0]), float(current[name][1]))
        for name in names
    }


def figure1_summary_table(dataframe=None, variables=None, variance_iterations=1000) -> list[dict]:
    df = _load_data() if dataframe is None else dataframe
    predictors = [c for c in (variables or PREDICTORS) if c in df.columns]
    z_scores = _zscore(df, predictors).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    betas = _standardized_linear_betas(df, predictors)
    variance_summary = _sampled_variance_summary(
        z_scores=z_scores,
        elite_mask=_elite_mask(df),
        predictors=predictors,
        iterations=variance_iterations,
        sample_size=22,
        seed=42,
    )
    rows = []
    for predictor in predictors:
        info = variance_summary[predictor]
        rows.append(
            {
                "variable": DISPLAY_NAMES.get(predictor, predictor.replace("_", " ").title()),
                "beta": float(betas.get(predictor, 0.0)),
                "elite_variance": float(info["elite"]),
                "semi_elite_variance_mean": float(info["semi_mean"]),
                "semi_elite_variance_min": float(info["semi_min"]),
                "semi_elite_variance_max": float(info["semi_max"]),
            }
        )
    return rows


def _draw_variance_glyph(
    ax,
    x: float,
    y: float,
    radius: float,
    variance_info: dict[str, float],
    global_maximum: float,
) -> None:
    """
    Draw a compact variance comparison directly inside a node.
    """
    if global_maximum <= 0 or not np.isfinite(global_maximum):
        global_maximum = 1.0

    elite = max(0.0, float(variance_info["elite"]))
    semi_mean = max(0.0, float(variance_info["semi_mean"]))
    semi_min = max(0.0, float(variance_info["semi_min"]))
    semi_max = max(0.0, float(variance_info["semi_max"]))

    chart_width = radius * 1.15
    chart_height = radius * 0.78
    axis_left_x = x - chart_width * 0.44
    axis_right_x = x + chart_width * 0.45
    baseline_y = y - chart_height * 0.34
    maximum_y = y + chart_height * 0.40
    usable_height = maximum_y - baseline_y

    def scale_value(value: float) -> float:
        ratio = np.clip(value / global_maximum, 0.0, 1.0)
        return baseline_y + ratio * usable_height

    elite_center_x = x - chart_width * 0.15
    semi_center_x = x + chart_width * 0.22
    bar_width = chart_width * 0.18
    elite_top = scale_value(elite)
    semi_top = scale_value(semi_mean)

    ax.plot(
        [axis_left_x, axis_right_x],
        [baseline_y, baseline_y],
        color="#444444",
        linewidth=0.75,
        zorder=6,
    )
    ax.plot(
        [axis_left_x, axis_left_x],
        [baseline_y, maximum_y],
        color="#444444",
        linewidth=0.75,
        zorder=6,
    )
    ax.plot(
        [axis_left_x - chart_width * 0.035, axis_left_x],
        [maximum_y, maximum_y],
        color="#444444",
        linewidth=0.65,
        zorder=6,
    )
    ax.text(
        axis_right_x + radius * 0.025,
        baseline_y - radius * 0.015,
        "x",
        fontsize=5.3,
        ha="left",
        va="top",
        color="#444444",
        zorder=7,
    )
    ax.text(
        axis_left_x - radius * 0.035,
        maximum_y + radius * 0.02,
        "y",
        fontsize=5.3,
        ha="right",
        va="bottom",
        color="#444444",
        zorder=7,
    )

    ax.add_patch(
        Rectangle(
            (elite_center_x - bar_width / 2, baseline_y),
            bar_width,
            max(elite_top - baseline_y, 0.006),
            facecolor="#e57373",
            edgecolor="none",
            alpha=0.88,
            zorder=5,
        )
    )
    ax.add_patch(
        Rectangle(
            (semi_center_x - bar_width / 2, baseline_y),
            bar_width,
            max(semi_top - baseline_y, 0.006),
            facecolor="#70c5c8",
            edgecolor="none",
            alpha=0.82,
            zorder=5,
        )
    )
    ax.vlines(
        semi_center_x,
        scale_value(semi_min),
        scale_value(semi_max),
        color="#19777a",
        linestyle="--",
        linewidth=1.2,
        zorder=6,
    )
    ax.text(elite_center_x, baseline_y - radius * 0.055, "E", fontsize=5.8, ha="center", va="top", zorder=7)
    ax.text(semi_center_x, baseline_y - radius * 0.055, "S", fontsize=5.8, ha="center", va="top", zorder=7)


def create_figure1(
    dataframe=None,
    variables=None,
    correlation_threshold=0.15,
    variance_iterations=1000,
    *,
    precomputed_components=None,
    figure_title=None,
):
    """
    Create a readable Nature-style Figure 1.

    Visual encoding:
    - node size: magnitude of standardized regression beta
    - node distance: absolute correlation after MDS and collision removal
    - edge width: absolute correlation
    - edge style/color: positive or negative correlation
    - internal glyph: elite variance and sampled semi-elite variance

    No raw athlete rows are exposed.
    """
    if precomputed_components is None:
        df = _load_data() if dataframe is None else dataframe
        predictors = [column for column in (variables or PREDICTORS) if column in df.columns]

        if len(predictors) < 2:
            raise ValueError("Not enough predictor columns to create Figure 1.")

        elite_mask = _elite_mask(df)
        z_scores = _zscore(df, predictors)
        z_scores = z_scores.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        correlation = z_scores[predictors].corr()
        betas = _standardized_linear_betas(df, predictors)
        variance_summary = _sampled_variance_summary(
            z_scores=z_scores,
            elite_mask=elite_mask,
            predictors=predictors,
            iterations=variance_iterations,
            sample_size=22,
            seed=42,
        )
    else:
        predictors = list(variables or precomputed_components.get("variables") or PREDICTORS)
        beta_values = precomputed_components["standardized_betas"]
        if isinstance(beta_values, dict):
            betas = {name: float(beta_values[name]) for name in predictors}
        else:
            betas = {
                name: float(value)
                for name, value in zip(predictors, np.asarray(beta_values, dtype=float))
            }
        correlation_values = precomputed_components["correlation_matrix"]
        correlation = pd.DataFrame(
            np.asarray(correlation_values, dtype=float),
            index=predictors,
            columns=predictors,
        )
        variance_summary = {
            name: {
                key: float(value)
                for key, value in precomputed_components["variance_summary"][name].items()
            }
            for name in predictors
        }

    maximum_absolute_beta = max(
        [abs(float(betas.get(variable, 0.0))) for variable in predictors] + [0.001]
    )
    radii = {}
    for variable in predictors:
        beta_ratio = abs(float(betas.get(variable, 0.0))) / maximum_absolute_beta
        radii[variable] = 0.56 + 0.22 * np.sqrt(beta_ratio)

    initial_positions = _correlation_mds_positions(correlation)
    positions = _relax_node_positions(
        initial_positions=initial_positions,
        radii=radii,
        minimum_gap=0.86,
        iterations=1200,
    )

    maximum_variance = max(
        [
            max(float(info["elite"]), float(info["semi_mean"]), float(info["semi_max"]))
            for info in variance_summary.values()
        ]
        + [1.0]
    )

    fig, ax = plt.subplots(figsize=(18, 14), dpi=140)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_title(
        figure_title
        or (
            "Nature-style Figure 1: group-level statistics\n"
            "Regression effects, predictor correlations, and elite vs semi-elite variance"
        ),
        fontsize=16,
        pad=25,
    )

    for first_index, first_variable in enumerate(predictors):
        for second_variable in predictors[first_index + 1:]:
            r = float(correlation.loc[first_variable, second_variable])
            if not np.isfinite(r) or abs(r) <= correlation_threshold:
                continue

            first_x, first_y = positions[first_variable]
            second_x, second_y = positions[second_variable]
            line_width = 0.8 + 5.0 * abs(r)
            line_color = "#303030" if r >= 0 else "#a84a4a"
            line_style = "-" if r >= 0 else "--"
            ax.plot(
                [first_x, second_x],
                [first_y, second_y],
                color=line_color,
                linestyle=line_style,
                linewidth=line_width,
                alpha=0.62,
                zorder=1,
            )
            midpoint_x = (first_x + second_x) / 2
            midpoint_y = (first_y + second_y) / 2
            edge_vector = np.array([second_x - first_x, second_y - first_y], dtype=float)
            edge_length = float(np.linalg.norm(edge_vector))
            if edge_length > 1e-8:
                perpendicular = np.array([-edge_vector[1], edge_vector[0]], dtype=float) / edge_length
                label_offset = perpendicular * 0.13
                midpoint_x += float(label_offset[0])
                midpoint_y += float(label_offset[1])
            if abs(r) >= 0.28:
                ax.text(
                    midpoint_x,
                    midpoint_y,
                    f"r = {r:.2f}",
                    fontsize=6.5,
                    ha="center",
                    va="center",
                    bbox={
                        "boxstyle": "round,pad=0.16",
                        "facecolor": "white",
                        "edgecolor": "#dddddd",
                        "linewidth": 0.45,
                        "alpha": 0.92,
                    },
                    zorder=3,
                )

    for predictor in predictors:
        x, y = positions[predictor]
        beta = float(betas.get(predictor, 0.0))
        radius = radii[predictor]
        node = Circle(
            (x, y),
            radius=radius,
            facecolor="#fcfcfc",
            edgecolor="#202020",
            linewidth=1.6,
            zorder=4,
        )
        ax.add_patch(node)

        _draw_variance_glyph(
            ax=ax,
            x=x,
            y=y,
            radius=radius,
            variance_info=variance_summary[predictor],
            global_maximum=maximum_variance,
        )

        label = DISPLAY_NAMES.get(predictor, predictor.replace("_", " ").title())
        label_direction = np.array([x, y], dtype=float)
        direction_length = float(np.linalg.norm(label_direction))
        if direction_length < 1e-8:
            label_direction = np.array([0.0, 1.0], dtype=float)
        else:
            label_direction = label_direction / direction_length
        label_x = x + float(label_direction[0]) * (radius + 0.28)
        label_y = y + float(label_direction[1]) * (radius + 0.28)
        if abs(float(label_direction[0])) < 0.35:
            horizontal_alignment = "center"
        elif label_direction[0] > 0:
            horizontal_alignment = "left"
        else:
            horizontal_alignment = "right"
        if abs(float(label_direction[1])) < 0.35:
            vertical_alignment = "center"
        elif label_direction[1] > 0:
            vertical_alignment = "bottom"
        else:
            vertical_alignment = "top"
        ax.text(
            label_x,
            label_y,
            f"{label}\nbeta = {beta:.2f}",
            fontsize=8.5,
            fontweight="medium",
            ha=horizontal_alignment,
            va=vertical_alignment,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.92,
            },
            zorder=8,
        )

    all_x = np.array([positions[name][0] for name in predictors], dtype=float)
    all_y = np.array([positions[name][1] for name in predictors], dtype=float)
    maximum_radius = max(radii.values())
    horizontal_margin = maximum_radius + 2.15
    vertical_margin = maximum_radius + 2.35
    ax.set_xlim(float(all_x.min() - horizontal_margin), float(all_x.max() + horizontal_margin))
    ax.set_ylim(float(all_y.min() - vertical_margin), float(all_y.max() + vertical_margin))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("MDS dimension 1: correlation-derived predictor similarity", fontsize=11, labelpad=12)
    ax.set_ylabel("MDS dimension 2: correlation-derived predictor similarity", fontsize=11, labelpad=12)
    ax.tick_params(axis="both", labelsize=9, colors="#555555")
    for spine in ax.spines.values():
        spine.set_color("#bbbbbb")
        spine.set_linewidth(0.8)
    ax.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)

    legend_handles = [
        Line2D([0], [0], color="#303030", linewidth=2.2, linestyle="-", label="Positive correlation"),
        Line2D([0], [0], color="#a84a4a", linewidth=2.2, linestyle="--", label="Negative correlation"),
        Rectangle((0, 0), 1, 1, facecolor="#e57373", edgecolor="none", label="Elite variance"),
        Rectangle((0, 0), 1, 1, facecolor="#70c5c8", edgecolor="none", label="Semi-elite sampled mean variance"),
        Line2D([0], [0], color="#19777a", linewidth=1.2, linestyle="--", label="Semi-elite sampled variance range"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )

    fig.text(
        0.5,
        0.025,
        "Node size represents |standardized beta|. Node proximity represents stronger absolute correlation. "
        f"Only correlations with |r| > {correlation_threshold:g} are shown. E = elite; S = semi-elite.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.20)

    return fig


def create_figure1_from_components(
    *,
    standardized_betas,
    correlation_matrix,
    variance_summary,
    variables=None,
    correlation_threshold=0.15,
    title="Mean Perturbed Figure 1",
):
    """Render a Figure 1 network from already aggregated statistical components.

    This deliberately delegates to :func:`create_figure1`, so the mean-perturbed
    network uses exactly the same node, edge, layout, and variance-glyph encoding
    as the original dashboard figure.
    """
    selected_variables = list(variables or PREDICTORS)
    return create_figure1(
        variables=selected_variables,
        correlation_threshold=correlation_threshold,
        precomputed_components={
            "variables": selected_variables,
            "standardized_betas": standardized_betas,
            "correlation_matrix": correlation_matrix,
            "variance_summary": variance_summary,
        },
        figure_title=title,
    )


def create_network_plot(group: str = "all"):
    """
    Protected network plot of the eight standardized athlete domains.

    Nodes are the domain scores. Edges are aggregate pairwise correlations.
    No athlete rows or raw measurements are shown.
    """
    group = _normalize_filter_text(group or "all")
    df = _filter_group_for_plot(_load_data(), group)

    predictors = [c for c in PREDICTORS if c in df.columns]
    if len(df) < 3:
        raise ValueError("Not enough protected rows for a network plot.")

    z = _zscore(df, predictors)
    corr = z.corr()
    n = len(predictors)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    layout_radius = 1.25
    positions = {
        predictor: layout_radius * np.array([np.cos(angle), np.sin(angle)])
        for predictor, angle in zip(predictors, angles)
    }

    fig, ax = plt.subplots(figsize=(11, 9.5))
    ax.set_title(
        f"Protected correlation network ({group})\n"
        "Edges show aggregate correlations among standardized domains",
        fontsize=14,
        pad=18,
    )

    for i, p1 in enumerate(predictors):
        for j, p2 in enumerate(predictors):
            if j <= i:
                continue
            r = float(corr.loc[p1, p2])
            if np.isnan(r) or abs(r) < 0.15:
                continue
            x1, y1 = positions[p1]
            x2, y2 = positions[p2]
            color = "#2f6f9f" if r >= 0 else "#b65f5f"
            ax.plot(
                [x1, x2],
                [y1, y2],
                linewidth=1.0 + 4.0 * abs(r),
                color=color,
                alpha=min(0.25 + abs(r), 0.85),
                zorder=1,
            )
            ax.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                f"{r:.2f}",
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.75),
                zorder=3,
            )

    for predictor in predictors:
        x, y = positions[predictor]
        ax.scatter(x, y, s=2600, color="#f2f2f2", edgecolor="#222222", linewidth=1.4, zorder=4)
        ax.text(
            x,
            y,
            _figure_label(predictor),
            fontsize=8.5,
            ha="center",
            va="center",
            zorder=5,
        )

    ax.text(-1.55, -1.52, "Blue = positive r; red = negative r; labels show aggregate r", fontsize=9)
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.65, 1.65)
    ax.axis("off")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def create_variance_plot(iterations: int = 1000, sample_size: int = 22, seed: int = 42, dataframe=None):
    """
    Protected elite vs semi-elite variance plot for the eight domains.

    The plot shows aggregate variance summaries only.
    """
    df = _load_data() if dataframe is None else dataframe.copy()
    predictors = [c for c in PREDICTORS if c in df.columns]
    elite = df[_elite_mask(df)].copy()
    semi = df[~_elite_mask(df)].copy()
    if elite.empty or semi.empty:
        raise ValueError("Elite or semi-elite group is empty.")

    sample_size = min(sample_size, len(elite), len(semi))
    rng = np.random.default_rng(seed)
    elite_var = elite[predictors].head(sample_size).var(ddof=1)
    sampled = {p: [] for p in predictors}
    for _ in range(iterations):
        draw = semi.sample(n=sample_size, replace=False, random_state=int(rng.integers(0, 1_000_000)))
        for predictor in predictors:
            sampled[predictor].append(float(draw[predictor].var(ddof=1)))

    semi_mean = np.array([np.mean(sampled[p]) for p in predictors])
    semi_min = np.array([np.min(sampled[p]) for p in predictors])
    semi_max = np.array([np.max(sampled[p]) for p in predictors])
    elite_values = elite_var[predictors].to_numpy(dtype=float)
    x = np.arange(len(predictors))

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.bar(x - 0.18, elite_values, width=0.36, label="Elite variance", color="#2f6f9f", alpha=0.88)
    ax.bar(x + 0.18, semi_mean, width=0.36, label="Semi-elite sampled mean", color="#8a8a8a", alpha=0.62)
    ax.vlines(x + 0.18, semi_min, semi_max, color="#444444", linewidth=1.4, alpha=0.7, label="Semi-elite sampled range")
    ax.set_title(
        "Protected variance comparison\n"
        "Elite variance vs sampled semi-elite variance across standardized domains",
        fontsize=14,
        pad=16,
    )
    ax.set_ylabel("Variance of standardized domain score")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [DISPLAY_NAMES.get(p, p.replace("_", " ").title()) for p in predictors],
        rotation=35,
        ha="right",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def create_figure2(
    group: str = "all",
    max_athletes: int | None = None,
    selected_indexes: list | None = None,
    dataframe=None,
    selected_dataframe=None,
    reference_dataframe=None,
    standardized_profiles=None,
    reference_profile=None,
    reference_label: str = "Selected Cohort Mean Profile",
    variables=None,
    group_label: str | None = None,
):
    """
    Large Figure 2-style z-score profile plot.

    Parameters
    ----------
    group:
        "all", "elite", or "semi_elite". Defaults to "all".
    max_athletes:
        Number of athletes to display. If None, show all selected athletes.
    """
    df = _load_data() if dataframe is None else dataframe.copy()
    predictors = [c for c in (variables or PREDICTORS) if c in df.columns]

    group = (group or "all").lower()

    if selected_dataframe is not None:
        selected = selected_dataframe.copy()
        group_label = group_label or "Selected cohort"
    elif group == "elite":
        selected = df[_elite_mask(df)].copy()
        group_label = "Elite athletes"
    elif group in {"semi_elite", "semi-elite", "semi"}:
        selected = df[~_elite_mask(df)].copy()
        group_label = "Semi-elite athletes"
    else:
        selected = df.copy()
        group_label = "All athletes"

    if selected.empty:
        raise ValueError(f"No athletes found for group: {group}")

    total_athletes = len(df) if selected_dataframe is not None else len(selected)

    if selected_indexes is not None:
        requested_indexes = [index for index in selected_indexes if index in selected.index]
        selected = selected.loc[requested_indexes].copy()
    elif max_athletes is not None and selected_dataframe is None:
        selected = selected.sample(
            n=min(int(max_athletes), len(selected)), random_state=2024
        )

    if standardized_profiles is not None:
        z_selected=standardized_profiles.loc[selected.index,predictors].copy()
    else:
        reference = df if reference_dataframe is None else reference_dataframe
        z_all = (df[predictors] - reference[predictors].mean()) / reference[predictors].std(ddof=0).replace(0, 1)
        z_selected = z_all.loc[selected.index, predictors]
    z_selected = z_selected.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    shown_athletes = len(z_selected)

    x = np.arange(len(predictors))

    fig_width = 15
    fig_height = min(24, max(8, 0.22 * shown_athletes + 6))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if shown_athletes <= 30:
        alpha = 0.45
        linewidth = 1.2
    elif shown_athletes <= 100:
        alpha = 0.22
        linewidth = 0.9
    else:
        alpha = 0.12
        linewidth = 0.7

    mean_series=z_selected.mean(axis=0) if reference_profile is None else reference_profile
    group_mean=mean_series.loc[predictors].values.astype(float)
    displayed_z_scores = z_selected.copy(deep=True)

    for _, row in displayed_z_scores.iterrows():
        ax.plot(
            x,
            row.values.astype(float),
            marker="o" if shown_athletes <= 60 else None,
            linewidth=linewidth,
            alpha=alpha,
            label=None,
        )

    ax.plot(
        x,
        group_mean,
        linestyle="--",
        linewidth=2.2,
        marker="o",
        markersize=4,
        color="black",
        label=reference_label,
        zorder=10,
    )

    ax.axhline(0, linestyle="-", linewidth=1.0, color="black", alpha=0.45)
    ax.axhline(1, linestyle=":", linewidth=1.0, color="black", alpha=0.35)
    ax.axhline(-1, linestyle=":", linewidth=1.0, color="black", alpha=0.35)

    if max_athletes is None:
        showing_text = f"Showing all {total_athletes} athletes"
    else:
        showing_text = f"Showing {shown_athletes} of {total_athletes} athletes"

    ax.set_title(
        f"Figure 2-style z-score profiles: {group_label}\n"
        f"{showing_text}",
        fontsize=15,
        pad=18,
    )

    ax.set_ylabel("z-score")
    ax.set_xlabel("Predictor variables")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [DISPLAY_NAMES.get(p, p.replace("_", " ").title()) for p in predictors],
        rotation=35,
        ha="right",
    )

    ax.set_ylim(
        min(-3.2, float(np.nanmin(displayed_z_scores.values)) - 0.4),
        max(3.2, float(np.nanmax(displayed_z_scores.values)) + 0.4),
    )

    ax.grid(axis="y", alpha=0.25)

    if shown_athletes <= 30:
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    else:
        ax.legend(loc="upper right", fontsize=8)

    fig.text(.5,.015,"The dashed line represents the mean z-score profile of the currently selected cohort.",
        ha="center",fontsize=8,color="#475569")
    fig.tight_layout(rect=(0,.04,1,1))
    return fig


def generate_individual_profile_line_figure(
    profile_rows: list[dict], *, profile_label: str, elite_mean_profile: list[float]
):
    """Render one privacy-safe eight-domain standardized profile."""
    by_key={row.get("domain_key"):row for row in profile_rows if isinstance(row,dict)}
    missing=[domain for domain in DOMAIN_ORDER if domain not in by_key]
    if missing:raise ValueError("Cannot draw individual profile. Missing domains: "+", ".join(missing))
    values=[]
    for domain in DOMAIN_ORDER:
        value=(by_key.get(domain) or {}).get("z_score")
        try:values.append(float(value) if value is not None else np.nan)
        except (TypeError,ValueError):values.append(np.nan)
    if len(elite_mean_profile)!=len(DOMAIN_ORDER):
        raise ValueError("Elite mean profile must contain exactly eight domains.")
    elite_values=np.asarray(elite_mean_profile,dtype=float)
    x=np.arange(len(DOMAIN_ORDER));fig,ax=plt.subplots(figsize=(12,5.8))
    colors=["#eef4fb","#f8eeee","#f5f0ea","#eef7ef","#eef7ef","#f8eeee","#f8eeee","#f5f0ea"]
    for index,color in enumerate(colors):ax.axvspan(index-.5,index+.5,color=color,alpha=.75,zorder=0)
    ax.axhline(y=0.0,color="black",linestyle="-",linewidth=1.2,alpha=.8,
        label="Overall Mean (z = 0)",zorder=1)
    ax.plot(x,elite_values,color="#111827",linestyle="--",linewidth=1.5,label="Elite Mean Profile",zorder=2)
    ax.plot(x,values,color="#c43d4d",linewidth=2.0,marker="o",markersize=5,label=profile_label,zorder=3)
    ax.axhline(1,color="#64748b",linestyle=":",linewidth=.8,alpha=.5,zorder=1)
    ax.axhline(-1,color="#64748b",linestyle=":",linewidth=.8,alpha=.5,zorder=1)
    ax.set_ylim(-3,3);ax.set_yticks(range(-3,4));ax.set_ylabel("z-score")
    ax.set_xticks(x);ax.set_xticklabels([DOMAIN_LABELS[key] for key in DOMAIN_ORDER],fontsize=8)
    ax.set_title("Individual Standardized Profile\nAnonymous Profile vs Elite Mean Profile",fontsize=13,pad=12)
    ax.grid(axis="y",alpha=.18);ax.spines[["top","right"]].set_visible(False)
    ax.legend(loc="upper right",frameon=False,fontsize=9)
    fig.text(.5,.015,"Positive values are above the full-sample average; negative values are below the full-sample average.",ha="center",fontsize=8,color="#475569")
    fig.tight_layout(rect=(0,.06,1,1));return fig
