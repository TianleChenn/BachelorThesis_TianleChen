from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover
    sm = None

try:
    from scipy.stats import chi2, jarque_bera, pearsonr
except Exception:  # pragma: no cover
    chi2 = None
    jarque_bera = None
    pearsonr = None

try:
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.stattools import durbin_watson
except Exception:  # pragma: no cover
    het_breuschpagan = None
    variance_inflation_factor = None
    durbin_watson = None

from .config import DATA_PATH, DISPLAY_NAMES, PREDICTORS

PAPER_PATTERN_DOMAINS = [
    "basic_cognitive_function",
    "lower_body_dynamics",
    "blood_micronutrients",
]
FIGURE2_RANDOM_SEED = 2024


def _display(name: str) -> str:
    return DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def _safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _with_rows(result: dict, rows: list[dict]) -> dict:
    """Expose both rows and table for compatibility with the UI/tests."""
    result["rows"] = rows
    result["table"] = rows
    return result


def _with_analysis_context(
    result: dict,
    *,
    cohort_size: int | None = None,
    group: str | None = None,
    dataframe=None,
) -> dict:
    if cohort_size is not None:
        result["cohort_size"] = int(cohort_size)
    if group is not None:
        result["group"] = group
    return result


def load_data() -> pd.DataFrame:
    path = Path(DATA_PATH)
    if not path.exists():
        raise FileNotFoundError(
            "Synthetic athlete dataset not found. "
            "Run: python data/generate_synthetic_athlete_data.py"
        )
    df = pd.read_csv(path)
    required_metadata = [
        "athlete_id",
        "age",
        "sex",
        "sport",
        "national_team",
        "expertise_value",
        "elite_status",
    ]
    missing_metadata = [c for c in required_metadata if c not in df.columns]
    missing_predictors = [c for c in PREDICTORS if c not in df.columns]
    if missing_metadata or missing_predictors:
        missing = missing_metadata + missing_predictors
        raise ValueError(f"Synthetic athlete dataset is missing required columns: {missing}")
    return df


def elite_mask(df: pd.DataFrame) -> pd.Series:
    if "elite_status" in df.columns:
        col = df["elite_status"]
        if pd.api.types.is_numeric_dtype(col):
            return col.astype(int) == 1
        return col.astype(str).str.lower().isin(["elite", "1", "true", "yes", "elite-like"])
    return df["expertise_value"] >= 13


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


def filter_group(df: pd.DataFrame, group: str = "all") -> pd.DataFrame:
    group = _normalize_filter_text(group or "all")
    df = _ensure_national_team_column(df)

    if group in {"all", "full", "complete", "all_athletes"}:
        return df.copy()

    mask = elite_mask(df)
    if group in {"elite", "top", "super"}:
        return df[mask].copy()
    if group in {"semi_elite", "semi-elite", "semi", "non_elite", "lower"}:
        return df[~mask].copy()

    if ":" not in group:
        return df.copy()

    category, value = group.split(":", 1)
    value = _normalize_filter_text(value)

    if category == "sport" and "sport" in df.columns:
        sport = df["sport"].map(_normalize_filter_text)
        return df[sport == value].copy()

    if category == "sex" and "sex" in df.columns:
        sex = df["sex"].map(_normalize_filter_text)
        return df[sex == value].copy()

    if category == "national_team" and "national_team" in df.columns:
        team = df["national_team"].map(_normalize_filter_text)
        return df[team == value].copy()

    if category == "age_group" and "age" in df.columns:
        age = pd.to_numeric(df["age"], errors="coerce")
        if value in {"under_20", "u20", "junior"}:
            return df[age < 20].copy()
        if value in {"20_and_above", "20_plus", "above_20", "senior"}:
            return df[age >= 20].copy()

    return df.copy()


def _design_matrix(
    df: pd.DataFrame,
    controls: list[str] | None = None,
    *,
    standardize_age: bool = False,
) -> pd.DataFrame:
    controls = controls or []
    X = df[PREDICTORS].copy()
    if "age" in controls and "age" in df.columns:
        age = pd.to_numeric(df["age"], errors="coerce")
        if standardize_age:
            age = (age - age.mean()) / age.std(ddof=0) if age.std(ddof=0) else age * 0
        X["age"] = age
    if "sex" in controls and "sex" in df.columns:
        X["sex_female"] = df["sex"].astype(str).str.lower().isin(["female", "f", "1"]).astype(int)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X.fillna(X.mean(numeric_only=True))


def _table1_model_specs() -> list[tuple[str, list[str]]]:
    return [
        ("a) 8 predictors", []),
        ("b) 8 predictors + sex", ["sex"]),
        ("c) 8 predictors + age", ["age"]),
        ("d) 8 predictors + age + sex", ["age", "sex"]),
    ]


def _table2_model_specs() -> list[tuple[str, list[str]]]:
    return [
        ("a) 8 predictors", []),
        ("b) 8 predictors + age", ["age"]),
        ("c) 8 predictors + sex", ["sex"]),
        ("d) 8 predictors + age + sex", ["age", "sex"]),
    ]


def _nagelkerke_r_squared(ll_full: float, ll_null: float, n: int) -> float:
    if n <= 0:
        return float("nan")
    cox_snell = 1.0 - np.exp((2.0 / n) * (ll_null - ll_full))
    maximum = 1.0 - np.exp((2.0 / n) * ll_null)
    if maximum <= 0:
        return float("nan")
    return float(cox_snell / maximum)


def _classification_accuracy(model, design_matrix: pd.DataFrame, outcome: pd.Series) -> float:
    probabilities = np.asarray(model.predict(design_matrix), dtype=float)
    predictions = (probabilities >= 0.5).astype(int)
    return float(np.mean(predictions == outcome.to_numpy(dtype=int)))


def _hc4_inference(model):
    from scipy.stats import t as student_t

    design = np.asarray(model.model.exog, dtype=float)
    residuals = np.asarray(model.resid, dtype=float)
    n, p = design.shape
    influence = model.get_influence()
    leverage = np.asarray(influence.hat_matrix_diag, dtype=float)
    delta = np.minimum(4.0, n * leverage / max(p, 1))
    denominator = np.power(np.clip(1.0 - leverage, 1e-8, None), delta)
    omega = (residuals ** 2) / denominator
    bread = np.linalg.pinv(design.T @ design)
    meat = design.T @ (omega[:, None] * design)
    covariance = bread @ meat @ bread
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
    parameters = np.asarray(model.params, dtype=float)
    t_values = np.divide(
        parameters,
        standard_errors,
        out=np.full_like(parameters, np.nan),
        where=standard_errors > 0,
    )
    degrees_of_freedom = max(n - p, 1)
    p_values = 2.0 * student_t.sf(np.abs(t_values), df=degrees_of_freedom)
    return {
        "covariance": covariance,
        "standard_errors": standard_errors,
        "t_values": t_values,
        "p_values": p_values,
        "type": "HC4",
    }


def table1_logistic_regressions(group: str = "all", dataframe=None) -> dict:
    """Nature Table 1-style logistic regression.

    Logistic classification requires both elite and semi-elite cases, so the
    function accepts group for compatibility but runs the binary model on all
    protected rows.
    """
    df = dataframe.copy() if dataframe is not None else load_data()
    y = elite_mask(df).astype(int)
    rows: list[dict] = []
    model_stats: list[dict] = []

    for model_name, controls in _table1_model_specs():
        X = _design_matrix(df, controls)
        if sm is None:
            rows.append({"model": model_name, "variable": "statsmodels unavailable"})
            continue
        try:
            X_const = sm.add_constant(X, has_constant="add")
            model = sm.Logit(y, X_const).fit(disp=False, maxiter=300)
            converged = bool(getattr(model, "mle_retvals", {}).get("converged", True))
            if not converged:
                raise ValueError("Logistic model did not converge.")

            params = model.params
            bse = getattr(model, "bse", pd.Series([np.nan] * len(params), index=params.index))
            pvalues = getattr(model, "pvalues", pd.Series([np.nan] * len(params), index=params.index))
            try:
                conf = model.conf_int()
            except Exception:
                conf = pd.DataFrame({0: [np.nan] * len(params), 1: [np.nan] * len(params)}, index=params.index)

            for var in params.index:
                se = bse[var]
                b = params[var]
                wald = (b / se) ** 2 if se not in [0, None] and not pd.isna(se) else np.nan
                odds_ratio = np.exp(b)
                or_ci_low = np.exp(conf.loc[var, 0])
                or_ci_high = np.exp(conf.loc[var, 1])
                rows.append(
                    {
                        "model": model_name,
                        "variable": "Constant" if var == "const" else _display(var),
                        "B": _safe_float(b),
                        "SE": _safe_float(se),
                        "Wald": _safe_float(wald),
                        "p": _safe_float(pvalues[var]),
                        "odds_ratio": _safe_float(odds_ratio),
                        "or_ci_low": _safe_float(or_ci_low),
                        "or_ci_high": _safe_float(or_ci_high),
                        "OR 95% CI": f"[{float(or_ci_low)}, {float(or_ci_high)}]",
                    }
                )
            null_model = sm.Logit(y, np.ones((len(y), 1))).fit(disp=False, maxiter=300)
            ll_full = float(model.llf)
            ll_null = float(null_model.llf)
            model_chi_square = 2.0 * (ll_full - ll_null)
            model_df = int(max(len(params) - 1, 0))
            model_p_value = chi2.sf(model_chi_square, model_df) if chi2 is not None and model_df > 0 else np.nan
            model_stats.append(
                {
                    "model": model_name,
                    "n": int(len(df)),
                    "model_chi_square": _safe_float(model_chi_square),
                    "model_degrees_of_freedom": model_df,
                    "model_p_value": _safe_float(model_p_value),
                    "nagelkerke_r_squared": _safe_float(_nagelkerke_r_squared(ll_full, ll_null, len(df))),
                    "classification_accuracy": _safe_float(_classification_accuracy(model, X_const, y)),
                    "log_likelihood": _safe_float(ll_full),
                    "null_log_likelihood": _safe_float(ll_null),
                    "converged": converged,
                    "privacy": "Aggregate model statistics only",
                }
            )
        except Exception as exc:
            rows.append({"model": model_name, "variable": "model_error", "error": "Logistic model failed safely."})
            model_stats.append(
                {
                    "model": model_name,
                    "n": int(len(df)),
                    "converged": False,
                    "error": str(exc),
                    "privacy": "No inferential statistics were fabricated for this failed model.",
                }
            )

    return _with_rows(
        {
            "analysis": "Table 1-style logistic regression",
            "cohort_size": int(len(df)),
            "model_stats": model_stats,
            "privacy_note": "Executed locally. Only aggregate coefficients are returned; raw athlete rows are hidden.",
        },
        rows,
    )


def fit_table2_models_raw(dataframe: pd.DataFrame, *, group: str = "all") -> dict:
    """Fit the four Table 2 specifications and retain full-precision results."""
    df = dataframe.copy(deep=True)
    result = {"group": group, "n": int(len(df)), "cohort_size": int(len(df)), "models": []}
    if len(df) < 12 or sm is None:
        return result

    y = pd.to_numeric(df["expertise_value"], errors="coerce")
    for model_name, controls in _table2_model_specs():
        X = _design_matrix(df, controls, standardize_age=True)
        valid = y.notna()
        X, y_valid = X.loc[valid], y.loc[valid]
        X_const = sm.add_constant(X, has_constant="add")
        base_model = sm.OLS(y_valid, X_const).fit()
        jb_stat, jb_p = (np.nan, np.nan)
        if jarque_bera is not None:
            jb_result = jarque_bera(base_model.resid)
            jb_stat = float(jb_result.statistic if hasattr(jb_result, "statistic") else jb_result[0])
            jb_p = float(jb_result.pvalue if hasattr(jb_result, "pvalue") else jb_result[1])
        robust_type = "HC4" if not pd.isna(jb_p) and jb_p < 0.05 else "HC3"
        if robust_type == "HC4":
            inference = _hc4_inference(base_model)
            bse = pd.Series(inference["standard_errors"], index=base_model.params.index)
            tvalues = pd.Series(inference["t_values"], index=base_model.params.index)
            pvalues = pd.Series(inference["p_values"], index=base_model.params.index)
        else:
            robust_model = base_model.get_robustcov_results(cov_type="HC3")
            bse = pd.Series(robust_model.bse, index=base_model.params.index)
            tvalues = pd.Series(robust_model.tvalues, index=base_model.params.index)
            pvalues = pd.Series(robust_model.pvalues, index=base_model.params.index)

        y_std = float(y_valid.std(ddof=0)) or np.nan
        x_stds = X_const.std(ddof=0).replace(0, np.nan)
        coefficients = {}
        for variable, coefficient in base_model.params.items():
            standardized = None
            if variable != "const" and not pd.isna(y_std):
                standardized = float(coefficient) * float(x_stds[variable]) / y_std
            coefficients[variable] = {
                "coefficient": float(coefficient),
                "standardized_coefficient": standardized,
                "standardized_beta": standardized,
                "standard_error": float(bse[variable]),
                "t_value": float(tvalues[variable]),
                "p_value": float(pvalues[variable]),
            }
        bp_stat, bp_p = (np.nan, np.nan)
        if het_breuschpagan is not None:
            bp_stat, bp_p, _, _ = het_breuschpagan(base_model.resid, X_const)
        cooks_distance = np.asarray(base_model.get_influence().cooks_distance[0], dtype=float)
        vif_table = []
        if variance_inflation_factor is not None:
            for index, column in enumerate(X_const.columns):
                if column != "const":
                    vif_table.append({
                        "variable": column,
                        "vif": float(variance_inflation_factor(X_const.to_numpy(dtype=float), index)),
                    })
        result["models"].append({
            "model": model_name,
            "controls": list(controls),
            "n": int(len(y_valid)),
            "predictor_order": list(X_const.columns),
            "coefficients": coefficients,
            "r_squared": float(base_model.rsquared),
            "adjusted_r_squared": float(base_model.rsquared_adj),
            "model_f": float(base_model.fvalue),
            "model_f_p_value": float(base_model.f_pvalue),
            "robust_standard_error_type": robust_type,
            "durbin_watson": float(durbin_watson(base_model.resid)) if durbin_watson is not None else np.nan,
            "jarque_bera_statistic": float(jb_stat),
            "jarque_bera_p_value": float(jb_p),
            "breusch_pagan_statistic": float(bp_stat),
            "breusch_pagan_p_value": float(bp_p),
            "maximum_cooks_distance": float(np.nanmax(cooks_distance)) if len(cooks_distance) else np.nan,
            "cooks_distance_cases_above_4_over_n": int(np.sum(cooks_distance > (4.0 / len(y_valid)))),
            "vif_table": vif_table,
            "maximum_vif": max((row["vif"] for row in vif_table), default=np.nan),
        })
    return result


def table2_linear_regressions(group: str = "all", dataframe=None) -> dict:
    """Nature Table 2-style multiple linear regression with rounded public output."""
    df = dataframe.copy() if dataframe is not None else filter_group(load_data(), group)
    rows: list[dict] = []
    model_stats: list[dict] = []
    if len(df) < 12:
        return _with_analysis_context(
            _with_rows(
                {
                    "analysis": "Table 2-style linear regression",
                    "group": group,
                    "model_stats": [],
                    "privacy_note": "Not enough rows for this group.",
                },
                rows,
            ),
            cohort_size=len(df),
            group=group,
        )
    if sm is None:
        rows.append({"group": group, "model": "all", "variable": "statsmodels unavailable"})
    else:
        try:
            raw = fit_table2_models_raw(df, group=group)
            for model in raw["models"]:
                model_name = model["model"]
                robust_type = model["robust_standard_error_type"]
                for var, values in model["coefficients"].items():
                    rows.append(
                        {
                            "group": group,
                            "model": model_name,
                            "variable": "Intercept" if var == "const" else _display(var),
                            "beta_nature": _safe_float(values["coefficient"]),
                            "standardized_beta": _safe_float(values["standardized_coefficient"]),
                            "SE": _safe_float(values["standard_error"]),
                            "t": _safe_float(values["t_value"]),
                            "p": _safe_float(values["p_value"]),
                            "robust_standard_error_type": robust_type,
                        }
                    )
                model_stats.append({
                    "group": group,
                    "model": model_name,
                    "n": model["n"],
                    "r_squared": _safe_float(model["r_squared"]),
                    "adjusted_r_squared": _safe_float(model["adjusted_r_squared"]),
                    "model_f": _safe_float(model["model_f"]),
                    "model_f_p_value": _safe_float(model["model_f_p_value"]),
                    "durbin_watson": _safe_float(model["durbin_watson"]),
                    "jarque_bera_statistic": _safe_float(model["jarque_bera_statistic"]),
                    "jarque_bera_p_value": _safe_float(model["jarque_bera_p_value"]),
                    "breusch_pagan_statistic": _safe_float(model["breusch_pagan_statistic"]),
                    "breusch_pagan_p_value": _safe_float(model["breusch_pagan_p_value"]),
                    "maximum_cooks_distance": _safe_float(model["maximum_cooks_distance"]),
                    "cooks_distance_cases_above_4_over_n": model["cooks_distance_cases_above_4_over_n"],
                    "maximum_vif": _safe_float(model["maximum_vif"]),
                    "vif_table": [{"variable":_display(row["variable"]),"vif":_safe_float(row["vif"])} for row in model["vif_table"]],
                    "robust_standard_error_type": robust_type,
                    "privacy": "Aggregate model statistics only",
                })
        except Exception as exc:
            rows.append({"group": group, "model": "all", "variable": "model_error", "error": str(exc)})

    return _with_analysis_context(
        _with_rows(
            {
                "analysis": "Table 2-style linear regression",
                "group": group,
                "model_stats": model_stats,
                "privacy_note": f"Executed locally on group='{group}'. Only aggregate coefficients are returned.",
            },
            rows,
        ),
        cohort_size=len(df),
        group=group, dataframe=df,
    )


def correlation_analysis(group: str = "all") -> dict:
    df = filter_group(load_data(), group)
    rows: list[dict] = []
    matrix_data = {}
    for a, b in combinations(PREDICTORS, 2):
        x = pd.to_numeric(df[a], errors="coerce")
        y = pd.to_numeric(df[b], errors="coerce")
        valid = x.notna() & y.notna()
        if valid.sum() < 3:
            r, p = np.nan, np.nan
        elif pearsonr is not None:
            r, p = pearsonr(x[valid], y[valid])
        else:
            r, p = x[valid].corr(y[valid]), np.nan
        rows.append(
            {
                "group": group,
                "variable_1": _display(a),
                "variable_2": _display(b),
                "r": _safe_float(r),
                "p": _safe_float(p),
                "p_value_method": "two-sided Pearson",
                "n": int(valid.sum()),
                "network_edge": bool(abs(r) >= 0.15) if not pd.isna(r) else False,
            }
        )
    for variable in PREDICTORS:
        matrix_data[variable] = pd.to_numeric(df[variable], errors="coerce")
    correlation_matrix = pd.DataFrame(matrix_data).corr()
    rows.sort(key=lambda row: abs(row["r"]) if row["r"] is not None else -1, reverse=True)
    return _with_analysis_context(
        _with_rows(
            {
                "analysis": "Correlation analysis",
                "group": group,
                "correlation_matrix": correlation_matrix,
                "method_note": (
                    "The paper describes one-sided Pearson correlations, while some reported "
                    "p-values are consistent with two-sided testing. This implementation "
                    "reports two-sided p-values explicitly."
                ),
                "privacy_note": f"Executed locally on group='{group}'. Only aggregate pairwise correlations are returned.",
            },
            rows,
        ),
        cohort_size=len(df),
        group=group, dataframe=df,
    )


def variance_analysis(iterations: int = 1000, sample_size: int = 22, seed: int = 42) -> dict:
    df = load_data()
    mask = elite_mask(df)
    elite = df[mask]
    semi = df[~mask]
    rows: list[dict] = []
    if elite.empty or semi.empty:
        return _with_rows({"analysis": "Variance analysis", "privacy_note": "Elite or semi-elite group is empty."}, rows)
    sample_size = min(sample_size, len(elite), len(semi))
    rng = np.random.default_rng(seed)
    elite_var = elite[PREDICTORS].head(sample_size).var(ddof=1)
    sampled = {p: [] for p in PREDICTORS}
    for _ in range(iterations):
        draw = semi.sample(n=sample_size, replace=False, random_state=int(rng.integers(0, 1_000_000)))
        for p in PREDICTORS:
            sampled[p].append(float(draw[p].var(ddof=1)))
    for p in PREDICTORS:
        vals = np.asarray(sampled[p], dtype=float)
        rows.append(
            {
                "variable": _display(p),
                "elite_variance": _safe_float(elite_var[p]),
                "semi_elite_variance_mean": _safe_float(vals.mean()),
                "semi_elite_variance_min": _safe_float(vals.min()),
                "semi_elite_variance_max": _safe_float(vals.max()),
                "iterations": int(iterations),
                "sample_size": int(sample_size),
            }
        )
    return _with_rows(
        {
            "analysis": "Elite vs semi-elite variance analysis",
            "cohort_size": int(min(len(elite), len(semi))),
            "privacy_note": "Executed locally. Only aggregate variance summaries are returned.",
        },
        rows,
    )


# Whitelisted backend functions for generated code.
def run_table1(group: str = "all", dataframe=None) -> dict:
    return table1_logistic_regressions(group=group, dataframe=dataframe)


def run_table2(group: str = "all", dataframe=None) -> dict:
    return table2_linear_regressions(group=group, dataframe=dataframe)


def run_variance() -> dict:
    return variance_analysis()


def run_figure1() -> dict:
    from .figures import create_figure1, figure1_summary_table

    rows = figure1_summary_table()

    return {
        "analysis": "Figure 1-style group statistics",
        "cohort_size": int(len(load_data())),
        "figure": create_figure1(),
        "table": rows,
        "rows": rows,
        "method_note": (
            "Node size represents regression coefficient magnitude. Closer nodes represent "
            "stronger absolute correlation. Line width represents stronger correlation. "
            "Internal bars compare elite variance with sampled semi-elite variance."
        ),
        "privacy_note": "Executed locally. The figure and table contain aggregate statistics only; raw athlete rows are hidden.",
    }


def run_figure2(*, variables=None, filters=None, max_athletes: int | None = None,
                reference_group: str = "selected_cohort", dataframe=None, group: str | None = None) -> dict:
    from .figures import create_figure2
    from .filters import apply_analysis_filters

    source = load_data() if dataframe is None else dataframe.copy()
    effective_filters = dict(filters or {})
    if not effective_filters and group in {"elite", "semi_elite"}:
        effective_filters["expertise_group"] = group
    cohort_df = apply_analysis_filters(source, effective_filters)
    variables = list(variables or PREDICTORS)
    selected = cohort_df.sample(frac=1.0, random_state=FIGURE2_RANDOM_SEED) if max_athletes is None else cohort_df.sample(
        n=min(int(max_athletes), len(cohort_df)), random_state=FIGURE2_RANDOM_SEED
    )
    profile_count = len(selected)
    full_numeric=source[variables].apply(pd.to_numeric,errors="coerce")
    full_z_scores=(full_numeric-full_numeric.mean())/full_numeric.std(ddof=0).replace(0,1)
    cohort_z_scores=full_z_scores.loc[cohort_df.index,variables]
    z_selected=cohort_z_scores.loc[selected.index,variables]
    selected_cohort_mean=cohort_z_scores.mean(axis=0)
    profile_summary = []
    for index, (_, row) in enumerate(z_selected.iterrows(), start=1):
        match = all(float(row[variable]) > 0 for variable in PAPER_PATTERN_DOMAINS)
        profile_summary.append(
            {
                "anonymous_profile_label": f"Profile {index:02d}",
                "pattern_match": bool(match),
                "pattern_description": "Matches the paper's three-domain group-level pattern" if match else "Does not match the paper's three-domain group-level pattern",
            }
        )
    paper_pattern_match_count = sum(1 for row in profile_summary if row["pattern_match"])
    total_athletes = len(cohort_df)
    group_label = " / ".join(str(value).replace("_", " ").title() for value in effective_filters.values()) or "All athletes"
    showing_text = (
        f"Showing all {total_athletes} athletes"
        if max_athletes is None
        else f"Showing {profile_count} of {total_athletes} athletes"
    )

    return _with_analysis_context(
        {
            "analysis": "Figure 2-style z-score profiles",
            "summary": f"Figure 2-style z-score profiles: {group_label}\n{showing_text}",
            "filters": effective_filters,
            "max_athletes": max_athletes,
            "profile_count": int(profile_count),
            "available_profiles": int(total_athletes),
            "shown_profiles": int(profile_count),
            "reference_type": "selected_cohort_mean",
            "reference_label": "Selected Cohort Mean Profile",
            "reference_profile": {domain:float(selected_cohort_mean[domain]) for domain in variables},
            "dataframe_scope": "active_filtered_cohort",
            "paper_pattern_match_count": int(paper_pattern_match_count),
            "paper_pattern_nonmatch_count": int(profile_count - paper_pattern_match_count),
            "profile_summary": profile_summary,
            "table": profile_summary,
            "figure": create_figure2(
                dataframe=cohort_df,
                selected_dataframe=selected,
                standardized_profiles=z_selected,
                reference_profile=selected_cohort_mean,
                reference_label="Selected Cohort Mean Profile",
                variables=variables,
                group_label=group_label,
                max_athletes=max_athletes,
            ),
            "method_note": (
                "Group profiles use anonymous labels. The three-domain pattern checks only "
                "whether basic cognitive function, lower-body dynamics, and blood micronutrients "
                "are above the full-sample mean."
            ),
            "privacy_note": (
                "Executed locally. The figure contains anonymized standardized z-score profiles only. "
                "Raw athlete measurements are hidden. The dashed line represents the mean z-score "
                "profile of the currently selected cohort."
            ),
        },
        cohort_size=total_athletes,
        group="selected",
    )
