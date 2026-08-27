"""Exhaustively validate the deterministic dashboard analysis matrix.

This diagnostic never calls an LLM. It renders the trusted request contract and
executes it through the same generated-code verifier and restricted local API
used after runtime code generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm.analysis_request_contracts import (  # noqa: E402
    build_request_contract,
    render_request_contract,
)
from llm.generated_code_verifier import verify_and_execute_generated_code  # noqa: E402
from privacy.numerical_perturbation import NOISE_HIGH, NOISE_LOW  # noqa: E402
from sports.analysis import load_data  # noqa: E402
from sports.analysis_noise_utility import EXPERIMENT_VERSION  # noqa: E402
from sports.config import DOMAIN_ORDER, SPORTS  # noqa: E402
from sports.filters import (  # noqa: E402
    ALLOWED_ANALYSIS_FILTERS,
    CANONICAL_FILTER_VALUES,
    apply_analysis_filters,
    validate_analysis_filters,
)
from ui.cohort_prompts import build_dashboard_prompt  # noqa: E402


ANALYSES = (
    "table1",
    "table2",
    "figure1",
    "figure2",
    "correlation",
    "variance_analysis",
)
NON_FIGURE2_ANALYSES = tuple(value for value in ANALYSES if value != "figure2")
FIGURE2_SIZE_OPTIONS = ("20", "50", "80", "All")
EXPECTED_NOISE_REPETITIONS = 50
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "diagnostics" / "dashboard_analysis_matrix.csv"


@dataclass(frozen=True)
class CohortSelection:
    category: str
    value: str
    prompt_text: str
    filters: dict[str, str]


@dataclass(frozen=True)
class MatrixCase:
    cohort: CohortSelection
    analysis: str
    figure2_size: str | None = None


def build_cohort_selections() -> list[CohortSelection]:
    """Derive the 17 current frontend selections from canonical configuration."""
    selections = [CohortSelection("All athletes", "all", "all athletes", {})]
    for value in CANONICAL_FILTER_VALUES["expertise_group"]:
        selections.append(CohortSelection(
            "Expertise group",
            value,
            "elite athletes" if value == "elite" else "semi-elite athletes",
            {"expertise_group": value},
        ))
    for value in SPORTS:
        selections.append(CohortSelection(
            "Sport discipline", value, f"{value} athletes", {"sport": value}
        ))
    for value in CANONICAL_FILTER_VALUES["sex"]:
        selections.append(CohortSelection(
            "Sex", value, f"{value} athletes", {"sex": value}
        ))
    for value in CANONICAL_FILTER_VALUES["national_team"]:
        selections.append(CohortSelection(
            "National team level",
            value,
            (
                "junior national team athletes"
                if value == "Junior"
                else "senior national team athletes"
            ),
            {"national_team": value},
        ))
    for value in CANONICAL_FILTER_VALUES["age_group"]:
        selections.append(CohortSelection(
            "Age group",
            value,
            "athletes under 20" if value == "under_20" else "athletes aged 20 and above",
            {"age_group": value},
        ))
    if len(selections) != 17:
        raise AssertionError(f"Expected 17 dashboard cohort selections, found {len(selections)}.")
    return selections


def build_matrix_cases() -> list[MatrixCase]:
    cases: list[MatrixCase] = []
    for cohort in build_cohort_selections():
        cases.extend(MatrixCase(cohort, analysis) for analysis in NON_FIGURE2_ANALYSES)
        cases.extend(MatrixCase(cohort, "figure2", size) for size in FIGURE2_SIZE_OPTIONS)
    if len(cases) != 153:
        raise AssertionError(f"Expected 153 detailed matrix cases, found {len(cases)}.")
    return cases


def validate_frontend_filter_wiring() -> list[str]:
    """Check that the active frontend selector, prompt, and service wiring agree."""
    source = (PROJECT_ROOT / "frontend.py").read_text(encoding="utf-8")
    compact = "".join(source.split())
    errors: list[str] = []
    required_filter_fragments = (
        'return"allathletes",{}',
        'filters={"expertise_group":expertise_group}',
        'filters={"sport":sport}',
        'filters={"sex":sex}',
        'filters={"national_team":"Junior"ifteam.startswith("junior")else"Senior"}',
        'filters={"age_group":age_group}',
    )
    for fragment in required_filter_fragments:
        if fragment not in compact:
            errors.append(f"Frontend cohort mapping fragment is missing: {fragment}")
    required_wiring = (
        "cohort_group,active_filters=_dashboard_cohort_selector()",
        "analysis_filters=active_filters",
        "analysis_filters=analysis_filters",
        "build_dashboard_prompt(analysis[\"requested_analysis\"],cohort_group)",
        "build_dashboard_prompt(\"figure2\",cohort_group,figure2_size_option)",
    )
    for fragment in required_wiring:
        if fragment not in compact:
            errors.append(f"Frontend request wiring fragment is missing: {fragment}")
    if tuple(SPORTS) != tuple(CANONICAL_FILTER_VALUES["sport"]):
        errors.append("sports.config.SPORTS differs from canonical sport filter values.")
    expected = {
        "sport", "sex", "expertise_group", "national_team", "age_group"
    }
    if not expected.issubset(ALLOWED_ANALYSIS_FILTERS):
        errors.append("The frontend filter keys are not all backend-allowlisted.")
    return errors


def _figure2_maximum(option: str | None) -> int | None:
    return None if option == "All" else int(str(option))


def _not_applicable_reason(
    analysis: str,
    cohort_size: int,
    elite_count: int,
    semi_elite_count: int,
) -> str | None:
    if analysis == "table1" and (elite_count == 0 or semi_elite_count == 0):
        return "Logistic regression requires both elite_status outcome classes."
    if analysis in {"figure1", "variance_analysis"} and (
        elite_count < 2 or semi_elite_count < 2
    ):
        return "Elite/semi-elite sample variance requires at least two athletes per group."
    if analysis == "table2" and cohort_size < 12:
        return "The current multiple-regression implementation requires at least 12 rows."
    if analysis == "correlation" and cohort_size < 3:
        return "Pairwise correlation requires at least three complete observations."
    if analysis == "figure2" and cohort_size < 1:
        return "No anonymous athlete profile exists in the selected cohort."
    return None


def _has_figure(value: Any) -> bool:
    return hasattr(value, "savefig")


def _contains_identifier(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).casefold() == "athlete_id" or _contains_identifier(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_identifier(item) for item in value)
    if isinstance(value, str):
        return "Athlete_" in value
    return False


def _noise_utility_valid(result: dict | None, analysis: str) -> tuple[bool, str]:
    if not isinstance(result, dict):
        return False, "No analysis result was available for noise validation."
    utility = result.get("noise_utility")
    if not isinstance(utility, dict):
        return False, "noise_utility is missing."
    mean_result = utility.get("mean_perturbed_result")
    average_difference = utility.get("average_difference")
    checks = {
        "analysis_key": utility.get("analysis_key") == analysis,
        "noise_range": utility.get("noise_range") == [NOISE_LOW, NOISE_HIGH],
        "repetitions": utility.get("repetitions") == EXPECTED_NOISE_REPETITIONS,
        "experiment_version": utility.get("experiment_version") == EXPERIMENT_VERSION,
        "mean_perturbed_result": isinstance(mean_result, dict) and (
            isinstance(mean_result.get("table"), (list, tuple))
            or _has_figure(mean_result.get("figure"))
        ),
        "average_difference": isinstance(average_difference, dict)
        and average_difference.get("value") is not None,
        "stability_figure": _has_figure(utility.get("stability_figure")),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return not failed, "" if not failed else "Noise checks failed: " + ", ".join(failed)


def _semantic_result_valid(
    result: dict | None,
    case: MatrixCase,
    cohort_size: int,
    elite_count: int,
    semi_elite_count: int,
) -> tuple[bool, str]:
    if not isinstance(result, dict):
        return False, "The restricted pipeline returned no result dictionary."
    analysis = case.analysis
    rows = result.get("rows") if "rows" in result else result.get("table")
    if analysis == "table1":
        stats = result.get("model_stats")
        fitted = isinstance(stats, list) and any(
            isinstance(model, dict) and model.get("converged") is True for model in stats
        )
        valid = (
            result.get("analysis") == "Table 1-style logistic regression"
            and isinstance(rows, list) and bool(rows)
            and isinstance(stats, list) and bool(stats)
            and fitted
        )
        return valid, "" if valid else "Table 1 did not contain a genuinely fitted logistic model."
    if analysis == "table2":
        stats = result.get("model_stats")
        valid = (
            result.get("analysis") == "Table 2-style linear regression"
            and isinstance(rows, list) and bool(rows)
            and isinstance(stats, list) and bool(stats)
            and any(isinstance(model, dict) and "r_squared" in model for model in stats)
        )
        return valid, "" if valid else "Table 2 did not contain real regression output."
    if analysis == "figure1":
        valid = (
            result.get("analysis") == "Figure 1-style group statistics"
            and isinstance(result.get("table"), list)
            and len(result["table"]) == len(DOMAIN_ORDER)
            and _has_figure(result.get("figure"))
        )
        return valid, "" if valid else "Figure 1 table or figure was incomplete."
    if analysis == "figure2":
        requested_maximum = _figure2_maximum(case.figure2_size)
        expected_count = cohort_size if requested_maximum is None else min(cohort_size, requested_maximum)
        valid = (
            "figure 2" in str(result.get("analysis") or "").casefold()
            and result.get("max_athletes") == requested_maximum
            and result.get("profile_count") == expected_count
            and result.get("available_profiles") == cohort_size
            and result.get("reference_type") == "selected_cohort_mean"
            and result.get("dataframe_scope") == "active_filtered_cohort"
            and _has_figure(result.get("figure"))
            and not _contains_identifier(result)
        )
        return valid, "" if valid else "Figure 2 size, scope, reference, figure, or anonymity check failed."
    if analysis == "correlation":
        correlation_rows = result.get("rows") or result.get("table") or []
        valid = (
            result.get("analysis") == "Correlation analysis"
            and result.get("method") == "pearson"
            and len(correlation_rows) == 28
            and all(row.get("n") == cohort_size for row in correlation_rows)
            and "figure" not in result
        )
        return valid, "" if valid else "Correlation did not return the expected 28-row table-only result."
    if analysis == "variance_analysis":
        variance_rows = result.get("table") or []
        valid = (
            result.get("analysis") == "Dynamic variance analysis"
            and result.get("elite_group_size") == elite_count
            and result.get("semi_elite_group_size") == semi_elite_count
            and len(variance_rows) == len(DOMAIN_ORDER)
            and all(
                {"elite_variance", "semi_elite_variance", "iterations"}.issubset(row)
                and row.get("iterations") == 1000
                for row in variance_rows
            )
            and _has_figure(result.get("figure"))
        )
        return valid, "" if valid else "Variance result metadata, rows, or figure was incomplete."
    return False, f"Unsupported semantic analysis check: {analysis}"


def _contract_error(case: MatrixCase, prompt: str, contract) -> str | None:
    if contract.method != case.analysis:
        return f"Contract method {contract.method!r} did not match {case.analysis!r}."
    if contract.arguments.get("filters") != case.cohort.filters:
        return "Contract filters did not match the selected frontend filters."
    if case.analysis == "figure2":
        expected = _figure2_maximum(case.figure2_size)
        if contract.arguments.get("max_athletes") != expected:
            return f"Figure 2 contract maximum did not match {expected!r}."
    if case.cohort.prompt_text not in prompt:
        return "The prompt text did not identify the selected frontend cohort."
    return None


def validate_case(case: MatrixCase, frontend_errors: list[str] | None = None) -> dict[str, Any]:
    dataframe = load_data()
    filters = validate_analysis_filters(case.cohort.filters)
    filtered = apply_analysis_filters(dataframe, filters)
    expertise = filtered["expertise_value"]
    elite_count = int(expertise.ge(13).sum())
    semi_elite_count = int(expertise.lt(13).sum())
    cohort_size = int(len(filtered))
    prompt = build_dashboard_prompt(
        case.analysis,
        case.cohort.prompt_text,
        case.figure2_size or "50",
    )
    contract = build_request_contract(case.analysis, filters, prompt)
    contract_error = _contract_error(case, prompt, contract)
    code = render_request_contract(contract)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        execution = verify_and_execute_generated_code(
            code,
            user_request=prompt,
            requested_analysis=case.analysis,
            requested_filters=filters,
            close_figures_after_execution=True,
        )
    result = execution.result
    filter_match = isinstance(result, dict) and result.get("filters") == filters
    cohort_size_match = isinstance(result, dict) and result.get("cohort_size") == cohort_size
    semantic_check, semantic_error = _semantic_result_valid(
        result, case, cohort_size, elite_count, semi_elite_count
    )
    noise_check, noise_error = _noise_utility_valid(result, case.analysis)
    not_applicable = _not_applicable_reason(
        case.analysis, cohort_size, elite_count, semi_elite_count
    )
    if (
        case.analysis == "table1"
        and execution.validation_error
        and "not statistically applicable" in execution.validation_error.casefold()
    ):
        not_applicable = execution.validation_error
    errors = [
        value for value in (
            "; ".join(frontend_errors or []),
            contract_error,
            execution.validation_error,
            semantic_error if not semantic_check else None,
            noise_error if not noise_check else None,
        ) if value
    ]
    old_mechanism_tokens = (
        "Small" + "GroupError",
        "MIN_" + "GROUP_SIZE",
        "minimum release " + "size",
        "at least 10",
    )
    old_mechanism_error = any(
        token.casefold() in " ".join(errors).casefold() for token in old_mechanism_tokens
    )
    if frontend_errors:
        status = "BUG_FILTER_MAPPING"
    elif contract_error:
        status = "BUG_REQUEST_CONTRACT"
    elif not execution.structure_validation_passed:
        status = "BUG_STRUCTURE_VALIDATION"
    elif not execution.request_match_passed:
        status = "BUG_REQUEST_VALIDATION"
    elif not_applicable and not old_mechanism_error:
        status = "NOT_APPLICABLE_STATISTICALLY"
        errors = [not_applicable]
        if execution.validation_error:
            errors.append(execution.validation_error)
    elif not execution.local_execution_passed:
        status = "BUG_LOCAL_EXECUTION"
    elif not execution.result_validation_passed:
        status = "BUG_RESULT_VALIDATION"
    elif not filter_match:
        status = "BUG_FILTER_MAPPING"
    elif not cohort_size_match:
        status = "BUG_WRONG_COHORT"
    elif not semantic_check:
        status = "BUG_EMPTY_RESULT"
    elif not noise_check:
        status = "ANALYSIS_PASS_NOISE_FAIL"
    else:
        status = "PASS"
        errors = []
    return {
        "cohort_category": case.cohort.category,
        "cohort_value": case.cohort.value,
        "filters": json.dumps(filters, ensure_ascii=False, sort_keys=True),
        "cohort_size": cohort_size,
        "elite_count": elite_count,
        "semi_elite_count": semi_elite_count,
        "analysis": case.analysis,
        "figure2_size": case.figure2_size or "",
        "structure_validation": bool(execution.structure_validation_passed),
        "request_match": bool(execution.request_match_passed),
        "local_execution": bool(execution.local_execution_passed),
        "result_validation": bool(execution.result_validation_passed),
        "filter_match": bool(filter_match),
        "cohort_size_match": bool(cohort_size_match),
        "semantic_result_check": bool(semantic_check),
        "noise_utility_check": bool(noise_check),
        "status": status,
        "error": " | ".join(dict.fromkeys(errors)),
    }


def run_dashboard_matrix(*, progress: bool = False) -> list[dict[str, Any]]:
    frontend_errors = validate_frontend_filter_wiring()
    rows = []
    cases = build_matrix_cases()
    for index, case in enumerate(cases, start=1):
        if progress:
            print(
                f"[{index:03d}/{len(cases)}] {case.cohort.category} / "
                f"{case.cohort.value} / {case.analysis} / {case.figure2_size or '-'}",
                flush=True,
            )
        rows.append(validate_case(case, frontend_errors))
    return rows


def write_matrix_csv(rows: Iterable[dict[str, Any]], output: Path) -> None:
    materialized = list(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def print_report(rows: list[dict[str, Any]]) -> None:
    print("\nCohort / value                         Analysis            Size   N   E   S   Status")
    print("-" * 106)
    for row in rows:
        cohort = f"{row['cohort_category']} / {row['cohort_value']}"
        print(
            f"{cohort[:38]:38} {row['analysis'][:19]:19} "
            f"{str(row['figure2_size'] or '-')[:4]:4} "
            f"{row['cohort_size']:3} {row['elite_count']:3} {row['semi_elite_count']:3} "
            f"{row['status']}"
        )
    counts = {
        "PASS": sum(row["status"] == "PASS" for row in rows),
        "NOT_APPLICABLE_STATISTICALLY": sum(
            row["status"] == "NOT_APPLICABLE_STATISTICALLY" for row in rows
        ),
        "BUG": sum(str(row["status"]).startswith("BUG_") for row in rows),
        "ANALYSIS_PASS_NOISE_FAIL": sum(
            row["status"] == "ANALYSIS_PASS_NOISE_FAIL" for row in rows
        ),
    }
    print("\n" + "=" * 60)
    print("Dashboard Exhaustive Validation Summary")
    print("=" * 60)
    print("Cohort selections: 17")
    print("Base analysis combinations: 102")
    print(f"Detailed cases including Figure 2 sizes: {len(rows)}")
    for name, count in counts.items():
        print(f"{name}: {count}")
    for row in rows:
        if row["status"] == "PASS":
            continue
        print(f"\n[{row['status']}]")
        print(f"{row['cohort_category']} / {row['cohort_value']} / {row['analysis']}")
        print(
            f"N={row['cohort_size']}, elite={row['elite_count']}, "
            f"semi_elite={row['semi_elite_count']}"
        )
        print(f"Reason: {row['error']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    rows = run_dashboard_matrix(progress=not args.no_progress)
    write_matrix_csv(rows, args.output)
    print_report(rows)
    print(f"\nCSV report: {args.output.resolve()}")
    has_bug = any(
        str(row["status"]).startswith("BUG_")
        or row["status"] == "ANALYSIS_PASS_NOISE_FAIL"
        for row in rows
    )
    return 1 if has_bug else 0


if __name__ == "__main__":
    raise SystemExit(main())
