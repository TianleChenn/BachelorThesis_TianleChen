"""Canonical contracts for restricted frontend analysis requests."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sports.config import PREDICTORS
from sports.filters import validate_analysis_filters


METHOD_ALIASES = {
    "logistic_regression": "table1",
    "multiple_linear_regression": "table2",
    "figure_1": "figure1",
    "figure_2": "figure2",
    "variance_comparison": "variance_analysis",
}

ARGUMENT_ALIASES = {
    "outcome": "target",
    "max_profiles": "max_athletes",
    "group_variable": "group_field",
}


@dataclass(frozen=True)
class AnalysisRequestContract:
    method: str
    arguments: dict


def parse_figure2_request_size(user_request: str | None) -> int | None:
    """Parse the dashboard's Figure 2 size without delegating it to an LLM."""
    text = str(user_request or "").lower()
    match = re.search(
        r"showing\s+(?:at\s+most\s+)?(20|50|80)(?:\s+anonymous)?\s+athletes",
        text,
    )
    if match:
        return int(match.group(1))
    if "showing all available anonymous athletes" in text or (
        "for all athletes" in text
        and not re.search(r"(?:showing|at most)\s+\d+", text)
    ):
        return None
    return 50


def build_request_contract(
    requested_analysis: str,
    requested_filters: dict | None = None,
    user_request: str | None = None,
) -> AnalysisRequestContract:
    method = METHOD_ALIASES.get(str(requested_analysis), str(requested_analysis))
    filters = validate_analysis_filters(requested_filters)
    domains = list(PREDICTORS)
    contracts = {
        "table1": {
            "predictors": domains,
            "target": "elite_status",
            "controls": [[], ["sex"], ["age"], ["sex", "age"]],
            "filters": filters,
        },
        "table2": {"predictors": domains, "filters": filters, "group": "all"},
        "figure1": {
            "variables": domains,
            "target": "expertise_value",
            "group_field": "elite_status",
            "correlation_threshold": 0.15,
            "variance_iterations": 1000,
            "filters": filters,
        },
        "correlation": {
            "variables": domains,
            "filters": filters,
            "method": "pearson",
        },
        "variance_analysis": {
            "variables": domains,
            "group_field": "elite_status",
            "groups": ["elite", "semi_elite"],
            "iterations": 1000,
            "filters": filters,
            "visualization": True,
        },
        "figure2": {
            "variables": domains,
            "filters": filters,
            "max_athletes": parse_figure2_request_size(user_request),
            "reference_group": "selected_cohort",
        },
        "individual_profile": {
            "subject_token": "CURRENT_SUBJECT",
            "variables": domains,
            "reference_group": "all",
            "output_mode": "standardized_profile",
        },
    }
    if method not in contracts:
        raise ValueError(f"Unsupported requested analysis '{requested_analysis}'.")
    return AnalysisRequestContract(method=method, arguments=contracts[method])


def render_request_contract(contract: AnalysisRequestContract) -> str:
    arguments = ", ".join(
        f"{name}={value!r}" for name, value in contract.arguments.items()
    )
    return f"result = analysis.{contract.method}({arguments})"
