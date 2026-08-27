"""Provider-independent safe pools for restricted code generation."""

from __future__ import annotations

from sports.config import PREDICTORS
from sports.filters import CANONICAL_FILTER_VALUES
from sports.restricted_analysis_api import ALLOWED_METHODS, METHOD_ARGUMENT_SCHEMAS


_DESCRIPTIONS = {
    "table1": "Run the predefined Table 1-style logistic regression analysis.",
    "table2": (
        "Run the predefined Table 2-style multiple linear regression analysis. "
        "The predefined analysis uses expertise_value as its continuous outcome."
    ),
    "figure1": "Generate Figure 1-style group-level statistics and visualization.",
    "figure2": "Generate anonymous standardized athlete profile lines.",
    "correlation": "Calculate pairwise correlations for selected public domains.",
    "variance_analysis": "Compare variance between allowed athlete groups.",
    "individual_profile": "Generate one protected standardized anonymous athlete profile.",
}

_COMMON_ARGUMENTS = {
    "predictors": {"type": "list[string]", "values_from": "public_domains"},
    "variables": {"type": "list[string]", "values_from": "public_domains"},
    "filters": {
        "type": "dictionary",
        "semantics": (
            "Use {} when no cohort restriction is requested. Otherwise use only "
            "allowed filter fields and values."
        ),
    },
    "target": {"type": "string", "values_from": "targets"},
    "controls": {
        "type": "list[list[string]]",
        "values_from": "controls",
        "semantics": (
            "Each inner list represents one model specification. An empty inner list "
            "represents a model without controls."
        ),
    },
    "visualization": {"type": "boolean"},
    "group_field": {"type": "string", "values_from": "group_fields"},
    "groups": {"type": "list[string]", "values_from": "variance_groups"},
}

_METHOD_ARGUMENTS = {
    "table1": {
        name: _COMMON_ARGUMENTS[name] for name in ("predictors", "target", "controls", "filters")
    },
    "table2": {
        "predictors": _COMMON_ARGUMENTS["predictors"],
        "filters": _COMMON_ARGUMENTS["filters"],
        "group": {"type": "string", "values_from": "table2_group_values"},
    },
    "figure1": {
        "variables": _COMMON_ARGUMENTS["variables"],
        "target": _COMMON_ARGUMENTS["target"],
        "group_field": _COMMON_ARGUMENTS["group_field"],
        "correlation_threshold": {"type": "number", "values_from": "correlation_threshold"},
        "variance_iterations": {"type": "integer", "values_from": "variance_iterations"},
        "filters": _COMMON_ARGUMENTS["filters"],
    },
    "figure2": {
        "variables": _COMMON_ARGUMENTS["variables"],
        "filters": _COMMON_ARGUMENTS["filters"],
        "max_athletes": {"type": "integer or null", "values_from": "figure2_max_athletes"},
        "reference_group": {"type": "string", "values_from": "reference_groups"},
    },
    "correlation": {
        "variables": _COMMON_ARGUMENTS["variables"],
        "filters": _COMMON_ARGUMENTS["filters"],
        "method": {"type": "string", "values_from": "correlation_methods"},
        "visualization": _COMMON_ARGUMENTS["visualization"],
    },
    "variance_analysis": {
        "variables": _COMMON_ARGUMENTS["variables"],
        "group_field": _COMMON_ARGUMENTS["group_field"],
        "groups": _COMMON_ARGUMENTS["groups"],
        "iterations": {"type": "integer", "values_from": "variance_iterations"},
        "filters": _COMMON_ARGUMENTS["filters"],
        "visualization": _COMMON_ARGUMENTS["visualization"],
    },
    "individual_profile": {
        "subject_token": {"type": "string", "values_from": "subject_tokens"},
        "variables": _COMMON_ARGUMENTS["variables"],
        "reference_group": {"type": "string", "values_from": "reference_groups"},
        "output_mode": {"type": "string", "values_from": "output_modes"},
    },
}

ANALYSIS_METHOD_POOL = {
    name: {
        "call": f"analysis.{name}",
        "arguments": arguments,
        "description": _DESCRIPTIONS[name],
    }
    for name, arguments in _METHOD_ARGUMENTS.items()
}

if set(ANALYSIS_METHOD_POOL) != set(ALLOWED_METHODS):
    raise RuntimeError("The code-generation method pool is out of sync with ALLOWED_METHODS.")
if set(ANALYSIS_METHOD_POOL) != set(METHOD_ARGUMENT_SCHEMAS):
    raise RuntimeError("The code-generation method pool is out of sync with API schemas.")

ALLOWED_VALUE_POOL = {
    "public_domains": list(PREDICTORS),
    "targets": ["elite_status", "expertise_value"],
    "controls": ["age", "sex"],
    "group_fields": ["elite_status"],
    "variance_groups": ["elite", "semi_elite"],
    "table2_group_values": ["all"],
    "correlation_methods": ["pearson", "spearman"],
    "filter_fields": ["sport", "sex", "expertise_group", "elite_status", "national_team", "age_group"],
    "sports": list(CANONICAL_FILTER_VALUES["sport"]),
    "sex_values": list(CANONICAL_FILTER_VALUES["sex"]),
    "expertise_group_values": list(CANONICAL_FILTER_VALUES["expertise_group"]),
    "elite_status_values": list(CANONICAL_FILTER_VALUES["elite_status"]),
    "national_team_values": list(CANONICAL_FILTER_VALUES["national_team"]),
    "age_group_values": list(CANONICAL_FILTER_VALUES["age_group"]),
    "figure2_max_athletes": [20, 50, 80, None],
    "reference_groups": ["selected_cohort", "all"],
    "output_modes": ["standardized_profile"],
    "subject_tokens": ["CURRENT_SUBJECT"],
    "correlation_threshold": {"type": "number", "minimum": 0, "maximum": 1},
    "variance_iterations": {"type": "integer", "minimum": 100, "maximum": 2000},
}
