"""Canonical structured filtering for every protected analysis."""
from __future__ import annotations

import pandas as pd

from .config import SPORTS


ALLOWED_ANALYSIS_FILTERS = {"sport", "sex", "expertise_group", "elite_status", "national_team", "age_group"}

# These are the canonical literal values shared by prompt construction, request
# contracts, validation, and protected backend filtering.  In particular, sport
# names intentionally retain their human-readable spaces (for example,
# ``"ice hockey"`` rather than ``"icehockey"``).
CANONICAL_FILTER_VALUES = {
    "sport": tuple(SPORTS),
    "sex": ("female", "male"),
    "expertise_group": ("elite", "semi_elite"),
    "elite_status": ("elite", "semi_elite"),
    "national_team": ("Junior", "Senior"),
    "age_group": ("under_20", "20_and_above"),
}


def validate_analysis_filters(filters: dict[str, str] | None) -> dict[str, str]:
    """Return a copy of filters only when every key/value is canonical."""
    if filters is None:
        return {}
    if not isinstance(filters, dict):
        raise ValueError("filters must be a literal dictionary.")
    validated = dict(filters)
    for key, value in validated.items():
        if key not in ALLOWED_ANALYSIS_FILTERS or not isinstance(value, str):
            raise ValueError("Unsupported filter.")
        if value not in CANONICAL_FILTER_VALUES[key]:
            raise ValueError(f"Invalid canonical value for filter '{key}'.")
    return validated


def apply_analysis_filters(dataframe: pd.DataFrame, filters: dict[str, str] | None) -> pd.DataFrame:
    filters = validate_analysis_filters(filters)
    filtered = dataframe.copy()
    for key, value in filters.items():
        if key == "sport":
            filtered = filtered[filtered["sport"].astype(str).str.casefold() == value.casefold()]
        elif key == "sex":
            filtered = filtered[filtered["sex"].astype(str).str.casefold() == value.casefold()]
        elif key == "expertise_group":
            expertise = pd.to_numeric(filtered["expertise_value"], errors="coerce")
            filtered = filtered[expertise.ge(13) if value == "elite" else expertise.lt(13)]
        elif key == "elite_status":
            mapping = {"elite": 1, "semi_elite": 0}
            filtered = filtered[filtered["elite_status"].astype(int) == mapping[value]]
        elif key == "national_team":
            filtered = filtered[
                filtered["national_team"].astype(str).str.casefold().str.contains(value.casefold(), na=False)
            ]
        elif key == "age_group":
            age = pd.to_numeric(filtered["age"], errors="coerce")
            filtered = filtered[age.lt(20) if value == "under_20" else age.ge(20)]
    return filtered.copy()
