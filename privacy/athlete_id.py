"""Shared Athlete ID recognition and normalization helpers."""

from __future__ import annotations

import re


# A trailing word boundary after ``athlete`` would reject ``athlete23`` because
# letters and digits are both word characters. This lookahead rejects plurals
# while allowing a number or one of the supported separators to follow.
ATHLETE_ID_PATTERN = re.compile(
    r"\bathlete(?![A-Za-z])"
    r"(?:\s*(?:id|identifier|number|no\.?)\s*)?"
    r"(?:[\s_:#.\-]*)"
    r"(?P<athlete_number>\d+)\b",
    re.IGNORECASE,
)


def normalize_athlete_id(value: str) -> str | None:
    """Return ``Athlete_<digits>`` while preserving leading zeroes."""
    match = ATHLETE_ID_PATTERN.fullmatch(str(value or "").strip())
    if not match:
        return None
    return f"Athlete_{match.group('athlete_number')}"


def redact_athlete_ids(value: str, replacement: str = "[REDACTED_ID]") -> str:
    return ATHLETE_ID_PATTERN.sub(replacement, str(value or ""))
