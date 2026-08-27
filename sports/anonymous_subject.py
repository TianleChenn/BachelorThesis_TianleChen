"""Server-local anonymous subject selection; internal IDs never enter results."""
from __future__ import annotations

import secrets
from collections.abc import Callable

import pandas as pd

from .analysis import load_data


ATHLETE_GROUP_OPTIONS = ("Elite", "Semi-elite", "All athletes")


def anonymous_candidate_frame(group: str, dataframe: pd.DataFrame | None = None) -> pd.DataFrame:
    if group not in ATHLETE_GROUP_OPTIONS:
        raise ValueError("Unsupported athlete group.")
    frame = load_data() if dataframe is None else dataframe.copy()
    expertise = pd.to_numeric(frame["expertise_value"], errors="coerce")
    if group == "Elite":
        frame = frame[expertise.ge(13)]
    elif group == "Semi-elite":
        frame = frame[expertise.lt(13)]
    return frame.copy()


def select_anonymous_subject(
    group: str,
    *,
    previous_id: str | None = None,
    dataframe: pd.DataFrame | None = None,
    chooser: Callable[[list[str]], str] = secrets.choice,
) -> str:
    candidates = anonymous_candidate_frame(group, dataframe)
    candidate_ids = candidates["athlete_id"].astype(str).tolist()
    if not candidate_ids:
        raise ValueError("No athletes are available in the selected group.")
    available = [candidate_id for candidate_id in candidate_ids if candidate_id != previous_id]
    return chooser(available or candidate_ids)
