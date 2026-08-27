"""Shared validation helpers for the reviewed LLM-generated 4D workflow."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .llm_soft_gating_model import FEATURE_NAMES, ROUTE_TO_INDEX

REVIEW_STATUSES = {"pending", "approved", "rejected", "corrected"}


def infer_prompt_family(prompt: str) -> str:
    """Assign one stable, auditable family used only to prevent template leakage."""
    text = str(prompt).casefold()
    families = [
        ("raw_request", r"\b(?:raw|export|download|original row|exact measurement)\b"),
        ("identifiable_individual", r"\bathlete[\s_:#.\-]*\d+\b|\bidentifiable\b|\bone athlete\b"),
        ("anonymous_profile", r"\banonymous\b.*\bprofile\b|\bindividual profile\b"),
        ("small_cohort", r"\b(?:small|two|three|very small|uniquely filtered)\b.*\b(?:cohort|group|athletes?)\b"),
        ("network", r"\b(?:network|figure 1)\b"),
        ("variance", r"\b(?:variance|figure 2)\b"),
        ("correlation", r"\bcorrelat"),
        ("regression", r"\b(?:regression|table 1|table 2)\b"),
        ("sensitive_cohort", r"\b(?:blood|mental health|genetic|micronutrient|biometric)\b"),
        ("table_analysis", r"\btable\b"),
    ]
    for family, pattern in families:
        if re.search(pattern, text):
            return family
    return "general_statistics"


def load_reviewed_features(path: str | Path, *, approved_only: bool = True) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    rows = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("LLM-generated training data must contain a samples list")
    accepted: list[dict] = []
    errors: list[str] = []
    for index, item in enumerate(rows, 1):
        if not isinstance(item, dict):
            errors.append(f"row {index} is not an object"); continue
        status = item.get("review_status")
        if status not in REVIEW_STATUSES:
            errors.append(f"row {index} has invalid review_status {status!r}"); continue
        if approved_only and status != "approved":
            continue
        route = item.get("route")
        features = item.get("features")
        if route not in ROUTE_TO_INDEX:
            errors.append(f"row {index} has invalid route {route!r}")
        if not isinstance(features, list) or len(features) != len(FEATURE_NAMES):
            errors.append(f"row {index} must have four features")
        elif any(isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1 for value in features):
            errors.append(f"row {index} has invalid features")
        if not item.get("prompt_family"):
            errors.append(f"row {index} is missing prompt_family")
        accepted.append(item)
    if errors:
        raise ValueError("Invalid reviewed 4D data: " + "; ".join(errors))
    if approved_only and not accepted:
        raise ValueError("No approved LLM-generated samples are available for training")
    return accepted


def assert_no_prompt_overlap(training_rows: list[dict], evaluation_rows: list[dict]) -> None:
    normalize = lambda value: " ".join(str(value).casefold().split())
    training_prompts = {normalize(row["prompt"]) for row in training_rows}
    evaluation_prompts = {normalize(row["prompt"]) for row in evaluation_rows}
    overlap = training_prompts & evaluation_prompts
    if overlap:
        raise ValueError(f"Training/evaluation prompt overlap detected: {len(overlap)}")
    training_families = Counter(row.get("prompt_family") for row in training_rows)
    if not training_families:
        raise ValueError("Training prompt families are unavailable")
