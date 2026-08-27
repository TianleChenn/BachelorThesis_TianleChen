"""Validation and schema normalization for the continuous 4D dataset."""

from __future__ import annotations

import json
import math
import warnings
from collections import Counter
from pathlib import Path

from .llm_soft_gating_model import FEATURE_NAMES, ROUTE_TO_INDEX

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_LLM_4D_TRAINING_DATASET = (
    PROJECT_ROOT / "evaluation" / "privacy_gating_train_4d_hard_llm_generated_90.json"
)
HARD_PROMPT_DATASET = (
    PROJECT_ROOT / "evaluation" / "privacy_gating_train_4d_hard_independent_90_prompts.json"
)

RECOMMENDED_SIZE = 90
MINIMUM_TRAINING_SIZE = 30
MINIMUM_PER_ROUTE = 5
REQUIRED_NORMALIZED_FIELDS = {"question", "ground_truth_route", *FEATURE_NAMES}
FIELD_MAPPING = {
    "question": "prompt -> question",
    "ground_truth_route": "route -> ground_truth_route",
    "features[0]": "privacy_risk_score",
    "features[1]": "subject_scope",
    "features[2]": "data_sensitivity",
    "features[3]": "disclosure_level",
}


def _normalize_row(item: dict, index: int) -> tuple[dict, dict[str, str]]:
    row = dict(item)
    mappings: dict[str, str] = {}
    if not str(row.get("question", "")).strip() and str(row.get("prompt", "")).strip():
        row["question"] = row["prompt"]
        mappings["question"] = "prompt"
    if not row.get("ground_truth_route"):
        for source in ("expected_route", "route"):
            if row.get(source):
                row["ground_truth_route"] = row[source]
                mappings["ground_truth_route"] = source
                break
    feature_vector = row.get("features")
    if isinstance(feature_vector, list) and len(feature_vector) == len(FEATURE_NAMES):
        for position, name in enumerate(FEATURE_NAMES):
            if name not in row:
                row[name] = feature_vector[position]
                mappings[name] = f"features[{position}]"
    row.setdefault("id", f"continuous4d_{index:03d}")
    return row, mappings


def load_llm_gating_dataset(
    path: str | Path,
    *,
    require_training_size: bool = False,
    return_info: bool = False,
    validate_distribution: bool = True,
):
    """Load, normalize and strictly validate the active continuous dataset."""
    source = Path(path)
    if not source.is_file():
        if source.resolve() == ACTIVE_LLM_4D_TRAINING_DATASET.resolve():
            raise FileNotFoundError(
                "The approved LLM-generated hard 4D training dataset is missing. "
                "Run feature generation and approval before training."
            )
        raise FileNotFoundError(f"4D training dataset not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    raw_rows = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(raw_rows, list):
        raise ValueError("The 4D dataset must be a JSON list or contain a samples list")

    rows: list[dict] = []
    errors: list[str] = []
    mappings: dict[str, set[str]] = {}
    counts: Counter[str] = Counter()
    ids: list[str] = []
    for index, item in enumerate(raw_rows, 1):
        if not isinstance(item, dict):
            errors.append(f"row {index} is not an object")
            continue
        row, row_mappings = _normalize_row(item, index)
        for target, source_name in row_mappings.items():
            mappings.setdefault(target, set()).add(source_name)
        missing = REQUIRED_NORMALIZED_FIELDS - set(row)
        if missing:
            errors.append(f"row {index} missing fields: {', '.join(sorted(missing))}")
            continue
        question = str(row["question"]).strip()
        if not question:
            errors.append(f"row {index} has an empty question")
        route = row["ground_truth_route"]
        if route not in ROUTE_TO_INDEX:
            errors.append(f"row {index} has invalid route {route!r}")
        else:
            counts[route] += 1
        for name in FEATURE_NAMES:
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"row {index} {name} is not numeric")
            elif not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                errors.append(f"row {index} {name} is outside [0, 1]")
        identifier = str(row.get("id", "")).strip()
        ids.append(identifier)
        rows.append(row)

    duplicates = [value for value, count in Counter(ids).items() if value and count > 1]
    if duplicates:
        errors.append(f"duplicate IDs: {', '.join(duplicates)}")
    underfilled = {
        route: counts.get(route, 0)
        for route in ROUTE_TO_INDEX
        if counts.get(route, 0) < MINIMUM_PER_ROUTE
    }
    if underfilled and validate_distribution:
        errors.append(
            f"each route requires at least {MINIMUM_PER_ROUTE} samples; found {underfilled}"
        )
    if require_training_size and len(rows) < MINIMUM_TRAINING_SIZE:
        errors.append(
            f"training requires at least {MINIMUM_TRAINING_SIZE} samples; found {len(rows)}"
        )
    if errors:
        raise ValueError("Invalid 4D training dataset: " + "; ".join(errors))
    if len(rows) != RECOMMENDED_SIZE:
        warnings.warn(
            f"4D dataset contains {len(rows)} samples instead of the recommended "
            f"{RECOMMENDED_SIZE}.", UserWarning, stacklevel=2,
        )
    info = {
        "sample_count": len(rows),
        "class_distribution": dict(counts),
        "field_mapping": {
            target: sorted(sources) for target, sources in sorted(mappings.items())
        },
        "required_fields": sorted(REQUIRED_NORMALIZED_FIELDS),
        "field_validation": "passed",
        "recommended_size": RECOMMENDED_SIZE,
        "minimum_training_size": MINIMUM_TRAINING_SIZE,
    }
    return (rows, info) if return_info else rows


def row_features(row: dict) -> list[float]:
    return [float(row[name]) for name in FEATURE_NAMES]
