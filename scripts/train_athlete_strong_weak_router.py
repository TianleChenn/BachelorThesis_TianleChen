from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.athlete_router_features import build_classifier_pipeline, build_prompt_texts


INPUT = ROOT / "artifacts" / "athlete_router_preferences_v2_valid.json"
MODEL_OUTPUT = ROOT / "artifacts" / "athlete_strong_weak_router.joblib"
METADATA_OUTPUT = ROOT / "artifacts" / "athlete_strong_weak_router.json"
SEED = 2026
QUALITY_TOLERANCE = 0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def label_from_scores(
    strong_valid: bool,
    weak_valid: bool,
    strong_score: float | None,
    weak_score: float | None,
) -> int | None:
    """Existing quality-aware label rule retained for evaluation compatibility."""
    if strong_valid and not weak_valid:
        return 1
    if weak_valid and not strong_valid:
        return 0
    if not strong_valid and not weak_valid:
        return None
    if strong_score is None or weak_score is None:
        return None
    return int(float(strong_score) - float(weak_score) > QUALITY_TOLERANCE)


def load_training_rows(path: Path = INPUT) -> list[dict]:
    """Load only existing valid, explicitly judged Strong/Weak preferences."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset_name") != "athlete_router_preferences_v2_valid":
        raise RuntimeError("Training input must use the athlete_router_preferences_v2_valid schema")
    source = payload.get("results") or []
    rows = []
    for item in source:
        if not isinstance(item, dict) or item.get("status") != "valid":
            continue
        preference = str(item.get("preference") or "").strip().lower()
        if preference not in {"strong", "weak"}:
            continue
        prompt = " ".join(str(item.get("prompt") or "").split())
        if not prompt:
            continue
        rows.append({
            "id": item.get("sample_id"),
            "prompt": prompt,
            "label": 1 if preference == "strong" else 0,
            "preferred_model": preference,
            # Scores are retained only for downstream evaluation helpers. They
            # are never passed to the preference classifier.
            "strong_score": float(item["strong_score"]),
            "weak_score": float(item["weak_score"]),
        })
    return rows


def threshold_candidates(_probabilities: np.ndarray | None = None) -> np.ndarray:
    """The locked candidate grid for the project-specific preference router."""
    return np.linspace(0.01, 0.99, 991)


def preference_metrics(probabilities, threshold: float, labels) -> dict:
    labels = np.asarray(labels, dtype=int)
    predictions = (np.asarray(probabilities, dtype=float) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "strong_precision": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "strong_recall": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "weak_recall": float(recall_score(labels, predictions, pos_label=0, zero_division=0)),
        "strong_usage_rate": float(np.mean(predictions)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def select_preference_threshold(probabilities, labels) -> dict:
    reports = [preference_metrics(probabilities, threshold, labels)
               for threshold in threshold_candidates()]
    return min(reports, key=lambda row: (
        -row["balanced_accuracy"],
        -row["accuracy"],
        row["strong_usage_rate"],
    ))


def routing_metrics(
    probabilities: np.ndarray,
    threshold: float,
    labels: np.ndarray,
    strong_scores: np.ndarray,
    weak_scores: np.ndarray,
) -> dict:
    """Existing quality-aware report used by independent evaluation artifacts."""
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    selected = np.where(predictions == 1, strong_scores, weak_scores)
    best = np.maximum(strong_scores, weak_scores)
    regret = best - selected
    return {
        "threshold": float(threshold),
        "quality_aware_accuracy": float(np.mean(regret <= QUALITY_TOLERANCE)),
        "strict_accuracy": float(np.mean(predictions == labels)),
        "mean_quality_regret": float(np.mean(regret)),
        "p95_quality_regret": float(np.percentile(regret, 95)),
        "strong_model_usage": float(np.mean(predictions)),
        "selected_average_score": float(np.mean(selected)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def train_router(
    input_path: Path = INPUT,
    model_path: Path = MODEL_OUTPUT,
    metadata_path: Path = METADATA_OUTPUT,
) -> dict:
    print(f"Training dataset: {input_path.resolve()}")
    rows = load_training_rows(input_path)
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    counts = Counter(labels.tolist())
    if len(rows) < 30 or set(counts) != {0, 1} or min(counts.values()) < 5:
        raise RuntimeError(
            "Training requires at least 30 valid preferences and five samples per class; "
            f"found total={len(rows)}, Strong={counts.get(1, 0)}, Weak={counts.get(0, 0)}"
        )

    print(f"Loaded preference data: {len(rows)} valid samples")
    print(f"Strong: {counts[1]}")
    print(f"Weak: {counts[0]}")
    print("\nRunning 5-fold out-of-fold evaluation...")

    prompts = build_prompt_texts(rows)
    pipeline = build_classifier_pipeline()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    probabilities = cross_val_predict(
        pipeline, prompts, labels, cv=folds, method="predict_proba"
    )[:, 1]
    metrics = select_preference_threshold(probabilities, labels)

    print(f"\nSelected threshold: {metrics['threshold']:.4f}")
    print(f"OOF Accuracy: {metrics['accuracy']:.1%}")
    print(f"OOF Balanced Accuracy: {metrics['balanced_accuracy']:.1%}")
    print(f"Strong Usage Rate: {metrics['strong_usage_rate']:.1%}")
    print(f"Strong Precision: {metrics['strong_precision']:.1%}")
    print(f"Strong Recall: {metrics['strong_recall']:.1%}")
    print(f"Weak Recall: {metrics['weak_recall']:.1%}")
    print(f"\nTraining final preference router on all {len(rows)} samples...")

    pipeline.fit(prompts, labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metadata = {
        "status": "trained",
        "router_type": "project_specific_preference_router",
        "inspiration": "RouteLLM preference-based routing",
        "strong_model": "gpt-4.1",
        "weak_model": "ministral-3-8b",
        "training_samples": len(rows),
        "strong_samples": counts[1],
        "weak_samples": counts[0],
        "uses_official_mf_score": False,
        "threshold": metrics["threshold"],
        "cv_accuracy": metrics["accuracy"],
        "cv_balanced_accuracy": metrics["balanced_accuracy"],
        "strong_precision": metrics["strong_precision"],
        "strong_recall": metrics["strong_recall"],
        "weak_recall": metrics["weak_recall"],
        "strong_usage_rate": metrics["strong_usage_rate"],
        "confusion_matrix_cv": metrics["confusion_matrix"],
        "random_state": SEED,
        "threshold_source": "stratified 5-fold out-of-fold probabilities",
        "threshold_candidates": "numpy.linspace(0.01, 0.99, 991)",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_dataset": display_path(input_path),
        "training_dataset_sha256": sha256_file(input_path),
        "model_sha256": sha256_file(model_path),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"\nSaved:\n{display_path(model_path)}\n{display_path(metadata_path)}"
    )
    return metadata


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the RouteLLM-inspired project-specific preference router."
    )
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--model-output", type=Path, default=MODEL_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=METADATA_OUTPUT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    train_router(args.input, args.model_output, args.metadata_output)
