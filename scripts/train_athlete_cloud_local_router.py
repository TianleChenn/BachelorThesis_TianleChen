"""Train the text-only cost-aware Cloud/Local router."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.athlete_router_features import build_classifier_pipeline, build_prompt_texts
from llm.code_generation_prompt import CODE_GENERATION_PROMPT_VERSION
from llm.env import load_local_env

INPUT = ROOT / "artifacts" / "athlete_cloud_local_preferences_valid.json"
MODEL_OUTPUT = ROOT / "artifacts" / "athlete_cloud_local_router.joblib"
METADATA_OUTPUT = ROOT / "artifacts" / "athlete_cloud_local_router.json"
SEED = 2026


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_rows(path: Path = INPUT) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in payload.get("results") or []:
        preference = str(item.get("preference") or "").lower()
        prompt = " ".join(str(item.get("prompt") or "").split())
        if preference in {"cloud", "local"} and prompt:
            rows.append({"id": item.get("id"), "prompt": prompt,
                         "label": int(preference == "cloud"), "preference": preference})
    return rows


def threshold_candidates() -> np.ndarray:
    return np.linspace(0.01, 0.99, 991)


def preference_metrics(probabilities, threshold: float, labels) -> dict:
    labels = np.asarray(labels, dtype=int)
    predictions = (np.asarray(probabilities, dtype=float) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "cloud_precision": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "cloud_recall": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "local_recall": float(recall_score(labels, predictions, pos_label=0, zero_division=0)),
        "cloud_usage_rate": float(np.mean(predictions)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def select_preference_threshold(probabilities, labels) -> dict:
    reports = [preference_metrics(probabilities, threshold, labels) for threshold in threshold_candidates()]
    return min(reports, key=lambda row: (-row["balanced_accuracy"], -row["accuracy"], row["cloud_usage_rate"]))


def train_router(input_path: Path = INPUT, model_path: Path = MODEL_OUTPUT,
                 metadata_path: Path = METADATA_OUTPUT) -> dict:
    load_local_env()
    rows = load_training_rows(input_path)
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    counts = Counter(labels.tolist())
    if len(rows) < 30 or set(counts) != {0, 1} or min(counts.values()) < 5:
        raise RuntimeError(
            "Training requires at least 30 valid preferences and five samples per class; "
            f"found total={len(rows)}, Cloud={counts.get(1, 0)}, Local={counts.get(0, 0)}. "
            "Collect more diverse requests; do not relabel samples."
        )
    prompts = build_prompt_texts(rows)
    pipeline = build_classifier_pipeline()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    probabilities = cross_val_predict(pipeline, prompts, labels, cv=folds, method="predict_proba")[:, 1]
    metrics = select_preference_threshold(probabilities, labels)
    pipeline.fit(prompts, labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metadata = {
        "status": "trained", "router_type": "project_specific_cloud_local_router",
        "inspiration": "RouteLLM preference-based routing adapted to Cloud/Local code generation",
        "cloud_model": os.getenv("LLM_GEMINI_MODEL", "gemini-3.5-flash"),
        "local_model": os.getenv("LLM_LOCAL_MODEL", "Ministral-3-8B-Local"),
        "training_samples": len(rows), "cloud_samples": counts[1], "local_samples": counts[0],
        "threshold": metrics["threshold"], "cv_accuracy": metrics["accuracy"],
        "cv_balanced_accuracy": metrics["balanced_accuracy"],
        "cloud_precision": metrics["cloud_precision"], "cloud_recall": metrics["cloud_recall"],
        "local_recall": metrics["local_recall"], "cloud_usage_rate": metrics["cloud_usage_rate"],
        "confusion_matrix_cv": metrics["confusion_matrix"],
        "prompt_version": CODE_GENERATION_PROMPT_VERSION,
        "label_policy": "local if local fully correct; else cloud if cloud fully correct; else invalid",
        "training_dataset": str(input_path.resolve().relative_to(ROOT)).replace("\\", "/"),
        "training_dataset_sha256": sha256_file(input_path),
        "threshold_source": "stratified 5-fold out-of-fold probabilities",
        "random_state": SEED, "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata["model_sha256"] = sha256_file(model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in (
        "training_samples", "cloud_samples", "local_samples", "threshold", "cv_accuracy",
        "cv_balanced_accuracy", "cloud_usage_rate")}, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--model-output", type=Path, default=MODEL_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=METADATA_OUTPUT)
    args = parser.parse_args()
    train_router(args.input, args.model_output, args.metadata_output)
