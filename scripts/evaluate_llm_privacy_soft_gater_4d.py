"""Internal validation report for the small simulated LLM 4D dataset."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from privacy.llm_soft_gating_data import ACTIVE_LLM_4D_TRAINING_DATASET, load_llm_gating_dataset, row_features
from privacy.llm_soft_gating_model import INDEX_TO_ROUTE, ROUTE_TO_INDEX, load_llm_privacy_soft_gater
DEFAULT_DATASET = ACTIVE_LLM_4D_TRAINING_DATASET


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--model", default="artifacts/prism_soft_gater_4d_llm_hard.pt")
    parser.add_argument("--output", default="artifacts/prism_soft_gater_4d_llm_hard_validation.json")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    dataset_path = resolve_project_path(args.dataset)
    model_path = resolve_project_path(args.model)
    output_path = resolve_project_path(args.output)
    if not dataset_path.exists():
        raise FileNotFoundError(f"4D privacy training dataset was not found: {dataset_path}")
    print(f"Dataset:\n{dataset_path}")
    rows, dataset_info = load_llm_gating_dataset(dataset_path, return_info=True)
    print(f"Samples: {dataset_info['sample_count']}")
    print(f"Class distribution: {dataset_info['class_distribution']}")
    labels = np.asarray([ROUTE_TO_INDEX[row["ground_truth_route"]] for row in rows])
    _, validation_index = train_test_split(
        np.arange(len(rows)), test_size=.20, random_state=args.seed, stratify=labels
    )
    x = torch.tensor([row_features(rows[index]) for index in validation_index], dtype=torch.float32)
    truth = labels[validation_index]
    model = load_llm_privacy_soft_gater(model_path)
    with torch.no_grad():
        probabilities = torch.softmax(model(x), dim=1).numpy()
    predicted = probabilities.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, average="macro", zero_division=0
    )
    matrix = confusion_matrix(truth, predicted, labels=range(3))
    result = {
        "status": "internal_validation",
        "notice": "This is an internal validation result on a small simulated dataset, not the final independent evaluation.",
        "sample_count": len(validation_index),
        "accuracy": float((predicted == truth).mean()),
        "macro_precision": float(precision), "macro_recall": float(recall),
        "macro_f1": float(f1),
        "per_class_accuracy": {
            INDEX_TO_ROUTE[i]: float(matrix[i, i] / matrix[i].sum()) if matrix[i].sum() else 0.0
            for i in range(3)
        },
        "confusion_matrix": matrix.tolist(),
        "average_predicted_probability": {
            INDEX_TO_ROUTE[i]: float(probabilities[:, i].mean()) for i in range(3)
        },
        "prediction_entropy": float(np.mean([
            -sum(float(p) * math.log(max(float(p), 1e-12)) for p in row)
            for row in probabilities
        ])),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
