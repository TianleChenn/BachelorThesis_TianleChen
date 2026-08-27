"""Read-only audit of the simulated 60-sample LLM 4D dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from privacy.llm_soft_gating_data import ACTIVE_LLM_4D_TRAINING_DATASET, load_llm_gating_dataset, row_features
from privacy.llm_soft_gating_model import FEATURE_NAMES
DEFAULT_DATASET = ACTIVE_LLM_4D_TRAINING_DATASET


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def audit(path: Path) -> dict:
    valid_rows, dataset_info = load_llm_gating_dataset(path, return_info=True)
    ids = [str(row.get("id", "")) for row in valid_rows]
    questions = [str(row.get("question", "")).strip() for row in valid_rows]
    route_counts = Counter(row.get("ground_truth_route") for row in valid_rows)
    complete = [row for row in valid_rows if all(name in row for name in FEATURE_NAMES)]
    matrix = np.asarray([row_features(row) for row in complete], dtype=float) if complete else np.empty((0, 4))
    by_route = defaultdict(list)
    for row in complete:
        by_route[row.get("ground_truth_route")].append(row_features(row))
    vectors = Counter(tuple(row_features(row)) for row in complete)
    route_to_index = {"cloud": 0, "collaboration": 1, "local_edge": 2}
    route_labels = np.asarray([route_to_index[row["ground_truth_route"]] for row in complete])
    single_feature_accuracy = {}
    if len(matrix):
        for feature_index, feature_name in enumerate(FEATURE_NAMES):
            classifier = DecisionTreeClassifier(max_depth=2, random_state=2026)
            classifier.fit(matrix[:, [feature_index]], route_labels)
            single_feature_accuracy[feature_name] = float(
                classifier.score(matrix[:, [feature_index]], route_labels)
            )
    over_separable_warnings = [
        f"{name} alone classifies {accuracy:.1%} of this simulation dataset"
        for name, accuracy in single_feature_accuracy.items() if accuracy >= .95
    ]
    questionable = []
    rubric_ranges = {
        "cloud": ((.05, .35), (.00, .25), (.05, .40), (.10, .45)),
        "collaboration": ((.40, .70), (.20, .60), (.55, .90), (.30, .60)),
        "local_edge": ((.65, 1.00), (.75, 1.00), (.30, 1.00), (.45, 1.00)),
    }
    for row in complete:
        values = row_features(row); route = row.get("ground_truth_route")
        if route == "cloud" and min(values) >= .70:
            questionable.append({"id": row.get("id"), "reason": "cloud with four high values"})
        if route == "local_edge" and max(values) <= .25:
            questionable.append({"id": row.get("id"), "reason": "local_edge with four low values"})
        ranges = rubric_ranges.get(route)
        if ranges and any(not low <= value <= high for value, (low, high) in zip(values, ranges)):
            questionable.append({
                "id": row.get("id"), "prompt": row.get("question"),
                "reason": f"features fall outside the semantic design range for {route}",
            })
    normalized_questions = defaultdict(set)
    for row in valid_rows:
        normalized_questions[str(row.get("question", "")).casefold().strip()].add(row.get("ground_truth_route"))
    conflicts = [question for question, routes in normalized_questions.items() if question and len(routes) > 1]
    return {
        "sample_count": len(valid_rows), "class_distribution": dict(route_counts),
        "feature_statistics": {
            name: {"min": float(matrix[:, i].min()), "max": float(matrix[:, i].max()),
                   "mean": float(matrix[:, i].mean()), "std": float(matrix[:, i].std())}
            for i, name in enumerate(FEATURE_NAMES)
        } if len(matrix) else {},
        "per_route_feature_means": {
            route: dict(zip(FEATURE_NAMES, np.asarray(values).mean(axis=0).tolist()))
            for route, values in by_route.items()
        },
        "duplicate_questions": [q for q, count in Counter(questions).items() if q and count > 1],
        "duplicate_ids": [identifier for identifier, count in Counter(ids).items() if identifier and count > 1],
        "out_of_range_values": sum(
            1 for row in complete for value in row_features(row) if not 0 <= value <= 1
        ),
        "missing_fields": [
            {"row": index, "fields": [name for name in FEATURE_NAMES if name not in row]}
            for index, row in enumerate(valid_rows, 1) if any(name not in row for name in FEATURE_NAMES)
        ],
        "identical_vector_groups": sum(1 for count in vectors.values() if count > 1),
        "single_feature_training_accuracy": single_feature_accuracy,
        "over_separable_warnings": over_separable_warnings,
        "conflicting_question_labels": conflicts,
        "questionable_samples": questionable,
        "rubric_consistency": "passed" if not questionable else "failed",
        "field_validation": dataset_info["field_validation"],
        "field_mapping": dataset_info["field_mapping"],
        "warning": (None if len(valid_rows) == 90 else
                    f"Expected 90 simulation samples; found {len(valid_rows)}."),
        "validation_error": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    args = parser.parse_args()
    dataset_path = resolve_project_path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(
            "4D privacy training dataset was not found.\n"
            f"Resolved path: {dataset_path}\n"
            f"Project root: {PROJECT_ROOT}\n"
            "Expected default file: evaluation/privacy_gating_train_4d_hard_llm_generated_90.json"
        )
    print(f"Dataset:\n{dataset_path}")
    report = audit(dataset_path)
    print(json.dumps(report, indent=2))
    distribution = report["class_distribution"]
    invalid = len(report["missing_fields"]) + report["out_of_range_values"]
    print(f"Training samples: {report['sample_count']}")
    print(f"Cloud: {distribution.get('cloud', 0)}")
    print(f"Collaboration: {distribution.get('collaboration', 0)}")
    print(f"Local Edge: {distribution.get('local_edge', 0)}")
    print(f"Invalid Samples: {invalid}")
    print(f"Duplicate Prompts: {len(report['duplicate_questions'])}")
    print(f"Duplicate Feature Vectors: {report['identical_vector_groups']}")
    print(f"Feature Range Check: {'Passed' if report['out_of_range_values'] == 0 else 'Failed'}")
    print(f"Route Balance: {'Passed' if set(distribution.values()) == {30} else 'Failed'}")
    print(f"Rubric Consistency: {report['rubric_consistency'].title()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
