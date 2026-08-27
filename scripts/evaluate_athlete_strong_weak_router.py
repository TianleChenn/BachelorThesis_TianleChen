"""Evaluate the saved text-only router on the independent frontend40 references."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.athlete_strong_weak_router import AthleteStrongWeakRouter
from scripts.athlete_router_evaluation_common import load_frontend_benchmark
from scripts.train_athlete_strong_weak_router import QUALITY_TOLERANCE


DATASET = ROOT / "evaluation" / "frontend_realistic_benchmark_60.json"
DETAILS = ROOT / "artifacts" / "routellm_frontend40_results.json"
OUTPUT = ROOT / "artifacts" / "athlete_strong_weak_router_evaluation.json"


def _select_example(
    rows: list[dict], truth: str | None, correct: bool, excluded: set[str] | None = None,
) -> dict | None:
    excluded = excluded or set()
    candidates = [row for row in rows if row["correct"] is correct and
                  row["sample_id"] not in excluded and
                  (truth is None or row["ground_truth"] == truth)]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (
        -abs(float(row["strong_model_probability"]) - float(row["threshold"])),
        len(row["prompt"]), row["sample_id"],
    ))


def _summary(rows: list[dict]) -> dict:
    total = len(rows)
    true_strong = sum(row["ground_truth"] == "strong" for row in rows)
    true_weak = total - true_strong
    strong_correct = sum(row["ground_truth"] == row["predicted_route"] == "strong" for row in rows)
    weak_correct = sum(row["ground_truth"] == row["predicted_route"] == "weak" for row in rows)
    strong_recall = strong_correct / true_strong if true_strong else 0.0
    weak_recall = weak_correct / true_weak if true_weak else 0.0
    return {
        "accuracy": (strong_correct + weak_correct) / total,
        "balanced_accuracy": (strong_recall + weak_recall) / 2.0,
        "strong_recall": strong_recall,
        "weak_recall": weak_recall,
        "strong_usage_rate": sum(row["predicted_route"] == "strong" for row in rows) / total,
        "confusion_matrix": {
            "true_weak": {"pred_weak": weak_correct, "pred_strong": true_weak - weak_correct},
            "true_strong": {"pred_weak": true_strong - strong_correct, "pred_strong": strong_correct},
        },
    }


def evaluate(
    dataset_path: Path = DATASET,
    details_path: Path = DETAILS,
    output_path: Path = OUTPUT,
    router: AthleteStrongWeakRouter | None = None,
) -> dict:
    metadata, eligible = load_frontend_benchmark(dataset_path)
    source_by_id = {str(row["id"]): row for row in eligible}
    details = json.loads(details_path.read_text(encoding="utf-8"))
    reference_by_id = {str(row.get("prompt_id")): row for row in details.get("results", [])}
    router = router or AthleteStrongWeakRouter()
    evaluated = []
    for sample in eligible:
        sample_id = str(sample["id"])
        reference = reference_by_id.get(sample_id)
        if not reference or reference.get("status") != "valid":
            continue
        strong_score = reference.get("strong_score")
        weak_score = reference.get("weak_score")
        if strong_score is None or weak_score is None:
            continue
        prompt = sample["prompt"]
        if reference.get("prompt") != prompt:
            raise RuntimeError(f"Saved reference prompt mismatch for {sample_id}")
        # This is the same cost-aware preference rule used for the 65 training labels.
        ground_truth = (
            "strong" if float(strong_score) - float(weak_score) > QUALITY_TOLERANCE
            else "weak"
        )
        prediction = router.predict(
            prompt,
            privacy_route=sample["privacy_route"],
            router_prompt_source="saved_frontend_realistic_evaluation",
        )
        predicted = str(prediction["selected_tier"])
        evaluated.append({
            "sample_id": sample_id,
            "prompt": prompt,
            "privacy_route": sample["privacy_route"],
            "ground_truth": ground_truth,
            "judge_label": reference.get("judge_label"),
            "strong_model_probability": float(prediction["p_strong"]),
            "threshold": float(prediction["threshold"]),
            "predicted_route": predicted,
            "prediction": predicted,
            "selected_model": "GPT-4.1" if predicted == "strong" else "Ministral-3-8B",
            "correct": predicted == ground_truth,
            "ground_truth_construction": {
                "strong_model_score": float(strong_score),
                "weak_model_score": float(weak_score),
            },
        })
    if len(evaluated) != 40:
        raise RuntimeError(f"Expected 40 valid independent references; found {len(evaluated)}")
    thresholds = {row["threshold"] for row in evaluated}
    if len(thresholds) != 1:
        raise RuntimeError("Router returned inconsistent learned thresholds")
    metrics = _summary(evaluated)
    router_metadata = json.loads(router.metadata_path.read_text(encoding="utf-8"))
    correct_strong_example = _select_example(evaluated, "strong", True)
    weak_example = (
        _select_example(evaluated, "weak", True)
        or _select_example(evaluated, "weak", False)
    )
    used_ids = {row["sample_id"] for row in (correct_strong_example, weak_example) if row}
    report = {
        "status": "evaluated",
        "evaluation_type": "independent_40_sample_test",
        "independent_evaluation": True,
        "training_samples": int(router_metadata["training_samples"]),
        "evaluation_samples": 40,
        "total_samples": 40,
        "valid_samples": 40,
        "ground_truth_strong": sum(row["ground_truth"] == "strong" for row in evaluated),
        "ground_truth_weak": sum(row["ground_truth"] == "weak" for row in evaluated),
        "threshold": evaluated[0]["threshold"],
        **metrics,
        "dataset_name": metadata["dataset_name"],
        "dataset_path": metadata["dataset_path"],
        "shared_benchmark_samples": metadata["shared_benchmark_samples"],
        "eligible_llm_samples": metadata["eligible_llm_samples"],
        "router": "RouteLLM-inspired project-specific preference router",
        "uses_official_mf_score": False,
        "per_sample_results": evaluated,
        "representative_examples": {
            "correct_strong": correct_strong_example,
            "weak_example": weak_example,
            "incorrect": _select_example(evaluated, None, False, used_ids),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def print_report(report: dict) -> None:
    print("========================================")
    print("Independent Strong/Weak Router Evaluation")
    print("========================================")
    for label, key, percentage in (
        ("Training samples", "training_samples", False),
        ("Independent evaluation samples", "evaluation_samples", False),
        ("Ground Truth Strong", "ground_truth_strong", False),
        ("Ground Truth Weak", "ground_truth_weak", False),
        ("Threshold", "threshold", False),
        ("Routing Accuracy", "accuracy", True),
        ("Balanced Accuracy", "balanced_accuracy", True),
        ("Strong Recall", "strong_recall", True),
        ("Weak Recall", "weak_recall", True),
        ("Strong Usage Rate", "strong_usage_rate", True),
    ):
        value = report[key]
        print(f"{label}: {value:.2%}" if percentage else f"{label}: {value}")
    matrix = report["confusion_matrix"]
    print("\n                 Pred Weak   Pred Strong")
    print(f"True Weak             {matrix['true_weak']['pred_weak']:>2}          {matrix['true_weak']['pred_strong']:>2}")
    print(f"True Strong           {matrix['true_strong']['pred_weak']:>2}          {matrix['true_strong']['pred_strong']:>2}")
    print("========================================")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the new router on the saved frontend40 references.")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--details", type=Path, default=DETAILS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    print_report(evaluate(
        args.dataset, args.details, args.output,
    ))
