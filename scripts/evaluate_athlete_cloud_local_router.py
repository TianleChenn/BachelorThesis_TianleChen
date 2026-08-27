"""Evaluate the Cloud/Local router on the independent objective benchmark."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.athlete_cloud_local_router import AthleteCloudLocalRouter
from scripts.cloud_local_evaluation_common import evaluate_cloud_and_local

DATASET = ROOT / "evaluation" / "athlete_cloud_local_independent_40.json"
OUTPUT = ROOT / "artifacts" / "athlete_cloud_local_router_evaluation.json"
PRICING = ROOT / "evaluation" / "cloud_codegen_pricing_snapshot_2026-08-19.json"


def _estimated_cost(rows: list[dict]) -> float | None:
    pricing = json.loads(PRICING.read_text(encoding="utf-8"))["per_1m_tokens"]["gemini"]
    if any(row["cloud"].get("input_tokens") is None or row["cloud"].get("output_tokens") is None for row in rows):
        return None
    return sum(
        row["cloud"]["input_tokens"] * pricing["input"] / 1_000_000
        + row["cloud"]["output_tokens"] * pricing["output"] / 1_000_000
        for row in rows
    )


def evaluate(dataset_path: Path = DATASET, output_path: Path = OUTPUT,
             router: AthleteCloudLocalRouter | None = None) -> dict:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    samples = dataset.get("samples") or []
    if len(samples) != 40 or dataset.get("independent_evaluation") is not True:
        raise RuntimeError("Expected the locked independent 40-request dataset.")
    router = router or AthleteCloudLocalRouter()
    rows = []
    for sample in samples:
        objective = evaluate_cloud_and_local(sample)
        prediction = router.predict(sample["prompt"], router_prompt_source="independent_evaluation")
        rows.append({**sample, **objective, "ground_truth": objective["preference"],
                     "prediction": prediction["selected_tier"], "p_cloud": prediction["p_cloud"],
                     "threshold": prediction["threshold"]})
    valid = [row for row in rows if row["ground_truth"] in {"cloud", "local"}]
    truth = [int(row["ground_truth"] == "cloud") for row in valid]
    predicted = [int(row["prediction"] == "cloud") for row in valid]
    if not valid or set(truth) != {0, 1}:
        raise RuntimeError("Independent evaluation requires valid Cloud and Local ground-truth samples.")
    matrix = confusion_matrix(truth, predicted, labels=[0, 1]).tolist()
    per_task = defaultdict(list)
    for row in valid:
        per_task[row["analysis_type"]].append(row["ground_truth"] == row["prediction"])
    total = len(rows)
    cloud_calls = sum(row["prediction"] == "cloud" for row in rows)
    router_correct = sum(
        row[row["prediction"]]["fully_correct"] for row in rows
    )
    cloud_cost = _estimated_cost([row for row in rows if row["prediction"] == "cloud"])
    metrics = {
        "routing_accuracy": sum(a == b for a, b in zip(truth, predicted)) / len(valid),
        "balanced_accuracy": balanced_accuracy_score(truth, predicted),
        "cloud_recall": recall_score(truth, predicted, pos_label=1, zero_division=0),
        "local_recall": recall_score(truth, predicted, pos_label=0, zero_division=0),
        "cloud_precision": precision_score(truth, predicted, pos_label=1, zero_division=0),
        "cloud_usage_rate": cloud_calls / total,
        "confusion_matrix": {"true_local": {"pred_local": matrix[0][0], "pred_cloud": matrix[0][1]},
                             "true_cloud": {"pred_local": matrix[1][0], "pred_cloud": matrix[1][1]}},
        "per_task_accuracy": {key: sum(values) / len(values) for key, values in sorted(per_task.items())},
    }
    strategies = {
        "all_cloud_baseline": {"cloud_calls": total, "local_calls": 0, "cloud_usage_rate": 1.0,
            "cloud_api_call_reduction_vs_all_cloud": 0.0,
            "fully_correct_end_to_end_rate": sum(row["cloud"]["fully_correct"] for row in rows) / total,
            "estimated_external_api_cost": _estimated_cost(rows)},
        "all_local_baseline": {"cloud_calls": 0, "local_calls": total, "cloud_usage_rate": 0.0,
            "cloud_api_call_reduction_vs_all_cloud": 1.0,
            "fully_correct_end_to_end_rate": sum(row["local"]["fully_correct"] for row in rows) / total,
            "estimated_external_api_cost": 0.0},
        "cost_aware_cloud_local_router": {"cloud_calls": cloud_calls, "local_calls": total-cloud_calls,
            "cloud_usage_rate": cloud_calls/total,
            "cloud_api_call_reduction_vs_all_cloud": 1-cloud_calls/total,
            "fully_correct_end_to_end_rate": router_correct/total,
            "estimated_external_api_cost": cloud_cost},
    }
    report = {"status": "evaluated", "independent_evaluation": True,
              "valid_ground_truth_samples": len(valid), "invalid_samples": total-len(valid),
              **metrics, "cost_perspective": strategies, "per_sample_results": rows,
              "cost_note": "Cloud API call reduction is not total system cost reduction; local hardware and energy costs are not measured."}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def print_report(report: dict) -> None:
    for label, key in (("Routing Accuracy", "routing_accuracy"), ("Balanced Accuracy", "balanced_accuracy"),
                       ("Cloud Recall", "cloud_recall"), ("Local Recall", "local_recall"),
                       ("Cloud Precision", "cloud_precision"), ("Cloud Usage Rate", "cloud_usage_rate")):
        print(f"{label}: {report[key]:.2%}")
    matrix = report["confusion_matrix"]
    print("\n                 Pred Local   Pred Cloud")
    print(f"True Local       {matrix['true_local']['pred_local']:>10} {matrix['true_local']['pred_cloud']:>12}")
    print(f"True Cloud       {matrix['true_cloud']['pred_local']:>10} {matrix['true_cloud']['pred_cloud']:>12}")
    print("\nCloud API call reduction and estimated external API cost:")
    print(json.dumps(report["cost_perspective"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print_report(evaluate(args.dataset, args.output))
