"""Summarize Strong/Weak call usage from the saved independent evaluation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts" / "athlete_strong_weak_router_evaluation.json"
DEFAULT_OUTPUT = ROOT / "evaluation" / "cost_aware_router_call_summary.csv"
OUTPUT_COLUMNS = (
    "Strategy",
    "Strong Calls",
    "Weak Calls",
    "Strong Model Usage (%)",
    "Strong Call Reduction (%)",
)


def calculate_summary(path: Path) -> tuple[int, list[dict]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("independent_evaluation") is not True:
        raise ValueError("Expected the saved independent Strong/Weak evaluation")
    predictions = report.get("per_sample_results")
    if not isinstance(predictions, list):
        raise ValueError("Evaluation report has no per_sample_results list")

    total_requests = len(predictions)
    sample_ids = [str(row.get("sample_id") or "").strip() for row in predictions]
    if not all(sample_ids) or len(set(sample_ids)) != total_requests:
        raise ValueError("Evaluation predictions must have unique, non-empty sample IDs")
    expected = report.get("evaluation_samples")
    if expected is not None and int(expected) != total_requests:
        raise ValueError(
            f"Report declares {expected} evaluation samples but contains {total_requests} predictions")
    if total_requests != 40:
        raise ValueError(f"Expected the existing 40-request evaluation; found {total_requests}")

    labels = []
    for row in predictions:
        label = str(row.get("predicted_route") or row.get("prediction") or "").strip().casefold()
        if label not in {"strong", "weak"}:
            raise ValueError(f"Invalid saved prediction for {row.get('sample_id')}: {label!r}")
        labels.append(label)
    strong_calls = sum(label == "strong" for label in labels)
    weak_calls = sum(label == "weak" for label in labels)
    assert strong_calls + weak_calls == total_requests

    strong_usage = strong_calls / total_requests * 100
    strong_reduction = (1 - strong_calls / total_requests) * 100
    rows = [
        {
            "Strategy": "All-Strong Baseline",
            "Strong Calls": total_requests,
            "Weak Calls": 0,
            "Strong Model Usage (%)": 100.0,
            "Strong Call Reduction (%)": 0.0,
        },
        {
            "Strategy": "Cost-aware Router",
            "Strong Calls": strong_calls,
            "Weak Calls": weak_calls,
            "Strong Model Usage (%)": strong_usage,
            "Strong Call Reduction (%)": strong_reduction,
        },
    ]
    return total_requests, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(total_requests: int, rows: list[dict]) -> None:
    print("COST-AWARE ROUTER CALL SUMMARY")
    print("=" * 60)
    print(f"Total Evaluation Requests : {total_requests}")
    print()
    print(f"{'Strategy':<22}{'Strong Calls':>14}{'Weak Calls':>13}"
          f"{'Strong Usage':>16}{'Strong Call Reduction':>24}")
    for row in rows:
        print(f"{row['Strategy']:<22}{row['Strong Calls']:>14}{row['Weak Calls']:>13}"
              f"{row['Strong Model Usage (%)']:>15.1f}%"
              f"{row['Strong Call Reduction (%)']:>23.1f}%")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    total_requests, rows = calculate_summary(args.input)
    write_csv(args.output, rows)
    print_summary(total_requests, rows)
    print(f"Saved table: {args.output}")


if __name__ == "__main__":
    main()
