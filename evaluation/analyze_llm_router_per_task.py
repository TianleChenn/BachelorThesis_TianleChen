"""Offline per-task analysis of the saved cost-aware LLM router evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "artifacts" / "athlete_strong_weak_router_evaluation.json"
DEFAULT_BENCHMARK = ROOT / "evaluation" / "frontend_realistic_benchmark_60.json"
DEFAULT_CSV = ROOT / "artifacts" / "llm_router_per_task_accuracy.csv"
DEFAULT_PNG = ROOT / "artifacts" / "llm_router_per_task_accuracy.png"

TASK_ORDER = [
    "table1_logistic_regression",
    "table2_multiple_linear_regression",
    "figure1_group_analysis",
    "correlation_analysis",
    "variance_analysis",
]
TASK_LABELS = {
    "table1_logistic_regression": "Logistic Regression",
    "table2_multiple_linear_regression": "Multiple Linear Regression",
    "figure1_group_analysis": "Group Analysis",
    "correlation_analysis": "Correlation Analysis",
    "variance_analysis": "Variance Analysis",
}


def load_rows(results_path: Path, benchmark_path: Path) -> pd.DataFrame:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if results.get("independent_evaluation") is not True:
        raise ValueError("The saved router report must be an independent evaluation")
    predictions = results.get("per_sample_results")
    samples = benchmark.get("samples")
    if not isinstance(predictions, list) or not isinstance(samples, list):
        raise ValueError("Expected per_sample_results and benchmark samples lists")
    family_by_id = {str(row["id"]): row.get("prompt_family") for row in samples}
    rows = []
    for prediction in predictions:
        sample_id = str(prediction.get("sample_id"))
        family = family_by_id.get(sample_id)
        if not family:
            raise ValueError(f"No benchmark task family found for {sample_id}")
        predicted = str(prediction.get("predicted_route") or prediction.get("prediction") or "").strip().casefold()
        truth = str(prediction.get("ground_truth") or "").strip().casefold()
        if predicted not in {"strong", "weak"} or truth not in {"strong", "weak"}:
            raise ValueError(f"Invalid saved strong/weak route for {sample_id}")
        rows.append({"sample_id": sample_id, "task": family, "correct": predicted == truth})
    frame = pd.DataFrame(rows)
    if len(frame) != 40 or frame["sample_id"].nunique() != 40:
        raise ValueError(f"Expected 40 unique saved evaluation samples; found {len(frame)} rows and "
                         f"{frame['sample_id'].nunique()} unique IDs")
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    summary = frame.groupby("task", as_index=False).agg(
        n_requests=("correct", "size"), n_correct=("correct", "sum"), accuracy=("correct", "mean"))
    summary["accuracy_percent"] = summary["accuracy"] * 100
    summary["task_label"] = summary["task"].map(TASK_LABELS).fillna(
        summary["task"].str.replace("_", " ").str.title())
    rank = {task: index for index, task in enumerate(TASK_ORDER)}
    summary["_rank"] = summary["task"].map(rank).fillna(len(rank))
    return summary.sort_values(["_rank", "task"]).drop(columns="_rank").reset_index(drop=True)


def plot(summary_df: pd.DataFrame, output_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(summary_df["task_label"], summary_df["accuracy_percent"], color="#4C78A8")
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in summary_df["accuracy_percent"]],
                 padding=3, fontsize=9)
    ax.set_ylabel("Routing Accuracy (%)")
    ax.set_xlabel("Analysis Task")
    ax.set_title("Cost-aware Router Accuracy by Analysis Task")
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", rotation=15)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.grid(axis="y", alpha=.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_PNG)
    args = parser.parse_args()
    summary = summarize(load_rows(args.results, args.benchmark))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    plot(summary, args.output_png)

    print("========================================")
    print("COST-AWARE ROUTER ACCURACY BY ANALYSIS TASK")
    print("Independent evaluation samples found: 40")
    for row in summary.itertuples(index=False):
        print(f"{row.task_label}: {row.accuracy_percent:.1f}% ({int(row.n_correct)}/{int(row.n_requests)})")
    print(f"Saved table: {args.output_csv}")
    print(f"Saved figure: {args.output_png}")
    print("========================================")


if __name__ == "__main__":
    main()
