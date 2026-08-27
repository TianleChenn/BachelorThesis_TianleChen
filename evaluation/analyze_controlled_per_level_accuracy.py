"""Analyze exact routing accuracy by level for the existing controlled benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHOD_ORDER = ["Method A", "Method B", "Method C"]
LEVEL_ORDER = ["0", "1", "2", "3"]
LEVEL_LABELS = {
    "0": "L0 Cloud",
    "1": "L1 Collaboration",
    "2": "L2 Local Edge",
    "3": "L3 Blocked",
}


def _controlled_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    frame = pd.read_csv(path, dtype={"privacy_level": "string"})
    required = {"sample_id", "benchmark", "privacy_level", "method",
                "ground_truth_route", "predicted_route"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    controlled = frame[frame["benchmark"].astype(str).str.strip().str.casefold().eq("controlled")].copy()
    controlled["privacy_level"] = controlled["privacy_level"].astype(str).str.strip()
    controlled["correct"] = (
        controlled["predicted_route"].astype(str).str.strip().str.casefold()
        == controlled["ground_truth_route"].astype(str).str.strip().str.casefold()
    )
    return controlled


def _warnings(controlled: pd.DataFrame) -> list[str]:
    warnings = []
    sample_count = controlled["sample_id"].nunique()
    expected_rows = sample_count * len(METHOD_ORDER)
    if sample_count != 32:
        warnings.append(f"expected 32 unique controlled samples but found {sample_count}")
    if len(controlled) != expected_rows:
        warnings.append(f"expected {expected_rows} prediction rows but found {len(controlled)}")
    counts = controlled.groupby(["sample_id", "method"]).size()
    duplicates = counts[counts > 1]
    if not duplicates.empty:
        warnings.append(f"found {len(duplicates)} duplicate sample/method combinations")
    expected_pairs = pd.MultiIndex.from_product(
        [controlled["sample_id"].dropna().unique(), METHOD_ORDER], names=["sample_id", "method"])
    missing_pairs = expected_pairs.difference(counts.index)
    if len(missing_pairs):
        warnings.append(f"missing {len(missing_pairs)} sample/method combinations")
    unexpected_methods = sorted(set(controlled["method"].dropna()) - set(METHOD_ORDER))
    if unexpected_methods:
        warnings.append(f"unexpected methods: {unexpected_methods}")
    return warnings


def summarize(controlled: pd.DataFrame) -> pd.DataFrame:
    summary = controlled.groupby(["method", "privacy_level"], as_index=False).agg(
        n_requests=("correct", "size"), n_correct=("correct", "sum"), accuracy=("correct", "mean"))
    summary["accuracy_percent"] = summary["accuracy"] * 100
    summary["level_label"] = summary["privacy_level"].map(LEVEL_LABELS)
    summary["_method_rank"] = summary["method"].map({value: i for i, value in enumerate(METHOD_ORDER)})
    summary["_level_rank"] = summary["privacy_level"].map({value: i for i, value in enumerate(LEVEL_ORDER)})
    return summary.sort_values(["_method_rank", "_level_rank"]).drop(
        columns=["_method_rank", "_level_rank"]).reset_index(drop=True)


def plot(summary: pd.DataFrame, output: Path) -> None:
    pivot = summary.pivot(index="privacy_level", columns="method", values="accuracy_percent").reindex(LEVEL_ORDER)
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    ax = pivot[METHOD_ORDER].plot(kind="bar", figsize=(9, 5.5), width=.75, color=colors)
    ax.set_xlabel("Controlled Privacy Condition")
    ax.set_ylabel("Exact Routing Accuracy (%)")
    ax.set_title("Routing Accuracy by Controlled Privacy Condition", pad=58)
    ax.set_xticklabels([LEVEL_LABELS[level] for level in LEVEL_ORDER], rotation=0)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=.25)
    ax.legend(title="Method", loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False)
    for container in ax.containers:
        ax.bar_label(container, labels=["" if pd.isna(bar.get_height()) else f"{bar.get_height():.1f}%"
                                        for bar in container], padding=3, fontsize=9)
    ax.figure.tight_layout()
    ax.figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path,
                        default=ROOT / "artifacts" / "privacy_benchmark_predictions.csv")
    parser.add_argument("--output-csv", type=Path,
                        default=ROOT / "artifacts" / "controlled_per_level_accuracy.csv")
    parser.add_argument("--output-png", type=Path,
                        default=ROOT / "artifacts" / "controlled_per_level_accuracy.png")
    args = parser.parse_args()
    controlled = _controlled_predictions(args.predictions)
    warnings = _warnings(controlled)
    summary = summarize(controlled)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    plot(summary, args.output_png)

    print("CONTROLLED PER-LEVEL ROUTING ACCURACY")
    print(f"Controlled prediction rows found: {len(controlled)}")
    print(f"Controlled samples found: {controlled['sample_id'].nunique()}")
    print(f"Expected prediction rows: 32 requests x 3 methods = 96")
    for warning in warnings:
        print(f"WARNING: {warning}")
    print("\nPer-Level Accuracy")
    for method in METHOD_ORDER:
        print(f"\n{method}")
        method_rows = summary[summary["method"] == method]
        for level in LEVEL_ORDER:
            row = method_rows[method_rows["privacy_level"] == level]
            if row.empty:
                print(f"  {LEVEL_LABELS[level]}: missing")
            else:
                value = row.iloc[0]
                print(f"  {LEVEL_LABELS[level]}: {value['accuracy_percent']:.1f}% "
                      f"({int(value['n_correct'])}/{int(value['n_requests'])})")
    print(f"\nSaved table: {args.output_csv}")
    print(f"Saved figure: {args.output_png}")


if __name__ == "__main__":
    main()
