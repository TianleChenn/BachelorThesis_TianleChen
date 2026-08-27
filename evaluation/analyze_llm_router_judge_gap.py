"""Analyze the saved cost-aware router evaluation by absolute judge-score gap."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts" / "athlete_strong_weak_router_evaluation.json"
DEFAULT_CSV = ROOT / "artifacts" / "llm_router_accuracy_by_judge_gap.csv"
DEFAULT_PNG = ROOT / "artifacts" / "llm_router_accuracy_by_judge_gap.png"
GROUPS = ("Gap = 0", "Gap = 2", "Gap >= 3")
EXPECTED_GAPS = {0.0, 2.0, 3.0, 4.0, 5.0}


def _group(gap: float) -> str:
    if gap == 0:
        return "Gap = 0"
    if gap == 2:
        return "Gap = 2"
    if gap >= 3:
        return "Gap >= 3"
    raise ValueError(f"Judge-score gap {gap:g} does not belong to a requested group")


def analyze(path: Path) -> list[dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    samples = report.get("per_sample_results")
    if report.get("independent_evaluation") is not True or not isinstance(samples, list):
        raise ValueError("Expected a saved independent evaluation with per_sample_results")
    if len(samples) != 40:
        raise ValueError(f"Expected exactly 40 evaluation samples; found {len(samples)}")
    grouped = {name: [] for name in GROUPS}
    observed_gaps = set()
    for index, sample in enumerate(samples):
        construction = sample.get("ground_truth_construction") or {}
        try:
            strong = float(construction["strong_model_score"])
            weak = float(construction["weak_model_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Sample {index} has invalid judge scores") from exc
        if not math.isfinite(strong) or not math.isfinite(weak):
            raise ValueError(f"Sample {index} has non-finite judge scores")
        gap = abs(strong - weak)
        observed_gaps.add(gap)
        truth = str(sample.get("ground_truth") or "").strip().casefold()
        if truth not in {"strong", "weak"}:
            raise ValueError(f"Sample {index} has invalid ground_truth: {truth!r}")
        if not isinstance(sample.get("correct"), bool):
            raise ValueError(f"Sample {index} has no Boolean correct value")
        grouped[_group(gap)].append({"correct": sample["correct"], "ground_truth": truth})
    if observed_gaps != EXPECTED_GAPS:
        raise ValueError(f"Expected judge-score gaps {sorted(EXPECTED_GAPS)}; found {sorted(observed_gaps)}")

    output = []
    for name in GROUPS:
        rows = grouped[name]
        requests = len(rows)
        correct = sum(row["correct"] for row in rows)
        output.append({
            "gap_group": name,
            "requests": requests,
            "correct": correct,
            "routing_accuracy": correct / requests if requests else None,
            "routing_accuracy_percent": 100 * correct / requests if requests else None,
            "ground_truth_strong": sum(row["ground_truth"] == "strong" for row in rows),
            "ground_truth_weak": sum(row["ground_truth"] == "weak" for row in rows),
        })
    if sum(row["requests"] for row in output) != 40:
        raise RuntimeError("The three requested judge-gap groups do not contain exactly 40 samples")
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = ("gap_group", "requests", "correct", "routing_accuracy",
               "routing_accuracy_percent", "ground_truth_strong", "ground_truth_weak")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, rows: list[dict]) -> None:
    labels = [f"{row['gap_group']}\nn = {row['requests']}" for row in rows]
    values = [row["routing_accuracy_percent"] for row in rows]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    bars = ax.bar(labels, values, color="#4C78A8", width=.62)
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=3, fontsize=10)
    ax.set_title("Cost-Aware Router Accuracy by Strong–Weak Judge Score Gap", pad=14)
    ax.set_xlabel("Strong–Weak Judge Score Gap")
    ax.set_ylabel("Routing Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_PNG)
    args = parser.parse_args()
    rows = analyze(args.input)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_csv, rows)
    plot(args.output_png, rows)

    print("========================================")
    print("ROUTER ACCURACY BY JUDGE SCORE GAP")
    print("========================================")
    for row in rows:
        print(f"{row['gap_group']}: {row['routing_accuracy_percent']:.1f}% "
              f"({row['correct']}/{row['requests']})")
    total = sum(row["requests"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    print(f"\nTotal samples: {total}")
    print(f"Total correct: {correct}/{total}")
    print(f"\nSaved table: {args.output_csv}")
    print(f"Saved figure: {args.output_png}")


if __name__ == "__main__":
    main()
