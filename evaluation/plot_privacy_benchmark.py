"""Create thesis figures for the privacy-routing benchmark."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("Method A", "Method B", "Method C")
ROUTES = ("cloud", "collaboration", "local_edge", "blocked")
LABELS = ("Cloud", "Collaboration", "Local Edge", "Blocked")
LEVEL = {route: i for i, route in enumerate(ROUTES)}
COLORS = ("#4C78A8", "#F58518", "#54A24B")


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _bars(ax, categories, series, ylabel):
    x, width = np.arange(len(categories)), .24
    for i, method in enumerate(METHODS):
        ax.bar(x + (i - 1) * width, series[method], width, label=method, color=COLORS[i])
    ax.set_xticks(x, categories)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=ROOT / "artifacts" / "privacy_benchmark_predictions.csv")
    parser.add_argument("--metrics", type=Path, default=ROOT / "artifacts" / "privacy_benchmark_metrics.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    with args.predictions.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    combined = {m: [r for r in rows if r["method"] == m and r["predicted_route"] in LEVEL] for m in METHODS}
    overall = {}
    for method, values in combined.items():
        n = len(values)
        overall[method] = [sum(r["predicted_route"] == r["ground_truth_route"] for r in values) / n,
                           sum(LEVEL[r["predicted_route"]] < LEVEL[r["ground_truth_route"]] for r in values) / n,
                           sum(LEVEL[r["predicted_route"]] > LEVEL[r["ground_truth_route"]] for r in values) / n]
    fig, ax = plt.subplots(figsize=(7.2, 4.3)); _bars(ax, ["Exact Accuracy", "Under-protection", "Over-protection"], overall, "Rate")
    _save(fig, args.output_dir / "privacy_overall_metrics.png")

    recall = {m: [] for m in METHODS}
    for method in METHODS:
        for route in ROUTES:
            subset = [r for r in combined[method] if r["ground_truth_route"] == route]
            recall[method].append(sum(r["predicted_route"] == route for r in subset) / len(subset) if subset else 0)
    fig, ax = plt.subplots(figsize=(7.2, 4.3)); _bars(ax, LABELS, recall, "Recall")
    _save(fig, args.output_dir / "privacy_per_route_recall.png")

    method_c = combined["Method C"]
    matrix = np.array([[sum(r["ground_truth_route"] == truth and r["predicted_route"] == pred for r in method_c)
                        for pred in ROUTES] for truth in ROUTES], dtype=float)
    matrix = np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix), where=matrix.sum(axis=1, keepdims=True) != 0)
    fig, ax = plt.subplots(figsize=(5.7, 4.8)); image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4): ax.text(j, i, f"{matrix[i,j]:.0%}", ha="center", va="center", color="white" if matrix[i,j] > .55 else "black")
    ax.set_xticks(range(4), LABELS, rotation=25, ha="right"); ax.set_yticks(range(4), LABELS)
    ax.set_xlabel("Predicted route"); ax.set_ylabel("Ground-truth route"); fig.colorbar(image, ax=ax, label="Row-normalized proportion")
    _save(fig, args.output_dir / "privacy_confusion_method_c.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for method, color in zip(METHODS, COLORS):
        values = [r for r in combined[method] if r["benchmark"] == "controlled"]
        means = [np.mean([LEVEL[r["predicted_route"]] for r in values if int(r["privacy_level"]) == level]) for level in range(4)]
        ax.plot(range(4), means, marker="o", linewidth=2, label=method, color=color)
    ax.plot(range(4), range(4), "--", color="black", label="Ideal y=x")
    ax.set_xticks(range(4), [f"Privacy Level {i}" for i in range(4)]); ax.set_yticks(range(4), LABELS)
    ax.set_ylabel("Average predicted protection level"); ax.grid(alpha=.25); ax.legend(frameon=False)
    _save(fig, args.output_dir / "privacy_controlled_ladder.png")

    features = ("privacy_risk_score", "subject_scope", "data_sensitivity", "disclosure_level")
    c_rows = [r for r in rows if r["benchmark"] == "controlled" and r["method"] == "Method C"]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for feature, color in zip(features, ("#4C78A8", "#F58518", "#E45756", "#72B7B2")):
        means = []
        for level in range(4):
            vals = [float(r[feature]) for r in c_rows if int(r["privacy_level"]) == level and r.get(feature) not in (None, "")]
            means.append(np.mean(vals) if vals else np.nan)
        ax.plot(range(4), means, marker="o", linewidth=2, label=feature.replace("_", " ").title(), color=color)
    ax.set_xticks(range(4), [f"Privacy Level {i}" for i in range(4)]); ax.set_ylabel("Mean feature value"); ax.set_ylim(0, 1)
    ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8)
    _save(fig, args.output_dir / "privacy_feature_response_method_c.png")
    print(f"Saved five privacy benchmark figures to {args.output_dir}")


if __name__ == "__main__":
    main()
