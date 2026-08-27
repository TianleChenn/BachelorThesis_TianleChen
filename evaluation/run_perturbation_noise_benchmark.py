"""Run the controlled multi-amplitude numerical perturbation benchmark."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.perturbation_benchmark_common import (
    ANALYSES, METRIC_FIELDS, build_context, evaluate_one, perturb,
)

RUN_COLUMNS = (
    "analysis", "noise_amplitude", "perturbation_run", "random_seed",
    "success", "error", "rmse", "relative_error", "structure_score",
    *METRIC_FIELDS,
)
SUMMARY_METRICS = (
    "rmse", "relative_error", "structure_score", "sign_agreement",
    "significance_agreement", "top3_overlap", "rank_score", "edge_jaccard",
    "group_order_agreement", "highest_domain_retention", "category_agreement",
    "r_squared_change",
)


def _finite(rows: list[dict], field: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def summarize(rows: list[dict], noise_levels: list[float]) -> list[dict]:
    output = []
    for analysis in ANALYSES:
        for amplitude in noise_levels:
            group = [row for row in rows if row["analysis"] == analysis
                     and math.isclose(float(row["noise_amplitude"]), amplitude)]
            successful = [row for row in group if row["success"]]
            summary = {
                "analysis": analysis, "noise_amplitude": amplitude,
                "total_runs": len(group), "successful_runs": len(successful),
                "failed_runs": len(group) - len(successful),
                "failure_rate": (len(group) - len(successful)) / len(group) if group else float("nan"),
            }
            for metric in SUMMARY_METRICS:
                values = _finite(successful, metric)
                summary[f"mean_{metric}"] = float(np.mean(values)) if values else float("nan")
                if metric in {"rmse", "relative_error", "structure_score"}:
                    summary[f"sd_{metric}"] = (
                        float(np.std(values, ddof=1)) if len(values) > 1 else
                        0.0 if len(values) == 1 else float("nan"))
            output.append(summary)
    return output


def _write_csv(path: Path, rows: list[dict], columns=None) -> None:
    columns = list(columns or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _matrix(summary: list[dict], noise_levels: list[float], field: str) -> np.ndarray:
    lookup = {(row["analysis"], float(row["noise_amplitude"])): row for row in summary}
    return np.asarray([[lookup[(analysis, amplitude)][field] for amplitude in noise_levels]
                       for analysis in ANALYSES], dtype=float)


def plot_heatmaps(summary: list[dict], noise_levels: list[float], output_dir: Path) -> None:
    relative = _matrix(summary, noise_levels, "mean_relative_error")
    structure = _matrix(summary, noise_levels, "mean_structure_score")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
    panels = ((relative, "Mean Relative Result Error", "viridis_r", False),
              (structure, "Mean Structure Preservation", "viridis", True))
    for axis, (values, title, cmap, percent) in zip(axes, panels):
        image = axis.imshow(values, aspect="auto", cmap=cmap)
        axis.set_title(title, fontsize=12)
        axis.set_xticks(range(len(noise_levels)), [f"{value:.2f}" for value in noise_levels])
        axis.set_yticks(range(len(ANALYSES)), ANALYSES)
        axis.set_xlabel("Noise Amplitude")
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                label = "NA" if not np.isfinite(value) else f"{value:.1%}" if percent else f"{value:.3f}"
                normalized = image.norm(value) if np.isfinite(value) else .5
                axis.text(column, row, label, ha="center", va="center",
                          color="white" if normalized < .25 or normalized > .75 else "black", fontsize=8)
        fig.colorbar(image, ax=axis, fraction=.046, pad=.04)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"perturbation_noise_task_heatmaps.{suffix}",
                    dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_rmse(summary: list[dict], noise_levels: list[float], output_dir: Path) -> None:
    lookup = {(row["analysis"], float(row["noise_amplitude"])): row for row in summary}
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    for axis, analysis in zip(axes.flat, ANALYSES):
        means = np.asarray([lookup[(analysis, level)]["mean_rmse"] for level in noise_levels], dtype=float)
        errors = np.asarray([lookup[(analysis, level)]["sd_rmse"] for level in noise_levels], dtype=float)
        axis.errorbar(noise_levels, means, yerr=errors, marker="o", linewidth=1.8,
                      capsize=3, color="#4C78A8")
        axis.set_title(analysis, fontsize=11)
        axis.set_xlabel("Noise Amplitude")
        axis.set_ylabel("Average Difference (RMSE)")
        axis.set_xticks(noise_levels)
        axis.grid(axis="y", alpha=.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"perturbation_raw_rmse_small_multiples.{suffix}",
                    dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-levels", nargs="+", type=float,
                        default=[.10, .25, .50, .75, 1.00])
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--variance-iterations", type=int, default=1000)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts" / "perturbation_noise")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if args.variance_iterations < 2:
        parser.error("--variance-iterations must be at least 2")
    if not args.noise_levels or any(not math.isfinite(value) or value <= 0 for value in args.noise_levels):
        parser.error("--noise-levels must contain positive finite amplitudes")
    if len(set(args.noise_levels)) != len(args.noise_levels):
        parser.error("--noise-levels must not contain duplicates")
    return args


def main() -> None:
    args = parse_args()
    noise_levels = list(args.noise_levels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    context = build_context(variance_iterations=args.variance_iterations)
    original_snapshot = context.original.copy(deep=True)
    runs = []
    total_datasets = len(noise_levels) * args.repetitions
    completed = 0
    for amplitude in noise_levels:
        for repetition in range(args.repetitions):
            seed = args.base_seed + repetition
            print(f"[Noise {amplitude:.2f}] Run {repetition + 1:02d}/{args.repetitions:02d}", flush=True)
            perturbed = perturb(context.original, amplitude=amplitude, seed=seed)
            for analysis in ANALYSES:
                result = evaluate_one(analysis, perturbed, context)
                runs.append({
                    "analysis": analysis, "noise_amplitude": amplitude,
                    "perturbation_run": repetition + 1, "random_seed": seed,
                    **result,
                })
                if not result["success"]:
                    print(f"  WARNING {analysis}: {result['error']}", flush=True)
            completed += 1
            print(f"Completed {completed} / {total_datasets} perturbation datasets", flush=True)
    if not context.original.equals(original_snapshot):
        raise RuntimeError("The clean source dataframe was modified during the benchmark")
    expected = len(noise_levels) * len(ANALYSES) * args.repetitions
    if len(runs) != expected:
        raise RuntimeError(f"Expected {expected} analysis results; produced {len(runs)}")

    summary = summarize(runs, noise_levels)
    runs_path = args.output_dir / "perturbation_noise_runs.csv"
    summary_path = args.output_dir / "perturbation_noise_summary.csv"
    _write_csv(runs_path, runs, RUN_COLUMNS)
    _write_csv(summary_path, summary)
    plot_heatmaps(summary, noise_levels, args.output_dir)
    plot_rmse(summary, noise_levels, args.output_dir)
    failures = sum(not row["success"] for row in runs)
    print(f"Completed {len(runs)} analysis results: {len(runs) - failures} successful, {failures} failed")
    print(f"Saved runs: {runs_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
