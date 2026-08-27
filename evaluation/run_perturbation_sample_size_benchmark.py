"""Controlled sample-size sensitivity benchmark for numerical perturbation."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.perturbation_benchmark_common import (
    BenchmarkContext,
    _correlation_result,
    _primary_table1,
    _primary_table2,
    _variance_result,
    evaluate_one,
    load_benchmark_data,
    perturb,
)
from sports.analysis_noise_utility import _elite_membership, _variance_sample_plan

ANALYSES = (
    "Logistic Regression",
    "Multiple Linear Regression",
    "Correlation Analysis",
    "Variance Analysis",
)
SAMPLE_DESIGN = {
    120: (10, 110),
    180: (15, 165),
    240: (20, 220),
    300: (25, 275),
}
RUN_COLUMNS = (
    "analysis", "sample_size", "elite_count", "semi_elite_count", "sample_id",
    "sample_seed", "perturbation_run", "noise_seed", "noise_amplitude",
    "success", "error", "rmse",
)
SUMMARY_COLUMNS = (
    "analysis", "sample_size", "elite_count", "semi_elite_count", "total_runs",
    "successful_runs", "failed_runs", "failure_rate", "mean_rmse", "sd_rmse",
)


def _context(clean_sample: pd.DataFrame, variance_iterations: int) -> BenchmarkContext:
    plan = _variance_sample_plan(clean_sample, iterations=variance_iterations)
    baselines = {
        "Logistic Regression": _primary_table1(clean_sample),
        "Multiple Linear Regression": _primary_table2(clean_sample),
        "Correlation Analysis": _correlation_result(clean_sample),
        "Variance Analysis": _variance_result(clean_sample, plan),
    }
    return BenchmarkContext(
        original=clean_sample,
        baselines=baselines,
        variance_plan=plan,
        figure2_cohort_indexes=[],
        figure2_selected_indexes=[],
        variance_iterations=variance_iterations,
    )


def _stratified_sample(dataframe: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    elite_required, semi_required = SAMPLE_DESIGN[sample_size]
    elite_mask = _elite_membership(dataframe)
    elite = dataframe.loc[elite_mask]
    semi = dataframe.loc[~elite_mask]
    if len(elite) < elite_required or len(semi) < semi_required:
        raise ValueError(
            f"Cannot create n={sample_size}: requires {elite_required} Elite and "
            f"{semi_required} Semi-elite, but dataset has {len(elite)} and {len(semi)}")
    selected_elite = elite.sample(n=elite_required, replace=False, random_state=seed)
    selected_semi = semi.sample(n=semi_required, replace=False, random_state=seed)
    sample = pd.concat([selected_elite, selected_semi], axis=0).sample(frac=1, random_state=seed)
    counts = _elite_membership(sample).value_counts()
    if len(sample) != sample_size or int(counts.get(True, 0)) != elite_required \
            or int(counts.get(False, 0)) != semi_required:
        raise RuntimeError(f"Stratified sampling invariant failed for n={sample_size}")
    return sample.copy(deep=True)


def _full_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    elite_required, semi_required = SAMPLE_DESIGN[300]
    mask = _elite_membership(dataframe)
    if len(dataframe) != 300 or int(mask.sum()) != elite_required \
            or int((~mask).sum()) != semi_required:
        raise ValueError(
            "The n=300 condition requires the complete dataset with exactly "
            "25 Elite and 275 Semi-elite athletes")
    return dataframe.copy(deep=True)


def summarize(rows: list[dict], sample_sizes: list[int]) -> list[dict]:
    output = []
    for analysis in ANALYSES:
        for sample_size in sample_sizes:
            group = [row for row in rows if row["analysis"] == analysis
                     and row["sample_size"] == sample_size]
            successful = [row for row in group if row["success"] and math.isfinite(float(row["rmse"]))]
            values = [float(row["rmse"]) for row in successful]
            elite_count, semi_count = SAMPLE_DESIGN[sample_size]
            output.append({
                "analysis": analysis,
                "sample_size": sample_size,
                "elite_count": elite_count,
                "semi_elite_count": semi_count,
                "total_runs": len(group),
                "successful_runs": len(successful),
                "failed_runs": len(group) - len(successful),
                "failure_rate": (len(group) - len(successful)) / len(group) if group else float("nan"),
                "mean_rmse": float(np.mean(values)) if values else float("nan"),
                "sd_rmse": (float(np.std(values, ddof=1)) if len(values) > 1 else
                            0.0 if len(values) == 1 else float("nan")),
            })
    return output


def _write_csv(path: Path, rows: list[dict], columns) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_sensitivity(summary: list[dict], sample_sizes: list[int], output_dir: Path) -> list[Path]:
    lookup = {(row["analysis"], row["sample_size"]): row for row in summary}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    panel_letters = ("a", "b", "c", "d")
    for axis, analysis, letter in zip(axes.flat, ANALYSES, panel_letters):
        means = [lookup[(analysis, size)]["mean_rmse"] for size in sample_sizes]
        errors = [lookup[(analysis, size)]["sd_rmse"] for size in sample_sizes]
        axis.errorbar(sample_sizes, means, yerr=errors, marker="o", linewidth=1.8,
                      capsize=4, color="#4C78A8")
        axis.set_title(f"({letter}) {analysis}", fontsize=11)
        axis.set_xlabel("Sample Size")
        axis.set_ylabel("Average Difference (RMSE)")
        axis.set_xticks(sample_sizes)
        axis.grid(axis="y", alpha=.25)
    fig.tight_layout()
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"perturbation_sample_size_sensitivity.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_failures(summary: list[dict], sample_sizes: list[int], output_dir: Path) -> list[Path]:
    if not any(float(row["failure_rate"]) > 0 for row in summary):
        return []
    lookup = {(row["analysis"], row["sample_size"]): row for row in summary}
    fig, axis = plt.subplots(figsize=(7.8, 5.0))
    for analysis in ANALYSES:
        rates = [100 * lookup[(analysis, size)]["failure_rate"] for size in sample_sizes]
        axis.plot(sample_sizes, rates, marker="o", linewidth=1.8, label=analysis)
    axis.set_xlabel("Sample Size")
    axis.set_ylabel("Failure Rate (%)")
    axis.set_xticks(sample_sizes)
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"perturbation_sample_size_failure_rate.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[120, 180, 240, 300])
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--runs-per-sample", type=int, default=10)
    parser.add_argument("--full-sample-runs", type=int, default=100)
    parser.add_argument("--noise-amplitude", type=float, default=.50)
    parser.add_argument("--variance-iterations", type=int, default=1000)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts" / "perturbation_sample_size")
    args = parser.parse_args()
    unsupported = sorted(set(args.sample_sizes) - set(SAMPLE_DESIGN))
    if unsupported:
        parser.error(f"unsupported sample sizes: {unsupported}; use only 120, 180, 240, 300")
    if len(set(args.sample_sizes)) != len(args.sample_sizes):
        parser.error("--sample-sizes must not contain duplicates")
    if min(args.sample_count, args.runs_per_sample, args.full_sample_runs,
           args.variance_iterations) < 1:
        parser.error("sample/run/iteration counts must be positive")
    if not math.isfinite(args.noise_amplitude) or args.noise_amplitude <= 0:
        parser.error("--noise-amplitude must be a positive finite number")
    return args


def main() -> None:
    args = parse_args()
    sample_sizes = list(args.sample_sizes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_data = load_benchmark_data()
    full_snapshot = full_data.copy(deep=True)
    expected_observations = sum(
        args.full_sample_runs if size == 300 else args.sample_count * args.runs_per_sample
        for size in sample_sizes)
    total_perturbations = expected_observations
    completed = 0
    rows = []
    for sample_size in sample_sizes:
        elite_count, semi_count = SAMPLE_DESIGN[sample_size]
        sample_total = 1 if sample_size == 300 else args.sample_count
        for sample_index in range(1, sample_total + 1):
            if sample_size == 300:
                sample_id = "full"
                sample_seed = args.base_seed
                clean = _full_dataset(full_data)
                run_count = args.full_sample_runs
                print("[Sample Size 300] Full Dataset", flush=True)
            else:
                sample_id = sample_index
                sample_seed = args.base_seed + sample_index
                clean = _stratified_sample(full_data, sample_size, sample_seed)
                run_count = args.runs_per_sample
                print(f"[Sample Size {sample_size}] Sample {sample_index:02d}/{args.sample_count:02d}", flush=True)
            clean_snapshot = clean.copy(deep=True)
            context = _context(clean, args.variance_iterations)
            for run_index in range(1, run_count + 1):
                width = 3 if sample_size == 300 else 2
                print(f"  Perturbation {run_index:0{width}d}/{run_count:0{width}d}", flush=True)
                noise_seed = args.base_seed + (0 if sample_size == 300 else sample_index * 10_000) + run_index - 1
                noisy = perturb(clean, amplitude=args.noise_amplitude, seed=noise_seed)
                for analysis in ANALYSES:
                    result = evaluate_one(analysis, noisy, context)
                    rows.append({
                        "analysis": analysis, "sample_size": sample_size,
                        "elite_count": elite_count, "semi_elite_count": semi_count,
                        "sample_id": sample_id, "sample_seed": sample_seed,
                        "perturbation_run": run_index, "noise_seed": noise_seed,
                        "noise_amplitude": args.noise_amplitude, **result,
                    })
                    if not result["success"]:
                        print(f"    WARNING {analysis}: {result['error']}", flush=True)
                completed += 1
                print(f"  Completed {completed} / {total_perturbations} perturbation datasets", flush=True)
            if not clean.equals(clean_snapshot):
                raise RuntimeError(f"Clean sample {sample_size}/{sample_id} was modified")
    if not full_data.equals(full_snapshot):
        raise RuntimeError("The full source dataset was modified")
    expected_results = expected_observations * len(ANALYSES)
    if len(rows) != expected_results:
        raise RuntimeError(f"Expected {expected_results} analysis results; produced {len(rows)}")

    summary = summarize(rows, sample_sizes)
    runs_path = args.output_dir / "perturbation_sample_size_runs.csv"
    summary_path = args.output_dir / "perturbation_sample_size_summary.csv"
    _write_csv(runs_path, rows, RUN_COLUMNS)
    _write_csv(summary_path, summary, SUMMARY_COLUMNS)
    figure_paths = plot_sensitivity(summary, sample_sizes, args.output_dir)
    failure_paths = plot_failures(summary, sample_sizes, args.output_dir)
    failures = sum(not row["success"] for row in rows)
    print(f"Total analysis results: {len(rows)}")
    print(f"Successful results: {len(rows) - failures}")
    print(f"Failed results: {failures}")
    print(f"Saved runs: {runs_path}")
    print(f"Saved summary: {summary_path}")
    for path in [*figure_paths, *failure_paths]:
        print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
