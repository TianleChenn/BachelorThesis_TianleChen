"""Plot separate controlled-level responses for each saved privacy feature."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("Method A", "Method B", "Method C")
LEVELS = (0, 1, 2, 3)
COLORS = {"Method A": "#4C78A8", "Method B": "#F58518", "Method C": "#54A24B"}
FEATURES = {
    "privacy_risk_score": ("Mean Privacy Risk Score", "controlled_privacy_risk_score.png"),
    "subject_scope": ("Mean Subject Scope", "controlled_subject_scope.png"),
    "data_sensitivity": ("Mean Data Sensitivity", "controlled_data_sensitivity.png"),
    "disclosure_level": ("Mean Disclosure Level", "controlled_disclosure_level.png"),
}


def _number(value: object) -> float:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=ROOT / "artifacts" / "controlled_privacy_feature_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    with args.summary.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(row["method"], int(row["privacy_level"])): row for row in rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for feature, (ylabel, filename) in FEATURES.items():
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        for method in METHODS:
            means = [_number(lookup[(method, level)][f"{feature}_mean"]) for level in LEVELS]
            stds = [_number(lookup[(method, level)][f"{feature}_std"]) for level in LEVELS]
            ax.errorbar(LEVELS, means, yerr=stds, marker="o", linewidth=2, capsize=4,
                        markersize=5, label=method, color=COLORS[method])
        ax.set_xlabel("Privacy Condition Level")
        ax.set_ylabel(ylabel)
        ax.set_xticks(LEVELS)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved four controlled privacy feature figures to {args.output_dir}")


if __name__ == "__main__":
    main()
