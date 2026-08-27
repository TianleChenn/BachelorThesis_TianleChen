"""Summarize the existing controlled benchmark's four privacy features."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("Method A", "Method B", "Method C")
FEATURES = ("privacy_risk_score", "subject_scope", "data_sensitivity", "disclosure_level")
LEVELS = (0, 1, 2, 3)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _controlled(rows: list[dict]) -> list[dict]:
    """Accept the current `benchmark` column and the requested source alias."""
    return [row for row in rows if (row.get("benchmark") or row.get("source") or "").strip().casefold() == "controlled"]


def _number(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def validate(rows: list[dict]) -> tuple[list[dict], list[str]]:
    warnings = []
    sample_ids = {row.get("sample_id") for row in rows if row.get("sample_id")}
    expected_pairs = {(sample_id, method) for sample_id in sample_ids for method in METHODS}
    actual = Counter((row.get("sample_id"), row.get("method")) for row in rows)
    missing = sorted(expected_pairs - set(actual))
    duplicates = sorted(pair for pair, count in actual.items() if count > 1)
    unknown_methods = sorted({row.get("method") for row in rows} - set(METHODS))
    invalid_levels = sorted({str(row.get("privacy_level")) for row in rows
                             if _number(row.get("privacy_level")) not in LEVELS})
    if len(sample_ids) != 32:
        warnings.append(f"expected 32 unique controlled samples, found {len(sample_ids)}")
    if len(rows) != 96:
        warnings.append(f"expected 96 controlled prediction rows, found {len(rows)}")
    if missing:
        warnings.append("missing sample/method predictions: " + ", ".join(f"{sample}/{method}" for sample, method in missing))
    if duplicates:
        warnings.append("duplicate sample/method predictions: " + ", ".join(f"{sample}/{method}" for sample, method in duplicates))
    if unknown_methods:
        warnings.append("unexpected methods: " + ", ".join(str(value) for value in unknown_methods))
    if invalid_levels:
        warnings.append("invalid privacy levels: " + ", ".join(invalid_levels))
    return rows, warnings


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for method in METHODS:
        for level in LEVELS:
            group = [row for row in rows if row.get("method") == method and _number(row.get("privacy_level")) == level]
            result = {"method": method, "privacy_level": level, "n_samples": len(group)}
            for feature in FEATURES:
                values = [value for value in (_number(row.get(feature)) for row in group) if value is not None]
                result[f"n_valid_{feature}"] = len(values)
                result[f"{feature}_mean"] = statistics.fmean(values) if values else None
                # Sample SD is conventional for descriptive benchmark summaries; N=1 has no defined sample SD.
                result[f"{feature}_std"] = statistics.stdev(values) if len(values) >= 2 else None
            output.append(result)
    return output


def feature_changes(summary: list[dict]) -> list[dict]:
    lookup = {(row["method"], row["privacy_level"]): row for row in summary}
    output = []
    for method in METHODS:
        for feature in FEATURES:
            means = [lookup[(method, level)][f"{feature}_mean"] for level in LEVELS]
            deltas = [None if means[i] is None or means[i + 1] is None else means[i + 1] - means[i]
                      for i in range(3)]
            complete = all(value is not None for value in means)
            output.append({
                "method": method, "feature": feature,
                **{f"level_{level}_mean": means[level] for level in LEVELS},
                "delta_0_1": deltas[0], "delta_1_2": deltas[1], "delta_2_3": deltas[2],
                "delta_0_3": means[3] - means[0] if complete else None,
                "monotonic": all(means[i] <= means[i + 1] for i in range(3)) if complete else None,
            })
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: object) -> str:
    return "NA" if value is None else f"{float(value):.3f}"


def print_summary(summary: list[dict], changes: list[dict], sample_count: int, row_count: int,
                  warnings: list[str]) -> None:
    print("CONTROLLED PRIVACY FEATURE RESPONSE")
    print(f"Controlled samples found: {sample_count}")
    print(f"Prediction rows found: {row_count} (expected: 32 requests x 3 methods = 96)")
    for warning in warnings:
        print(f"WARNING: {warning}")
    change_lookup = {(row["method"], row["feature"]): row for row in changes}
    summary_lookup = {(row["method"], row["privacy_level"]): row for row in summary}
    for feature in FEATURES:
        print(f"\n{feature.replace('_', ' ').title()}")
        for method in METHODS:
            means = [summary_lookup[(method, level)][f"{feature}_mean"] for level in LEVELS]
            change = change_lookup[(method, feature)]
            print(f"{method}: " + ", ".join(f"L{i}={_fmt(value)}" for i, value in enumerate(means)) +
                  f", delta_0_3={_fmt(change['delta_0_3'])}, monotonic={change['monotonic']}")
    print("\nLargest and smallest total increase for each feature")
    for feature in FEATURES:
        available = [row for row in changes if row["feature"] == feature and row["delta_0_3"] is not None]
        if not available:
            print(f"{feature}: unavailable")
            continue
        largest = max(available, key=lambda row: row["delta_0_3"])
        smallest = min(available, key=lambda row: row["delta_0_3"])
        print(f"{feature}: largest {largest['method']} ({_fmt(largest['delta_0_3'])}); "
              f"smallest {smallest['method']} ({_fmt(smallest['delta_0_3'])})")
    non_monotonic = [f"{row['method']} x {row['feature']}" for row in changes if row["monotonic"] is False]
    print("Non-monotonic combinations: " + (", ".join(non_monotonic) if non_monotonic else "none"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=ROOT / "artifacts" / "privacy_benchmark_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    controlled, warnings = validate(_controlled(_read_csv(args.predictions)))
    summary = summarize(controlled)
    changes = feature_changes(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "controlled_privacy_feature_summary.csv", summary)
    _write_csv(args.output_dir / "controlled_privacy_feature_changes.csv", changes)
    payload = {"benchmark": "controlled", "controlled_sample_count": len({row.get('sample_id') for row in controlled}),
               "prediction_row_count": len(controlled), "expected_prediction_row_count": 96,
               "standard_deviation": "sample", "warnings": warnings, "summary": summary}
    (args.output_dir / "controlled_privacy_feature_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_summary(summary, changes, payload["controlled_sample_count"], len(controlled), warnings)


if __name__ == "__main__":
    main()
