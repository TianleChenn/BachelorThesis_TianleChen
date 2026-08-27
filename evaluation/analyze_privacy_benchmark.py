"""Calculate thesis privacy-routing metrics from saved benchmark predictions."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTES = ("cloud", "collaboration", "local_edge", "blocked")
LEVEL = {route: i for i, route in enumerate(ROUTES)}


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _divide(a: int, b: int) -> float:
    return a / b if b else 0.0


def calculate(rows: list[dict]) -> dict:
    valid = [r for r in rows if r.get("ground_truth_route") in LEVEL and r.get("predicted_route") in LEVEL]
    n = len(valid)
    exact = sum(r["ground_truth_route"] == r["predicted_route"] for r in valid)
    under = sum(LEVEL[r["predicted_route"]] < LEVEL[r["ground_truth_route"]] for r in valid)
    over = sum(LEVEL[r["predicted_route"]] > LEVEL[r["ground_truth_route"]] for r in valid)
    matrix = [[sum(r["ground_truth_route"] == truth and r["predicted_route"] == pred for r in valid)
               for pred in ROUTES] for truth in ROUTES]
    per_route = {}
    for i, route in enumerate(ROUTES):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(4) if j != i)
        fn = sum(matrix[i][j] for j in range(4) if j != i)
        precision, recall = _divide(tp, tp + fp), _divide(tp, tp + fn)
        per_route[route] = {"precision": precision, "recall": recall,
                            "f1": _divide(2 * precision * recall, precision + recall),
                            "support": sum(matrix[i])}
    return {
        "sample_count": n, "failed_prediction_count": len(rows) - n,
        "exact_route_accuracy": _divide(exact, n), "safety_aware_accuracy": _divide(n - under, n),
        "under_protection_rate": _divide(under, n), "over_protection_rate": _divide(over, n),
        "mean_route_distance": _divide(sum(abs(LEVEL[r["predicted_route"]] - LEVEL[r["ground_truth_route"]]) for r in valid), n),
        "per_route": per_route, "confusion_matrix": {"labels": list(ROUTES), "counts": matrix},
    }


def monotonicity(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        if row.get("benchmark") == "controlled" and row.get("predicted_route") in LEVEL:
            groups[row["family"]].append(row)
    violations = comparisons = 0
    family_results = {}
    for family, family_rows in sorted(groups.items()):
        ordered = sorted(family_rows, key=lambda r: int(r["privacy_level"]))
        family_violations = sum(LEVEL[b["predicted_route"]] < LEVEL[a["predicted_route"]]
                                for a, b in zip(ordered, ordered[1:]))
        pairs = max(0, len(ordered) - 1)
        violations += family_violations
        comparisons += pairs
        family_results[family] = {"violations": family_violations, "adjacent_comparisons": pairs}
    return {"violation_rate": _divide(violations, comparisons), "violations": violations,
            "adjacent_comparisons": comparisons, "by_family": family_results}


def _write_csv(path: Path, results: dict) -> None:
    fields = ["benchmark", "method", "sample_count", "failed_prediction_count", "exact_route_accuracy",
              "safety_aware_accuracy", "under_protection_rate", "over_protection_rate", "mean_route_distance",
              "monotonicity_violation_rate"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for benchmark, methods in results.items():
            for method, metrics in methods.items():
                writer.writerow({key: (benchmark if key == "benchmark" else method if key == "method" else
                    metrics.get("monotonicity", {}).get("violation_rate") if key == "monotonicity_violation_rate" else metrics.get(key)) for key in fields})


def _write_route_csv(path: Path, results: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["benchmark", "method", "route", "precision", "recall", "f1", "support"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for benchmark, methods in results.items():
            for method, metrics in methods.items():
                for route, values in metrics["per_route"].items():
                    writer.writerow({"benchmark": benchmark, "method": method, "route": route, **values})


def _write_confusion_csv(path: Path, results: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["benchmark", "method", "ground_truth_route", *[f"predicted_{route}" for route in ROUTES]]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for benchmark, methods in results.items():
            for method, metrics in methods.items():
                for truth, counts in zip(ROUTES, metrics["confusion_matrix"]["counts"]):
                    writer.writerow({"benchmark": benchmark, "method": method, "ground_truth_route": truth,
                                     **{f"predicted_{route}": count for route, count in zip(ROUTES, counts)}})


def _summary(results: dict) -> None:
    print("Privacy benchmark summary")
    for benchmark, methods in results.items():
        print(f"\n{benchmark.title()}:")
        for method, m in methods.items():
            collab = m["per_route"]["collaboration"]["recall"]
            recalls = {route: values["recall"] for route, values in m["per_route"].items()}
            best = max(recalls, key=recalls.get)
            worst = min(recalls, key=recalls.get)
            note = "no under-protection" if m["under_protection_rate"] == 0 else f"UNDER-PROTECTION {m['under_protection_rate']:.1%}"
            mono = f", monotonicity violations {m['monotonicity']['violation_rate']:.1%}" if "monotonicity" in m else ""
            failures = f"; {m['failed_prediction_count']} failed" if m["failed_prediction_count"] else ""
            print(f"  {method}: exact {m['exact_route_accuracy']:.1%}; best {best.replace('_', ' ')} recall {recalls[best]:.1%}; "
                  f"weakest {worst.replace('_', ' ')} recall {recalls[worst]:.1%}; Collaboration recall {collab:.1%}; {note}{mono}{failures}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=ROOT / "artifacts" / "privacy_benchmark_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    rows = _read(args.predictions)
    results = {}
    for benchmark in ("independent", "controlled"):
        results[benchmark] = {}
        for method in ("Method A", "Method B", "Method C"):
            subset = [r for r in rows if r.get("benchmark") == benchmark and r.get("method") == method]
            metrics = calculate(subset)
            if benchmark == "controlled":
                metrics["monotonicity"] = monotonicity(subset)
            results[benchmark][method] = metrics
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "privacy_benchmark_metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    _write_csv(args.output_dir / "privacy_benchmark_metrics.csv", results)
    _write_route_csv(args.output_dir / "privacy_benchmark_per_route.csv", results)
    _write_confusion_csv(args.output_dir / "privacy_benchmark_confusion_matrices.csv", results)
    _summary(results)


if __name__ == "__main__":
    main()
