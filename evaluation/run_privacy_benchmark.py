"""Run the existing privacy routers on independent and controlled benchmarks."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.build_privacy_controlled_benchmark import build_benchmark
from privacy.llm_soft_gating_model import FEATURE_NAMES
from privacy.method_a_fixed_4d_router import route_with_method_a_fixed_4d
from privacy.llm_method_b_4d_router import route_with_method_b_4d_soft_gating
from privacy.prism_router import prism_route

METHODS = ("Method A", "Method B", "Method C")
FIELDS = ("privacy_risk_score", "subject_scope", "data_sensitivity", "disclosure_level")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _samples(data: dict, source: str) -> list[dict]:
    rows = []
    for sample in data["samples"]:
        rows.append({
            "sample_id": sample["id"], "prompt": sample.get("prompt") or sample["question"],
            "family": sample.get("family") or sample.get("prompt_family") or sample.get("requested_analysis") or "unknown",
            "privacy_level": sample.get("privacy_level"), "ground_truth_route": sample["ground_truth_route"],
            "benchmark": source,
        })
    return rows


def _feature_values(features) -> dict:
    values = list(features or [])
    return {name: values[i] if i < len(values) else None for i, name in enumerate(FEATURE_NAMES)}


def _route(method: str, prompt: str) -> dict:
    start = time.perf_counter()
    if method == "Method A":
        decision = route_with_method_a_fixed_4d(prompt)
        extra = _feature_values(decision.features)
        extra.update(predicted_route=decision.route, blocked_request=decision.hard_blocked,
                     soft_gating_probabilities=decision.probabilities)
    elif method == "Method B":
        decision = route_with_method_b_4d_soft_gating(prompt)
        if not decision.success:
            raise RuntimeError(decision.error or "Method B routing failed")
        extra = _feature_values(decision.features)
        extra.update(predicted_route=decision.route, blocked_request=decision.blocked_request,
                     soft_gating_probabilities=decision.probabilities)
    else:
        decision = prism_route(prompt)
        assessment = decision.llm_privacy_assessment or {}
        extra = {name: assessment.get(name) for name in FIELDS}
        extra.update(predicted_route=decision.route,
                     blocked_request=bool(assessment.get("blocked_request", decision.blocked)),
                     soft_gating_probabilities=decision.probabilities)
    extra["routing_latency_seconds"] = time.perf_counter() - start
    return extra


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--independent", type=Path, default=ROOT / "evaluation" / "frontend_realistic_benchmark_60.json")
    parser.add_argument("--controlled", type=Path, default=ROOT / "evaluation" / "privacy_controlled_benchmark.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "privacy_benchmark_predictions.csv")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    controlled = _load(args.controlled) if args.controlled.exists() else build_benchmark()
    samples = _samples(_load(args.independent), "independent") + _samples(controlled, "controlled")
    if args.limit:
        samples = samples[:args.limit]
    rows, failures = [], []
    for sample in samples:
        for method in METHODS:
            try:
                rows.append({**sample, "method": method, **_route(method, sample["prompt"]), "error": ""})
            except Exception as exc:
                failures.append((sample["sample_id"], method, str(exc)))
                rows.append({**sample, "method": method, "predicted_route": "", "error": f"{type(exc).__name__}: {exc}"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["sample_id", "prompt", "family", "privacy_level", "ground_truth_route", "method", "predicted_route",
               *FIELDS, "blocked_request", "soft_gating_probabilities", "routing_latency_seconds", "benchmark", "error"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["soft_gating_probabilities"] = json.dumps(row.get("soft_gating_probabilities")) if row.get("soft_gating_probabilities") is not None else ""
            writer.writerow(row)
    print(f"Saved {len(rows)} predictions to {args.output} ({len(failures)} failures).")
    if failures:
        for sample_id, method, error in failures[:10]:
            print(f"  {sample_id} / {method}: {error}")


if __name__ == "__main__":
    main()
