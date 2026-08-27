"""Evaluate Minimal, Defined, and Full privacy prompts with one frozen pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.model_config import get_strong_model_name
from privacy.evaluation_metrics import PROTECTION_LEVEL, calculate_privacy_metrics
from privacy.prompt_ablation import (
    FEATURE_KEYS,
    PROMPT_VERSIONS,
    evaluate_privacy_prompt,
    prompt_metadata,
)
from privacy.prism_router import get_active_prism_gater_path

OUTPUT_DIR = ROOT / "evaluation" / "results" / "prompt_ablation"
CHECKPOINT = OUTPUT_DIR / "privacy_prompt_ablation_checkpoint.jsonl"
CONTROLLED_CSV = OUTPUT_DIR / "privacy_prompt_ablation_controlled.csv"
INDEPENDENT_CSV = OUTPUT_DIR / "privacy_prompt_ablation_independent.csv"
SUMMARY_JSON = OUTPUT_DIR / "privacy_prompt_ablation_summary.json"
FIGURE_PNG = OUTPUT_DIR / "privacy_prompt_ablation.png"
FIGURE_PDF = OUTPUT_DIR / "privacy_prompt_ablation.pdf"
CONTROLLED = ROOT / "evaluation" / "privacy_controlled_benchmark.json"
INDEPENDENT = ROOT / "evaluation" / "frontend_realistic_benchmark_60.json"


def _load_samples(path: Path, benchmark: str) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("samples") or []
    rows = [{
        "sample_id": item["id"],
        "benchmark": benchmark,
        "privacy_level": item.get("privacy_level"),
        "user_request": item.get("prompt") or item.get("question"),
        "ground_truth_route": item["ground_truth_route"],
    } for item in samples]
    if benchmark == "controlled":
        distribution = Counter(row["ground_truth_route"] for row in rows)
        if len(rows) != 32 or distribution != Counter({route: 8 for route in PROTECTION_LEVEL}):
            raise ValueError("Controlled benchmark must contain 32 samples, eight per route.")
    if benchmark == "independent" and len(rows) != 60:
        raise ValueError("Independent benchmark must contain exactly 60 samples.")
    if benchmark == "independent":
        clean = {key: value for key, value in data.items() if key != "dataset_sha256"}
        digest = hashlib.sha256(json.dumps(clean, sort_keys=True, ensure_ascii=False,
            separators=(",", ":")).encode("utf-8")).hexdigest()
        if data.get("dataset_sha256") != digest or data.get("locked") is not True:
            raise ValueError("Independent benchmark must be locked and match its SHA256.")
        if data.get("used_for_training") is not False or data.get("used_for_threshold_calibration") is not False:
            raise ValueError("Independent benchmark must remain evaluation-only.")
    return data, rows


def _read_checkpoint() -> list[dict]:
    if not CHECKPOINT.exists():
        return []
    return [json.loads(line) for line in CHECKPOINT.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _append_checkpoint(row: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _latest(rows: list[dict]) -> list[dict]:
    latest = {}
    for row in rows:
        latest[(row["benchmark"], row["prompt_version"], row["sample_id"])] = row
    return list(latest.values())


def _require_complete(rows: list[dict]) -> None:
    """Reject thesis summaries that omit infrastructure/model failures."""
    expected_pairs = {
        ("controlled", version): 32 for version in PROMPT_VERSIONS
    }
    expected_pairs.update({
        ("independent", version): 60 for version in PROMPT_VERSIONS
    })

    incomplete = []
    for (benchmark, version), expected in expected_pairs.items():
        selected = [
            row for row in rows
            if row["benchmark"] == benchmark
            and row["prompt_version"] == version
            and row.get("assessment_success")
            and row.get("predicted_route")
        ]
        if len(selected) != expected:
            incomplete.append(f"{benchmark}/{version}: {len(selected)}/{expected}")

    if incomplete:
        print("\nERROR: Privacy Prompt Ablation is incomplete.")
        print("The following groups still contain API/model failures:")
        for item in incomplete:
            print("  -", item)
        print("\nRun again with:")
        print("python scripts/evaluate_privacy_prompt_ablation.py --resume")
        raise RuntimeError(
            "Incomplete Privacy Prompt Ablation. "
            "Do not use partial results in the thesis."
        )


def _summarize(rows: list[dict], expected_count: int) -> dict:
    successful = [row for row in rows if row.get("assessment_success") and row.get("predicted_route")]
    metric_input = [{"ground_truth_route": row["ground_truth_route"],
                     "predicted_route": row["predicted_route"]} for row in successful]
    metrics = calculate_privacy_metrics(metric_input) if metric_input else None
    per_route = (metrics or {}).get("per_route_metrics", {})
    controlled_levels = {}
    for level in range(4):
        selected = [row for row in successful if str(row.get("privacy_level")) == str(level)]
        controlled_levels[f"L{level}"] = (
            sum(row["predicted_route"] == row["ground_truth_route"] for row in selected) / len(selected)
            if selected else None
        )
    return {
        "expected_samples": expected_count,
        "completed_samples": len(successful),
        "api_failures": sum(not row.get("api_call_success") for row in rows),
        "model_or_parse_failures": sum(row.get("api_call_success") and not row.get("assessment_success") for row in rows),
        "complete": len(successful) == expected_count,
        "exact_route_accuracy": (metrics or {}).get("exact_route_match_rate"),
        "recall": {route: per_route.get(route, {}).get("recall") for route in PROTECTION_LEVEL},
        "under_protection_rate": ((metrics or {}).get("privacy_safety_metrics") or {}).get("underprotection_rate"),
        "exact_route_accuracy_by_level": controlled_levels,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = ["sample_id", "benchmark", "privacy_level", "ground_truth_route", "prompt_version",
               "prompt_sha256", *FEATURE_KEYS, "blocked_request", "predicted_route", "exact_correct",
               "under_protection", "assessment_success", "api_call_success", "api_error", "model_error",
               "requested_model", "actual_model", "provider", "latency_seconds", "raw_model_output"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot(summary: dict) -> None:
    import matplotlib.pyplot as plt
    labels = [name.replace("_", " ").title() for name in PROMPT_VERSIONS]
    x = range(len(labels)); width = .34
    controlled = [(summary["benchmarks"]["controlled"][v]["exact_route_accuracy"] or 0) * 100 for v in PROMPT_VERSIONS]
    independent = [(summary["benchmarks"]["independent"][v]["exact_route_accuracy"] or 0) * 100 for v in PROMPT_VERSIONS]
    fig, ax = plt.subplots(figsize=(8, 5))
    first = ax.bar([i-width/2 for i in x], controlled, width, label="Controlled")
    second = ax.bar([i+width/2 for i in x], independent, width, label="Independent")
    ax.bar_label(first, fmt="%.1f%%", padding=3); ax.bar_label(second, fmt="%.1f%%", padding=3)
    ax.set_xticks(list(x), labels); ax.set_ylim(0, 105); ax.set_ylabel("Exact Route Accuracy (%)")
    ax.set_title("Privacy Prompt Design Ablation"); ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(FIGURE_PNG, dpi=300); fig.savefig(FIGURE_PDF); plt.close(fig)


def _percent(value) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def print_summary(summary: dict) -> None:
    print("=" * 62); print("Privacy Prompt Design Ablation"); print("=" * 62)
    for benchmark in ("controlled", "independent"):
        print(f"\n{benchmark.title()} Benchmark")
        print(f"{'Prompt':12} {'Exact':>8} {'Cloud':>8} {'Collab':>8} {'Local':>8} {'Blocked':>8} {'Under':>8}")
        for version in PROMPT_VERSIONS:
            row = summary["benchmarks"][benchmark][version]; recall = row["recall"]
            print(f"{version.title():12} {_percent(row['exact_route_accuracy']):>8} {_percent(recall['cloud']):>8} "
                  f"{_percent(recall['collaboration']):>8} {_percent(recall['local_edge']):>8} "
                  f"{_percent(recall['blocked']):>8} {_percent(row['under_protection_rate']):>8}")
        if benchmark == "controlled":
            print("\nExact Route Accuracy by Privacy Level")
            print(f"{'Prompt':12} {'L0 Cloud':>10} {'L1 Collab':>10} {'L2 Local':>10} {'L3 Blocked':>11}")
            for version in PROMPT_VERSIONS:
                levels = summary["benchmarks"][benchmark][version]["exact_route_accuracy_by_level"]
                print(f"{version.title():12} " + " ".join(f"{_percent(levels[f'L{i}']):>10}" for i in range(4)))


def run(*, fresh: bool, resume: bool, limit: int | None = None) -> dict:
    controlled_meta, controlled = _load_samples(CONTROLLED, "controlled")
    independent_meta, independent = _load_samples(INDEPENDENT, "independent")
    all_samples = controlled + independent
    if limit is not None:
        all_samples = all_samples[:limit]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if fresh:
        for path in (CHECKPOINT, CONTROLLED_CSV, INDEPENDENT_CSV, SUMMARY_JSON, FIGURE_PNG, FIGURE_PDF):
            if path.exists(): path.unlink()
    if CHECKPOINT.exists() and not (fresh or resume):
        raise RuntimeError("A prompt-ablation checkpoint exists. Use --resume or --fresh.")
    prior = _read_checkpoint() if resume else []
    active_model = get_strong_model_name()
    active_hashes = {version: prompt_metadata()[version]["sha256"] for version in PROMPT_VERSIONS}
    if any(row.get("requested_model") != active_model for row in prior):
        raise RuntimeError("Checkpoint model differs from the active Privacy Assessor model; use --fresh.")
    if any(row.get("prompt_sha256") != active_hashes.get(row.get("prompt_version")) for row in prior):
        raise RuntimeError("Checkpoint prompt hash differs from the active ablation prompt; use --fresh.")
    completed = {(row["benchmark"], row["prompt_version"], row["sample_id"])
                 for row in prior
                 if row.get("assessment_success") and row.get("predicted_route")}
    rows = list(prior)
    print(f"Provider: OpenAI privacy assessor\nModel: {active_model}")
    print(f"Benchmark samples: controlled=32, independent=60")
    for version in PROMPT_VERSIONS:
        print(f"Active prompt version: {version}")
        for sample in all_samples:
            key = (sample["benchmark"], version, sample["sample_id"])
            if key in completed: continue
            result = {**sample, **evaluate_privacy_prompt(version, sample["user_request"])}
            if result.get("assessment_success"):
                result["exact_correct"] = result["predicted_route"] == result["ground_truth_route"]
                result["under_protection"] = PROTECTION_LEVEL[result["predicted_route"]] < PROTECTION_LEVEL[result["ground_truth_route"]]
            _append_checkpoint(result); rows.append(result)
    rows = _latest(rows)
    for benchmark, path in (("controlled", CONTROLLED_CSV), ("independent", INDEPENDENT_CSV)):
        _write_csv(path, [row for row in rows if row["benchmark"] == benchmark])
    # Final evaluation artifacts are only valid when every sample completed.
    _require_complete(rows)
    summary = {
        "evaluation_type": "privacy_prompt_design_ablation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": active_model, "provider": "openai_privacy_assessor",
        "temperature": 0.0, "max_tokens": 500,
        "gating_model_path": str(get_active_prism_gater_path()),
        "gating_model_sha256": hashlib.sha256(get_active_prism_gater_path().read_bytes()).hexdigest(),
        "prompts": prompt_metadata(),
        "datasets": {"controlled": controlled_meta.get("dataset_name"), "independent": independent_meta.get("dataset_name")},
        "benchmarks": {benchmark: {version: _summarize(
            [row for row in rows if row["benchmark"] == benchmark and row["prompt_version"] == version],
            32 if benchmark == "controlled" else 60) for version in PROMPT_VERSIONS}
            for benchmark in ("controlled", "independent")},
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _plot(summary); print_summary(summary); return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    lifecycle = parser.add_mutually_exclusive_group()
    lifecycle.add_argument("--fresh", action="store_true"); lifecycle.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Development-only limit across both benchmarks.")
    args = parser.parse_args(argv)
    run(fresh=args.fresh, resume=args.resume, limit=args.limit); return 0


if __name__ == "__main__":
    raise SystemExit(main())
