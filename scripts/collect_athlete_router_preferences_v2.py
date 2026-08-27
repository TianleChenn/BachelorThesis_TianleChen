"""Collect real Strong/Weak/Judge preferences for the v2 training requests."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.env import load_local_env
from llm.model_clients import call_weak_model
from scripts.athlete_router_evaluation_common import (
    call_strong_model,
    get_athlete_router_evaluation_models,
    judge_comparison,
    safe_error,
    validate_model_access,
)


DATASET = ROOT / "evaluation" / "athlete_router_training_prompts_v2_100.json"
RAW_OUTPUT = ROOT / "artifacts" / "athlete_router_preferences_v2_raw.json"
VALID_OUTPUT = ROOT / "artifacts" / "athlete_router_preferences_v2_valid.json"


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    temporary.replace(path)


def _load_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["sample_id"]): row for row in payload.get("results", [])}


def _evaluate(sample: dict, models) -> dict:
    prompt = sample["prompt"]
    weak = call_weak_model([{"role": "user", "content": prompt}], max_tokens=900)
    strong = call_strong_model(models, [{"role": "user", "content": prompt}], max_tokens=900)
    base = {
        "sample_id": sample["id"],
        "prompt": prompt,
        "analysis_type": sample["analysis_type"],
        "difficulty": sample["difficulty"],
        "prompt_family": sample["prompt_family"],
        "expected_complexity_reason": sample["expected_complexity_reason"],
        "strong_model": models.strong_model,
        "weak_model": models.weak_model,
        "judge_model": models.judge_model,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "strong_response": strong.content if strong.success else None,
        "weak_response": weak.content if weak.success else None,
        "strong_call_success": bool(strong.success),
        "weak_call_success": bool(weak.success),
    }
    if not strong.success or not weak.success or not strong.content or not weak.content:
        errors = [value for value in (strong.error, weak.error) if value]
        return {**base, "status": "invalid", "preference": "invalid_or_tie",
                "strong_score": None, "weak_score": None, "judge_result": None,
                "error": safe_error("; ".join(errors) or "Model response missing")}
    judged = judge_comparison(prompt, weak.content, strong.content, models)
    if judged is None:
        return {**base, "status": "invalid", "preference": "invalid_or_tie",
                "strong_score": None, "weak_score": None, "judge_result": None,
                "error": "Judge result was missing or invalid"}
    label, weak_score, strong_score, judge_result = judged
    preference = "strong" if label == "strong_win" else "weak" if label == "weak_win" else "invalid_or_tie"
    return {
        **base,
        "status": "valid" if preference in {"strong", "weak"} else "invalid",
        "preference": preference,
        "strong_score": float(strong_score),
        "weak_score": float(weak_score),
        "judge_result": judge_result,
        "error": None if preference in {"strong", "weak"} else "Judge preference was tied",
    }


def _summaries(samples: list[dict], rows: list[dict]) -> tuple[dict, list[dict]]:
    by_id = {str(row["sample_id"]): row for row in rows}
    ordered = [by_id[str(sample["id"])] for sample in samples if str(sample["id"]) in by_id]
    valid = [row for row in ordered if row.get("preference") in {"strong", "weak"}]
    strong = sum(row["preference"] == "strong" for row in valid)
    weak = sum(row["preference"] == "weak" for row in valid)
    invalid = len(ordered) - len(valid)
    difficulty = defaultdict(lambda: Counter(strong=0, weak=0, invalid=0))
    analysis = defaultdict(lambda: Counter(strong=0, weak=0, invalid=0))
    for row in ordered:
        bucket = row["preference"] if row.get("preference") in {"strong", "weak"} else "invalid"
        difficulty[row["difficulty"]][bucket] += 1
        analysis[row["analysis_type"]][bucket] += 1
    strong_pct = strong / len(valid) if valid else 0.0
    weak_pct = weak / len(valid) if valid else 0.0
    balance = "ideal" if valid and .4 <= strong_pct <= .6 else "acceptable" if valid and .35 <= strong_pct <= .65 else "warning"
    summary = {
        "candidate_requests": len(samples),
        "processed_requests": len(ordered),
        "valid_preferences": len(valid),
        "strong_wins": strong,
        "weak_wins": weak,
        "invalid_or_ties": invalid,
        "strong_percentage": strong_pct,
        "weak_percentage": weak_pct,
        "balance_diagnostic": balance,
        "distribution_by_difficulty": {key: dict(value) for key, value in sorted(difficulty.items())},
        "distribution_by_analysis_type": {key: dict(value) for key, value in sorted(analysis.items())},
    }
    return summary, valid


def _print_summary(summary: dict) -> None:
    print(f"Candidate requests: {summary['candidate_requests']}")
    print(f"Valid preferences: {summary['valid_preferences']}")
    print(f"Strong wins: {summary['strong_wins']}")
    print(f"Weak wins: {summary['weak_wins']}")
    print(f"Invalid/ties: {summary['invalid_or_ties']}")
    print(f"Strong percentage: {summary['strong_percentage']:.1%}")
    print(f"Weak percentage: {summary['weak_percentage']:.1%}")
    print("\nDifficulty    Strong    Weak    Invalid")
    for key in ("simple", "medium", "hard"):
        row = summary["distribution_by_difficulty"].get(key, {})
        print(f"{key.title():<13}{row.get('strong', 0):>6}{row.get('weak', 0):>8}{row.get('invalid', 0):>11}")
    print("\nAnalysis type distributions:")
    for key, row in summary["distribution_by_analysis_type"].items():
        print(f"{key}: Strong={row.get('strong', 0)}, Weak={row.get('weak', 0)}, Invalid={row.get('invalid', 0)}")
    print(f"\nBalance diagnostic: {summary['balance_diagnostic'].upper()}")


def collect(
    dataset_path: Path = DATASET,
    raw_output: Path = RAW_OUTPUT,
    valid_output: Path = VALID_OUTPUT,
    *,
    resume: bool = True,
    limit: int | None = None,
    max_retries: int = 2,
) -> dict:
    load_local_env()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    samples = list(dataset.get("samples") or [])
    if len(samples) != 100:
        raise RuntimeError(f"Expected 100 candidate requests; found {len(samples)}")
    selected = samples if limit is None else samples[:limit]
    models = get_athlete_router_evaluation_models()
    cached = _load_rows(raw_output) if resume else {}
    remaining = [sample for sample in selected if str(sample["id"]) not in cached]
    if remaining:
        validate_model_access(models)
    rows = dict(cached)
    for index, sample in enumerate(selected, 1):
        sample_id = str(sample["id"])
        if sample_id in rows:
            continue
        result = None
        for attempt in range(1, max_retries + 2):
            try:
                result = _evaluate(sample, models)
            except Exception as exc:
                result = {**sample, "sample_id": sample_id, "status": "invalid",
                          "preference": "invalid_or_tie", "error": _safe_error(exc)}
            result["attempts"] = attempt
            if result.get("status") == "valid" or attempt > max_retries:
                break
            time.sleep(2 ** attempt)
        rows[sample_id] = result
        ordered = [rows[str(item["id"])] for item in samples if str(item["id"]) in rows]
        summary, valid = _summaries(samples, ordered)
        _atomic_write(raw_output, {"dataset_name": dataset["dataset_name"], "summary": summary, "results": ordered})
        _atomic_write(valid_output, {"dataset_name": "athlete_router_preferences_v2_valid",
                                     "source": str(raw_output.relative_to(ROOT)).replace("\\", "/"),
                                     "summary": summary, "results": valid})
        print(f"{index}/100 | valid={summary['valid_preferences']} | invalid/ties={summary['invalid_or_ties']}", flush=True)
    ordered = [rows[str(item["id"])] for item in samples if str(item["id"]) in rows]
    summary, valid = _summaries(samples, ordered)
    _atomic_write(raw_output, {"dataset_name": dataset["dataset_name"], "summary": summary, "results": ordered})
    _atomic_write(valid_output, {"dataset_name": "athlete_router_preferences_v2_valid",
                                 "source": str(raw_output.relative_to(ROOT)).replace("\\", "/"),
                                 "summary": summary, "results": valid})
    _print_summary(summary)
    return summary


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Collect real v2 Strong/Weak preferences.")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--raw-output", type=Path, default=RAW_OUTPUT)
    parser.add_argument("--valid-output", type=Path, default=VALID_OUTPUT)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-retries", type=int, default=2)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    collect(args.dataset, args.raw_output, args.valid_output,
            resume=args.resume, limit=args.limit, max_retries=args.max_retries)
