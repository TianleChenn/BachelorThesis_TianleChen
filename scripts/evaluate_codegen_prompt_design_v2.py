"""First-pass Restricted Code Generation Prompt Design Evaluation V2."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.code_generation_prompt_design_v2 import (
    PROMPT_DISPLAY_NAMES,
    PROMPT_VERSIONS,
    build_codegen_prompt_design_v2_messages,
    prompt_metadata,
    prompt_sha256,
)
from llm.generated_code_verifier import (
    extract_restricted_assignment,
    to_dict as verification_to_dict,
    verify_and_execute_generated_code,
)
from llm.env import load_local_env
from llm.model_clients import (ModelCallResult, call_gemini_cloud_model,
    get_cloud_codegen_evaluation_models, get_cloud_codegen_evaluation_runtime)
from scripts.athlete_router_evaluation_common import safe_error
from scripts.evaluate_cloud_codegen_models import TASKS, TASK_LABELS, load_evaluation_samples

OUTPUT_DIR = ROOT / "evaluation" / "results" / "prompt_design_v2"
CHECKPOINT = OUTPUT_DIR / "codegen_prompt_design_v2_checkpoint.jsonl"
PREDICTIONS = OUTPUT_DIR / "codegen_prompt_design_v2.csv"
SUMMARY = OUTPUT_DIR / "codegen_prompt_design_v2_summary.json"
FAILURES = OUTPUT_DIR / "codegen_prompt_design_v2_failures.csv"
FIXED_MODEL_KEY = "gemini"
FIXED_MODEL_DISPLAY_NAME = "Gemini 3.5 Flash"
TEMPERATURE = 0.0
MAX_TOKENS = 2048
EXPECTED_SAMPLES = 40
EXPECTED_TASK_SAMPLES = 8
ERROR_STAGES = ("API_ERROR", "EMPTY_OUTPUT", "STRUCTURE_FAIL", "REQUEST_MATCH_FAIL",
                "EXECUTION_FAIL", "RESULT_FAIL")


def _model_config(model_key: str):
    models = {model.key: model for model in get_cloud_codegen_evaluation_models()}
    if model_key not in models:
        raise ValueError(f"Unknown evaluation model: {model_key}")
    return models[model_key]


def call_v2_model(messages, *, max_tokens: int) -> ModelCallResult:
    return call_gemini_cloud_model(messages, temperature=TEMPERATURE, max_tokens=max_tokens)


def _validated_gemini_runtime() -> dict:
    """Require production Gemini model/endpoint to match the benchmark registry."""
    load_local_env()
    benchmark = get_cloud_codegen_evaluation_runtime(FIXED_MODEL_KEY)
    production = {
        "model": os.getenv("LLM_GEMINI_MODEL", "gemini-3.5-flash").strip(),
        "provider": os.getenv("LLM_GEMINI_PROVIDER", "openai_compatible").strip(),
        "base_url": os.getenv("LLM_GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/").strip(),
    }
    if production["model"] != benchmark["model"] or production["base_url"] != benchmark["base_url"]:
        raise RuntimeError("Production Gemini model/endpoint differs from the cloud benchmark configuration.")
    return production


def _verification_without_known_statsmodels_noise(*args, **kwargs):
    """Suppress only the known degenerate-OLS display warning, not any failure."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"divide by zero encountered in scalar divide",
                                category=RuntimeWarning, module=r"statsmodels\..*")
        return verify_and_execute_generated_code(*args, **kwargs)


def evaluate_pair(sample: dict, prompt_version: str, *, caller=call_v2_model) -> dict:
    config = _model_config(FIXED_MODEL_KEY)
    messages = build_codegen_prompt_design_v2_messages(prompt_version, sample["prompt"])
    call = caller(messages, max_tokens=MAX_TOKENS)
    raw = call.content or ""
    generated_call, candidate_count = extract_restricted_assignment(raw)
    verification = _verification_without_known_statsmodels_noise(
        generated_call if raw else None,
        user_request=sample["prompt"],
        requested_analysis=sample["requested_analysis"],
        requested_filters=sample.get("analysis_filters") or {},
    )
    verified = verification_to_dict(verification)
    api_success = bool(call.success and not call.fallback_used)
    scored = bool(api_success and raw)
    if not call.success or call.fallback_used:
        error_stage = "API_ERROR"
        error_reason = "Model fallback is forbidden." if call.fallback_used else safe_error(call.error)
    elif not raw:
        error_stage = "EMPTY_OUTPUT"; error_reason = "The successful API call returned no model output."
    elif not verified.get("structure_validation_passed"):
        error_stage = "STRUCTURE_FAIL"; error_reason = verified.get("validation_error")
    elif not verified.get("request_match_passed"):
        error_stage = "REQUEST_MATCH_FAIL"; error_reason = verified.get("validation_error")
    elif not verified.get("local_execution_passed"):
        error_stage = "EXECUTION_FAIL"; error_reason = verified.get("validation_error")
    elif not verified.get("result_validation_passed"):
        error_stage = "RESULT_FAIL"; error_reason = verified.get("validation_error")
    else:
        error_stage = None; error_reason = None
    return {
        "sample_id": sample["id"], "task": sample["requested_analysis"],
        "prompt_version": prompt_version, "prompt_sha256": prompt_sha256(prompt_version),
        "user_request": sample["prompt"], "raw_model_output": raw,
        "generated_call": generated_call if raw else None,
        "candidate_assignment_count": candidate_count,
        "generated_method": verified.get("generated_method"),
        "generated_arguments": verified.get("generated_arguments"),
        "expected_method": verified.get("expected_method"),
        "expected_arguments": verified.get("expected_arguments"),
        "structure_valid": bool(verified.get("structure_validation_passed")),
        "request_match": bool(verified.get("request_match_passed")),
        "execute_valid": bool(verified.get("local_execution_passed")),
        "result_valid": bool(verified.get("result_validation_passed")),
        "fully_correct": bool(verified.get("fully_correct")) if scored else None,
        "request_mismatches": verified.get("request_mismatches") or [],
        "error_stage": error_stage, "error_reason": error_reason,
        "validator_failure_stage": verified.get("failure_stage"),
        "api_call_success": api_success, "evaluation_scored": scored,
        "api_error": None if api_success else safe_error(call.error),
        "latency": call.latency_seconds, "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens, "finish_reason": call.finish_reason,
        "model_key": FIXED_MODEL_KEY, "model_display_name": config.display_name,
        "requested_model": call.requested_model, "actual_model": call.actual_model,
        "provider": call.provider, "generation_attempts": 1, "auto_correction_used": False,
    }


def _read_checkpoint() -> list[dict]:
    if not CHECKPOINT.exists(): return []
    return [json.loads(line) for line in CHECKPOINT.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_checkpoint(row: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _clear_v2_outputs() -> None:
    """Clear only files inside the dedicated V2 result namespace."""
    if not OUTPUT_DIR.exists(): return
    expected = OUTPUT_DIR.resolve()
    if expected != (ROOT / "evaluation" / "results" / "prompt_design_v2").resolve():
        raise RuntimeError("Refusing to clear an unexpected output directory.")
    for path in OUTPUT_DIR.iterdir():
        if path.is_dir():
            raise RuntimeError(f"Unexpected subdirectory in V2 output namespace: {path}")
        path.unlink()


def _latest(rows: list[dict]) -> list[dict]:
    latest = {}
    for row in rows:
        latest[(row["model_key"], row["prompt_version"], row["sample_id"])] = row
    return list(latest.values())


def _fixed_rate(rows: list[dict], field: str, expected: int) -> float | None:
    scored = [row for row in rows if row.get("evaluation_scored")]
    return sum(bool(row.get(field)) for row in scored) / expected if len(scored) == expected else None


def summarize_prompt(rows: list[dict], expected_ids: set[str]) -> dict:
    attempted_ids = {row["sample_id"] for row in rows}
    completed = [row for row in rows if row.get("api_call_success")]
    scored = [row for row in rows if row.get("evaluation_scored")]
    scored_ids = {row["sample_id"] for row in scored}
    complete = attempted_ids == expected_ids and len(completed) == len(scored) == EXPECTED_SAMPLES
    return {
        "expected_samples": EXPECTED_SAMPLES, "attempted_samples": len(attempted_ids),
        "completed_model_calls": len(completed),
        "api_failures": sum(not row.get("api_call_success") for row in rows),
        "empty_outputs": sum(row.get("error_stage") == "EMPTY_OUTPUT" for row in rows),
        "scored_samples": len(scored), "complete": complete,
        "missing_sample_ids": sorted(expected_ids - attempted_ids),
        "unscored_sample_ids": sorted(expected_ids - scored_ids),
        "structure_validation": _fixed_rate(rows, "structure_valid", EXPECTED_SAMPLES),
        "request_match": _fixed_rate(rows, "request_match", EXPECTED_SAMPLES),
        "execution": _fixed_rate(rows, "execute_valid", EXPECTED_SAMPLES),
        "result_valid": _fixed_rate(rows, "result_valid", EXPECTED_SAMPLES),
        "fully_correct": _fixed_rate(rows, "fully_correct", EXPECTED_SAMPLES),
        "fully_correct_by_task": {task: _fixed_rate(
            [row for row in rows if row["task"] == task], "fully_correct", EXPECTED_TASK_SAMPLES)
            for task in TASKS},
        "error_stage_counts": dict(Counter(row.get("error_stage") or "SUCCESS" for row in rows)),
    }


_CSV_COLUMNS = [
    "sample_id", "task", "prompt_version", "prompt_sha256", "user_request", "raw_model_output",
    "generated_call", "generated_method", "generated_arguments", "expected_method", "expected_arguments",
    "structure_valid", "request_match", "execute_valid", "result_valid", "fully_correct",
    "request_mismatches", "error_stage", "error_reason", "validator_failure_stage", "latency",
    "api_call_success", "evaluation_scored", "api_error", "model_key", "model_display_name",
    "requested_model", "actual_model", "provider", "generation_attempts", "auto_correction_used",
]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS, extrasaction="ignore"); writer.writeheader()
        for row in rows:
            item = dict(row)
            for field in ("generated_arguments", "expected_arguments", "request_mismatches"):
                item[field] = json.dumps(item.get(field) or ([] if field == "request_mismatches" else {}), ensure_ascii=False)
            writer.writerow(item)


def _percent(value) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def print_summary(summary: dict) -> None:
    print(f"{'Prompt':18} {'Samples':>8} {'Structure':>10} {'Request':>10} {'Execute':>10} {'Result':>10} {'Fully Correct':>14}")
    for version in PROMPT_VERSIONS:
        row = summary["prompts"][version]
        print(f"{PROMPT_DISPLAY_NAMES[version]:18} {row['scored_samples']:>8} {_percent(row['structure_validation']):>10} "
              f"{_percent(row['request_match']):>10} {_percent(row['execution']):>10} "
              f"{_percent(row['result_valid']):>10} {_percent(row['fully_correct']):>14}")
    print("\nFully Correct Accuracy by Task")
    print(f"{'Task':24} " + " ".join(f"{PROMPT_DISPLAY_NAMES[v]:>16}" for v in PROMPT_VERSIONS))
    for task in TASKS:
        print(f"{TASK_LABELS[task]:24} " + " ".join(
            f"{_percent(summary['prompts'][v]['fully_correct_by_task'][task]):>16}" for v in PROMPT_VERSIONS))
    print("\nEvaluation Integrity Check")
    print(f"{'Prompt':18} {'Expected':>9} {'Attempted':>10} {'Completed':>10} {'API Failures':>13} {'Scored':>8}")
    for version in PROMPT_VERSIONS:
        row = summary["prompts"][version]
        print(f"{PROMPT_DISPLAY_NAMES[version]:18} {row['expected_samples']:>9} {row['attempted_samples']:>10} "
              f"{row['completed_model_calls']:>10} {row['api_failures']:>13} {row['scored_samples']:>8}")
        if row["unscored_sample_ids"]:
            print("  Unscored sample IDs: " + ", ".join(row["unscored_sample_ids"]))
    if not summary["evaluation_complete"]:
        print("\nWARNING: EVALUATION INCOMPLETE")
        print("Use --resume to retry missing/API-failed calls.")


def inspect_results(task_name: str) -> int:
    if not PREDICTIONS.exists():
        raise FileNotFoundError(f"V2 predictions not found: {PREDICTIONS}")
    task = "variance_analysis" if task_name == "variance" else task_name
    with PREDICTIONS.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("task") == task]
    if not rows:
        print(f"No saved {task} rows found."); return 1
    for row in rows:
        print("=" * 62)
        print(f"Sample ID: {row['sample_id']}\nPrompt: {row['prompt_version']}\nUser Request: {row['user_request']}")
        print(f"Expected requirements: method={row.get('expected_method')}; arguments={row.get('expected_arguments')}")
        print(f"Generated call: {row.get('generated_call') or '<none>'}")
        print(f"Structure result: {row.get('structure_valid')}\nRequest Match result: {row.get('request_match')}")
        print(f"Execution result: {row.get('execute_valid')}\nResult result: {row.get('result_valid')}")
        print(f"Exact validation error: {row.get('error_reason') or '<none>'}")
    return 0


def run(*, fresh: bool, resume: bool, limit: int | None = None) -> dict:
    metadata, samples = load_evaluation_samples()
    expected_ids = {sample["id"] for sample in samples}
    if len(samples) != EXPECTED_SAMPLES or len(expected_ids) != EXPECTED_SAMPLES:
        raise ValueError("V2 requires exactly 40 unique benchmark samples.")
    if limit is not None: samples = samples[:limit]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if fresh:
        _clear_v2_outputs()
    if CHECKPOINT.exists() and not (fresh or resume):
        raise RuntimeError("A V2 checkpoint exists. Use --resume or --fresh.")
    prior = _read_checkpoint() if resume else []
    if any(row.get("model_key") != FIXED_MODEL_KEY for row in prior):
        raise RuntimeError("V2 checkpoint model differs from the selected model; use --fresh.")
    hashes = {version: prompt_sha256(version) for version in PROMPT_VERSIONS}
    if any(row.get("prompt_sha256") != hashes.get(row.get("prompt_version")) for row in prior):
        raise RuntimeError("V2 checkpoint prompt hash differs from current prompts; use --fresh.")
    completed = {(row["model_key"], row["prompt_version"], row["sample_id"])
                 for row in prior if row.get("evaluation_scored")}
    rows = list(prior)
    config = _model_config(FIXED_MODEL_KEY)
    runtime = _validated_gemini_runtime()
    if config.display_name != FIXED_MODEL_DISPLAY_NAME:
        raise RuntimeError("The fixed Gemini evaluation registry display name changed unexpectedly.")
    print("=" * 62); print("Restricted Code Generation Prompt Design Evaluation V2"); print("=" * 62)
    print(f"Model: {FIXED_MODEL_DISPLAY_NAME}\nBenchmark samples: 40")
    print("Prompt variants: Basic Interface, Defined, Full")
    print(f"Runtime model ID: {runtime['model']}\nProvider: {runtime['provider']}")
    print("First-pass generations only: YES\nAuto-correction: NO")
    for version in PROMPT_VERSIONS:
        print(f"Active prompt: {PROMPT_DISPLAY_NAMES[version]}")
        for sample in samples:
            key = (FIXED_MODEL_KEY, version, sample["id"])
            if key in completed: continue
            row = evaluate_pair(sample, version); _append_checkpoint(row); rows.append(row)
    rows = _latest(rows)
    _write_csv(PREDICTIONS, rows)
    _write_csv(FAILURES, [row for row in rows if row.get("error_stage")])
    prompt_summaries = {version: summarize_prompt(
        [row for row in rows if row["prompt_version"] == version], expected_ids)
        for version in PROMPT_VERSIONS}
    summary = {
        "evaluation_type": "restricted_codegen_prompt_design_v2_first_pass",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": metadata["dataset_path"], "dataset_sha256": metadata["dataset_sha256"],
        "benchmark_samples_per_prompt": EXPECTED_SAMPLES, "total_expected_generations": 120,
        "sample_ids": sorted(expected_ids), "model_key": FIXED_MODEL_KEY,
        "model_display_name": FIXED_MODEL_DISPLAY_NAME, "model": runtime["model"],
        "provider": runtime["provider"], "endpoint": runtime["base_url"],
        "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
        "model_calls_per_sample": 1, "auto_correction_used": False,
        "known_statsmodels_warning_suppressed": "divide by zero encountered in scalar divide",
        "known_warning_affects_scoring": False, "prompt_metadata": prompt_metadata(),
        "prompts": prompt_summaries,
        "evaluation_complete": all(value["complete"] for value in prompt_summaries.values()),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(summary); return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    lifecycle = parser.add_mutually_exclusive_group()
    lifecycle.add_argument("--fresh", action="store_true"); lifecycle.add_argument("--resume", action="store_true")
    parser.add_argument("--inspect", choices=("correlation", "variance"))
    parser.add_argument("--limit", type=int, help="Development-only sample limit.")
    args = parser.parse_args(argv)
    if args.inspect: return inspect_results(args.inspect)
    run(fresh=args.fresh, resume=args.resume, limit=args.limit); return 0


if __name__ == "__main__":
    raise SystemExit(main())
