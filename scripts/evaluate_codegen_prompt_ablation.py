"""First-pass prompt ablation for restricted analysis-call generation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.code_generation_prompt_ablation import (
    PROMPT_DISPLAY_NAMES,
    PROMPT_VERSIONS,
    build_codegen_ablation_messages,
    prompt_metadata,
    prompt_sha256,
)
from llm.generated_code_verifier import (
    extract_restricted_assignment,
    to_dict as verification_to_dict,
    verify_and_execute_generated_code,
)
from llm.model_clients import (
    ModelCallResult,
    _call,
    get_cloud_codegen_evaluation_models,
    get_cloud_codegen_evaluation_runtime,
)
from scripts.evaluate_cloud_codegen_models import TASKS, TASK_LABELS, load_evaluation_samples
from scripts.athlete_router_evaluation_common import safe_error
from llm.analysis_request_contracts import build_request_contract, render_request_contract

OUTPUT_DIR = ROOT / "evaluation" / "results" / "prompt_ablation"
CHECKPOINT = OUTPUT_DIR / "codegen_prompt_ablation_checkpoint.jsonl"
PREDICTIONS = OUTPUT_DIR / "codegen_prompt_ablation.csv"
SUMMARY_JSON = OUTPUT_DIR / "codegen_prompt_ablation_summary.json"
FIGURE_PNG = OUTPUT_DIR / "codegen_prompt_ablation.png"
FIGURE_PDF = OUTPUT_DIR / "codegen_prompt_ablation.pdf"
DEFAULT_MODEL_KEY = "gpt4_1"
MAX_TOKENS = 2048
TEMPERATURE = 0.0
EXPECTED_SAMPLES = 40
EXPECTED_TASK_SAMPLES = 8
ERROR_STAGES = {
    "api": "API_ERROR", "empty": "EMPTY_OUTPUT", "structure": "STRUCTURE_FAIL",
    "request": "REQUEST_MATCH_FAIL", "execution": "EXECUTION_FAIL", "result": "RESULT_FAIL",
}


def _model_config(model_key: str):
    models = {model.key: model for model in get_cloud_codegen_evaluation_models()}
    if model_key not in models:
        raise ValueError(f"Unknown code-generation model: {model_key}")
    return models[model_key]


def call_codegen_ablation_model(model_key: str, messages, *, max_tokens: int) -> ModelCallResult:
    """Use one explicit deterministic configuration for every prompt version."""
    runtime = get_cloud_codegen_evaluation_runtime(model_key)
    return _call(messages, runtime["model"], runtime["provider"], runtime["api_key"],
                 runtime["base_url"], TEMPERATURE, max_tokens)


def evaluate_pair(
    sample: dict,
    prompt_version: str,
    model_key: str,
    *,
    caller=call_codegen_ablation_model,
) -> dict:
    """Make exactly one model call, then reuse the existing five-stage verifier."""
    config = _model_config(model_key)
    messages = build_codegen_ablation_messages(prompt_version, sample["prompt"])
    call: ModelCallResult = caller(model_key, messages, max_tokens=MAX_TOKENS)
    raw = call.content or ""
    cleaned, candidate_count = extract_restricted_assignment(raw)
    verified = verification_to_dict(verify_and_execute_generated_code(
        cleaned if raw else None,
        user_request=sample["prompt"],
        requested_analysis=sample["requested_analysis"],
        requested_filters=sample.get("analysis_filters") or {},
    ))
    api_success = bool(call.success and not call.fallback_used)
    scored = bool(api_success and call.content)
    if not call.success or call.fallback_used:
        error_stage = ERROR_STAGES["api"]
        error_reason = "Model fallback is forbidden." if call.fallback_used else safe_error(call.error)
    elif not call.content:
        error_stage = ERROR_STAGES["empty"]
        error_reason = "The model call succeeded but returned an empty output."
    elif not verified.get("structure_validation_passed"):
        error_stage = ERROR_STAGES["structure"]
        error_reason = verified.get("validation_error")
    elif not verified.get("request_match_passed"):
        error_stage = ERROR_STAGES["request"]
        error_reason = verified.get("validation_error")
    elif not verified.get("local_execution_passed"):
        error_stage = ERROR_STAGES["execution"]
        error_reason = verified.get("validation_error")
    elif not verified.get("result_validation_passed"):
        error_stage = ERROR_STAGES["result"]
        error_reason = verified.get("validation_error")
    else:
        error_stage = None
        error_reason = None
    return {
        "sample_id": sample["id"], "task": sample["requested_analysis"],
        "prompt_version": prompt_version, "prompt_sha256": prompt_sha256(prompt_version),
        "user_request": sample["prompt"], "model_key": model_key,
        "model_display_name": config.display_name, "provider": config.provider,
        "requested_model": call.requested_model, "actual_model": call.actual_model,
        "raw_model_output": raw, "generated_call": cleaned if raw else None,
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
        "error_stage": error_stage, "error_reason": error_reason,
        "validator_failure_stage": verified.get("failure_stage"),
        "request_mismatches": verified.get("request_mismatches") or [],
        "api_call_success": api_success, "evaluation_scored": scored,
        "api_error": None if api_success else safe_error(call.error),
        "latency": call.latency_seconds, "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens, "finish_reason": call.finish_reason,
        "generation_attempts": 1, "correction_used": False,
    }


def _read_checkpoint() -> list[dict]:
    if not CHECKPOINT.exists(): return []
    return [json.loads(line) for line in CHECKPOINT.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append(row: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _latest(rows: list[dict]) -> list[dict]:
    latest = {}
    for row in rows:
        latest[(row["model_key"], row["prompt_version"], row["sample_id"])] = row
    return list(latest.values())


def _fixed_rate(rows: list[dict], field: str, expected: int):
    scored = [row for row in rows if row.get("evaluation_scored")]
    return sum(bool(row.get(field)) for row in scored) / expected if len(scored) == expected else None


def _summary_for(rows: list[dict], expected_ids: set[str]) -> dict:
    attempted_ids = {row["sample_id"] for row in rows}
    completed = [row for row in rows if row.get("api_call_success")]
    scored = [row for row in rows if row.get("evaluation_scored")]
    missing_ids = sorted(expected_ids - attempted_ids)
    unscored_ids = sorted(expected_ids - {row["sample_id"] for row in scored})
    complete = (attempted_ids == expected_ids and len(completed) == EXPECTED_SAMPLES
                and len(scored) == EXPECTED_SAMPLES)
    return {
        "expected_samples": EXPECTED_SAMPLES, "attempted_samples": len(attempted_ids),
        "completed_model_calls": len(completed),
        "api_failures": sum(not row.get("api_call_success") for row in rows),
        "empty_outputs": sum(row.get("error_stage") == ERROR_STAGES["empty"] for row in rows),
        "scored_samples": len(scored), "missing_sample_ids": missing_ids,
        "unscored_sample_ids": unscored_ids, "complete": complete,
        "structure_validation": _fixed_rate(rows, "structure_valid", EXPECTED_SAMPLES),
        "request_match": _fixed_rate(rows, "request_match", EXPECTED_SAMPLES),
        "execution": _fixed_rate(rows, "execute_valid", EXPECTED_SAMPLES),
        "result_valid": _fixed_rate(rows, "result_valid", EXPECTED_SAMPLES),
        "fully_correct": _fixed_rate(rows, "fully_correct", EXPECTED_SAMPLES),
        "fully_correct_by_task": {task: _fixed_rate(
            [row for row in rows if row["task"] == task], "fully_correct", EXPECTED_TASK_SAMPLES)
            for task in TASKS},
        "failure_stages": dict(Counter(row.get("error_stage") or "SUCCESS" for row in rows)),
    }


def _write_csv(rows: list[dict]) -> None:
    columns = ["sample_id", "task", "prompt_version", "prompt_sha256", "user_request",
               "generated_call", "generated_method", "generated_arguments", "expected_method",
               "expected_arguments", "structure_valid", "request_match", "execute_valid", "result_valid",
               "fully_correct", "error_stage", "error_reason", "validator_failure_stage",
               "request_mismatches", "latency", "api_call_success", "evaluation_scored", "api_error",
               "model_key", "model_display_name", "provider",
               "requested_model", "actual_model", "raw_model_output", "generation_attempts", "correction_used"]
    with PREDICTIONS.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore"); writer.writeheader()
        for row in rows:
            item = dict(row)
            for field in ("generated_arguments", "expected_arguments", "request_mismatches"):
                item[field] = json.dumps(item.get(field) or ([] if field == "request_mismatches" else {}), ensure_ascii=False)
            writer.writerow(item)


def _plot(summary: dict) -> None:
    import matplotlib.pyplot as plt
    labels = [PROMPT_DISPLAY_NAMES[version] for version in PROMPT_VERSIONS]
    metrics = (("structure_validation", "Structure"), ("request_match", "Request Match"),
               ("fully_correct", "Fully Correct")); width = .24; x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for index, (key, label) in enumerate(metrics):
        values = [(summary["prompts"][version][key] or 0) * 100 for version in PROMPT_VERSIONS]
        bars = ax.bar([value + (index-1)*width for value in x], values, width, label=label)
        ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=3, fontsize=8)
    ax.set_xticks(list(x), labels); ax.set_ylim(0, 108); ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Restricted Code Generation Prompt Design Ablation")
    ax.legend(); ax.grid(axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(FIGURE_PNG, dpi=300); fig.savefig(FIGURE_PDF); plt.close(fig)


def _percent(value) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def print_summary(summary: dict) -> None:
    print(f"{'Prompt':16} {'Samples':>8} {'Structure':>10} {'Request':>10} {'Execute':>10} {'Result':>10} {'Fully Correct':>14}")
    for version in PROMPT_VERSIONS:
        row = summary["prompts"][version]
        print(f"{PROMPT_DISPLAY_NAMES[version]:16} {row['scored_samples']:>8} {_percent(row['structure_validation']):>10} "
              f"{_percent(row['request_match']):>10} {_percent(row['execution']):>10} "
              f"{_percent(row['result_valid']):>10} {_percent(row['fully_correct']):>14}")
    print("\nFully Correct Accuracy by Task")
    print(f"{'Task':24} " + " ".join(f"{PROMPT_DISPLAY_NAMES[v]:>14}" for v in PROMPT_VERSIONS))
    for task in TASKS:
        print(f"{TASK_LABELS[task]:24} " + " ".join(
            f"{_percent(summary['prompts'][version]['fully_correct_by_task'][task]):>14}"
            for version in PROMPT_VERSIONS))
    print("\nEvaluation Integrity Check")
    print(f"{'Prompt':12} {'Expected':>9} {'Attempted':>10} {'Completed':>10} {'API Failures':>13} {'Scored':>8}")
    for version in PROMPT_VERSIONS:
        row = summary["prompts"][version]
        print(f"{PROMPT_DISPLAY_NAMES[version]:12} {row['expected_samples']:>9} {row['attempted_samples']:>10} "
              f"{row['completed_model_calls']:>10} {row['api_failures']:>13} {row['scored_samples']:>8}")
        if row["unscored_sample_ids"]:
            print("  Unscored sample IDs: " + ", ".join(row["unscored_sample_ids"]))
    if not summary["comparison_complete"]:
        print("\n" + "!" * 62)
        print("WARNING: PROMPT COMPARISON IS INCOMPLETE.")
        print("Use --resume to retry missing API calls.")
        print("!" * 62)


def inspect_results(task_name: str) -> int:
    if not PREDICTIONS.exists():
        raise FileNotFoundError(f"Prompt-ablation predictions not found: {PREDICTIONS}")
    requested_task = "variance_analysis" if task_name == "variance" else task_name
    _, samples = load_evaluation_samples()
    sample_map = {sample["id"]: sample for sample in samples}
    with PREDICTIONS.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("task") == requested_task]
    if not rows:
        print(f"No saved {requested_task} rows were found in {PREDICTIONS}.")
        return 1
    for row in rows:
        sample = sample_map[row["sample_id"]]
        contract = build_request_contract(sample["requested_analysis"],
            sample.get("analysis_filters") or {}, sample["prompt"])
        expected_call = render_request_contract(contract)
        if row.get("expected_method") and row.get("expected_arguments"):
            try:
                arguments = json.loads(row["expected_arguments"])
                expected_call = "result = analysis." + row["expected_method"] + "(" + ", ".join(
                    f"{name}={value!r}" for name, value in arguments.items()) + ")"
            except json.JSONDecodeError:
                pass
        print("=" * 62)
        print(f"sample_id: {row['sample_id']}\nprompt_version: {row['prompt_version']}\ntask: {row['task']}")
        print(f"user_request: {row['user_request']}")
        print(f"expected_call: {expected_call}")
        print(f"generated_call: {row.get('generated_call') or '<none>'}")
        print(f"structure_valid: {row.get('structure_valid')}\nrequest_match: {row.get('request_match')}")
        print(f"execute_valid: {row.get('execute_valid')}\nresult_valid: {row.get('result_valid')}")
        print(f"error_stage: {row.get('error_stage') or row.get('failure_stage') or '<none>'}")
        print(f"error_reason: {row.get('error_reason') or '<none>'}")
    return 0


def run(*, model_key: str, fresh: bool, resume: bool, limit: int | None = None) -> dict:
    metadata, samples = load_evaluation_samples()
    if limit is not None: samples = samples[:limit]
    config = _model_config(model_key); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if fresh:
        for path in (CHECKPOINT, PREDICTIONS, SUMMARY_JSON, FIGURE_PNG, FIGURE_PDF):
            if path.exists(): path.unlink()
    if CHECKPOINT.exists() and not (fresh or resume):
        raise RuntimeError("A prompt-ablation checkpoint exists. Use --resume or --fresh.")
    prior = _read_checkpoint() if resume else []
    if any(row.get("model_key") != model_key for row in prior):
        raise RuntimeError("Checkpoint model differs from the selected model.")
    active_hashes = {version: prompt_sha256(version) for version in PROMPT_VERSIONS}
    if any(row.get("prompt_sha256") != active_hashes.get(row.get("prompt_version")) for row in prior):
        raise RuntimeError("Checkpoint prompt hash differs from the active ablation prompt; use --fresh.")
    completed = {(row["model_key"], row["prompt_version"], row["sample_id"])
                 for row in prior if row.get("evaluation_scored")}
    rows = list(prior)
    runtime = get_cloud_codegen_evaluation_runtime(model_key)
    print("=" * 62); print("Restricted Code Generation Prompt Design Ablation"); print("=" * 62)
    print(f"Model: {runtime['model']}\nProvider: {config.provider}\nBenchmark samples: 40")
    print(f"Temperature: {TEMPERATURE}\nMax tokens: {MAX_TOKENS}")
    print("Prompt variants: Basic, Interface Guided, Full\nFirst-pass only: YES\nAuto-correction: NO")
    for version in PROMPT_VERSIONS:
        print(f"Active prompt version: {version}")
        for sample in samples:
            key = (model_key, version, sample["id"])
            if key in completed: continue
            row = evaluate_pair(sample, version, model_key); _append(row); rows.append(row)
    rows = _latest(rows); _write_csv(rows)
    expected_ids = {sample["id"] for sample in load_evaluation_samples()[1]}
    prompt_summaries = {version: _summary_for(
        [row for row in rows if row["prompt_version"] == version], expected_ids)
        for version in PROMPT_VERSIONS}
    summary = {
        "evaluation_type": "restricted_codegen_prompt_design_ablation_first_pass",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": metadata["dataset_path"], "dataset_sha256": metadata["dataset_sha256"],
        "benchmark_samples": 40, "model_key": model_key, "model": runtime["model"],
        "provider": config.provider, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
        "model_calls_per_sample": 1,
        "auto_correction_used": False, "prompt_metadata": prompt_metadata(),
        "prompts": prompt_summaries,
        "comparison_complete": all(row["complete"] for row in prompt_summaries.values()),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _plot(summary); print_summary(summary); return summary


def main(argv=None) -> int:
    choices = [model.key for model in get_cloud_codegen_evaluation_models()]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=choices, default=DEFAULT_MODEL_KEY)
    lifecycle = parser.add_mutually_exclusive_group()
    lifecycle.add_argument("--fresh", action="store_true"); lifecycle.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Development-only request limit.")
    parser.add_argument("--inspect", choices=("correlation", "variance"))
    args = parser.parse_args(argv)
    if args.inspect:
        return inspect_results(args.inspect)
    run(model_key=args.model, fresh=args.fresh, resume=args.resume, limit=args.limit); return 0


if __name__ == "__main__":
    raise SystemExit(main())
