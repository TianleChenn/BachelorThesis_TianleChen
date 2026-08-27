"""Local Ministral-3-8B version of Restricted Code Generation Prompt Design V2."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.code_generation_prompt import build_code_generation_messages
from llm.code_generation_prompt_design_v2 import (
    PROMPT_VERSIONS,
    build_codegen_prompt_design_v2_messages,
    prompt_metadata,
    prompt_sha256,
)
from llm.generated_code_verifier import extract_restricted_assignment, to_dict as verification_to_dict
from llm.model_clients import ModelCallResult, call_local_codegen_model
from scripts.athlete_router_evaluation_common import safe_error
from scripts.evaluate_codegen_prompt_design_v2 import (
    ERROR_STAGES,
    EXPECTED_SAMPLES,
    EXPECTED_TASK_SAMPLES,
    TASKS,
    TASK_LABELS,
    _verification_without_known_statsmodels_noise,
    summarize_prompt,
)
from scripts.evaluate_local_codegen_models import (
    LocalEvaluationModel,
    find_llama_server,
    load_evaluation_samples,
    model_map,
    port_is_available,
    running_model_server,
    warm_up_model,
)

MODEL_KEY = "ministral"
MODEL_DISPLAY_NAME = "Ministral-3-8B"
MODEL_HF_SPEC = "mistralai/Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M"
MODEL_ALIAS = "ministral-3-8b-local-eval"
QUANTIZATION = "Q4_K_M"
PROVIDER = "Local / llama.cpp"
TEMPERATURE = 0.0
MAX_TOKENS = 2048
EXPECTED_GENERATIONS = 120
PROMPT_DISPLAY_NAMES = {
    "basic_interface": "Simple",
    "defined": "Medium",
    "full": "Restricted Code Generation Prompt",
}

OUTPUT_DIR = ROOT / "evaluation" / "results" / "local_prompt_design_v2"
CHECKPOINT = OUTPUT_DIR / "local_codegen_prompt_design_v2_checkpoint.jsonl"
PREDICTIONS = OUTPUT_DIR / "local_codegen_prompt_design_v2.csv"
SUMMARY = OUTPUT_DIR / "local_codegen_prompt_design_v2_summary.json"
FAILURES = OUTPUT_DIR / "local_codegen_prompt_design_v2_failures.csv"


def ministral_model() -> LocalEvaluationModel:
    model = model_map()[MODEL_KEY]
    if (model.display_name, model.hf_spec, model.alias) != (
        MODEL_DISPLAY_NAME, MODEL_HF_SPEC, MODEL_ALIAS
    ):
        raise RuntimeError("The Ministral evaluation registry changed unexpectedly.")
    return model


def validate_samples() -> tuple[dict, list[dict]]:
    metadata, samples = load_evaluation_samples()
    ids = [sample["id"] for sample in samples]
    distribution = Counter(sample["requested_analysis"] for sample in samples)
    if len(ids) != len(set(ids)):
        raise ValueError("Local Prompt Design V2 requires 40 unique samples.")
    if len(ids) != EXPECTED_SAMPLES or distribution != Counter({task: EXPECTED_TASK_SAMPLES for task in TASKS}):
        raise ValueError("Local Prompt Design V2 requires exactly eight samples for each of five tasks.")
    return metadata, samples


def evaluate_pair(sample: dict, prompt_version: str, *, caller=call_local_codegen_model) -> dict:
    model = ministral_model()
    messages = build_codegen_prompt_design_v2_messages(prompt_version, sample["prompt"])
    call: ModelCallResult = caller(messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    alias_valid = (call.requested_model == model.alias and
                   (call.actual_model == model.alias if call.success else True))
    raw = call.content or ""
    generated_call, candidate_count = extract_restricted_assignment(raw)
    verified = verification_to_dict(_verification_without_known_statsmodels_noise(
        generated_call if raw else None,
        user_request=sample["prompt"], requested_analysis=sample["requested_analysis"],
        requested_filters=sample.get("analysis_filters") or {}, close_figures_after_execution=True,
    ))
    api_success = bool(call.success and not call.fallback_used and alias_valid)
    scored = bool(api_success and raw)
    if not alias_valid:
        error_stage, error_reason = "API_ERROR", "Local response model alias did not match Ministral."
    elif not call.success or call.fallback_used:
        error_stage, error_reason = "API_ERROR", safe_error(call.error)
    elif not raw:
        error_stage, error_reason = "EMPTY_OUTPUT", "The Local model returned no output."
    elif not verified.get("structure_validation_passed"):
        error_stage, error_reason = "STRUCTURE_FAIL", verified.get("validation_error")
    elif not verified.get("request_match_passed"):
        error_stage, error_reason = "REQUEST_MATCH_FAIL", verified.get("validation_error")
    elif not verified.get("local_execution_passed"):
        error_stage, error_reason = "EXECUTION_FAIL", verified.get("validation_error")
    elif not verified.get("result_validation_passed"):
        error_stage, error_reason = "RESULT_FAIL", verified.get("validation_error")
    else:
        error_stage, error_reason = None, None
    return {
        "sample_id": sample["id"], "task": sample["requested_analysis"],
        "prompt_version": prompt_version, "prompt_sha256": prompt_sha256(prompt_version),
        "dataset_sha256": sample["_dataset_sha256"], "user_request": sample["prompt"],
        "raw_model_output": raw, "generated_call": generated_call if raw else None,
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
        "api_error": None if api_success else error_reason,
        "latency": call.latency_seconds, "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens, "finish_reason": call.finish_reason,
        "model_key": model.key, "model_display_name": model.display_name,
        "model_hf_spec": model.hf_spec, "model_alias": model.alias,
        "requested_model": call.requested_model, "actual_model": call.actual_model,
        "provider": PROVIDER, "quantization": QUANTIZATION,
        "generation_attempts": 1, "auto_correction_used": False,
    }


def _read_checkpoint() -> list[dict]:
    if not CHECKPOINT.exists(): return []
    return [json.loads(line) for line in CHECKPOINT.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_checkpoint(row: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _latest(rows: list[dict]) -> list[dict]:
    latest = {}
    for row in rows: latest[(row["model_key"], row["prompt_version"], row["sample_id"])] = row
    return list(latest.values())


def _clear_outputs() -> None:
    if not OUTPUT_DIR.exists(): return
    expected = (ROOT / "evaluation" / "results" / "local_prompt_design_v2").resolve()
    if OUTPUT_DIR.resolve() != expected: raise RuntimeError("Refusing to clear an unexpected directory.")
    for path in OUTPUT_DIR.iterdir():
        if path.is_dir(): raise RuntimeError(f"Unexpected Local V2 subdirectory: {path}")
        path.unlink()


def _validate_checkpoint(rows: list[dict], dataset_sha256: str) -> None:
    model = ministral_model(); hashes = {version: prompt_sha256(version) for version in PROMPT_VERSIONS}
    for row in rows:
        if row.get("model_key") != model.key or row.get("model_hf_spec") != model.hf_spec or row.get("model_alias") != model.alias:
            raise RuntimeError("Checkpoint Local model specification changed; use --fresh.")
        if row.get("dataset_sha256") != dataset_sha256:
            raise RuntimeError("Checkpoint dataset SHA256 changed; use --fresh.")
        if row.get("prompt_sha256") != hashes.get(row.get("prompt_version")):
            raise RuntimeError("Checkpoint prompt SHA256 changed; use --fresh.")


_CSV_COLUMNS = ["sample_id", "task", "prompt_version", "prompt_sha256", "dataset_sha256",
    "user_request", "raw_model_output", "generated_call", "generated_method", "generated_arguments",
    "expected_method", "expected_arguments", "structure_valid", "request_match", "execute_valid",
    "result_valid", "fully_correct", "request_mismatches", "error_stage", "error_reason",
    "validator_failure_stage", "latency", "api_call_success", "evaluation_scored", "api_error",
    "model_key", "model_display_name", "model_hf_spec", "model_alias", "requested_model",
    "actual_model", "provider", "quantization", "generation_attempts", "auto_correction_used"]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS, extrasaction="ignore"); writer.writeheader()
        for row in rows:
            item = dict(row)
            for field in ("generated_arguments", "expected_arguments", "request_mismatches"):
                item[field] = json.dumps(item.get(field) or ([] if field == "request_mismatches" else {}), ensure_ascii=False)
            writer.writerow(item)


def _percent(value) -> str: return "N/A" if value is None else f"{value * 100:.1f}%"


def _summary(metadata: dict, samples: list[dict], rows: list[dict]) -> dict:
    expected_ids = {sample["id"] for sample in samples}
    prompts = {version: summarize_prompt([row for row in rows if row["prompt_version"] == version], expected_ids)
               for version in PROMPT_VERSIONS}
    prompts_metadata=prompt_metadata()
    for version in PROMPT_VERSIONS:prompts_metadata[version]["display_name"]=PROMPT_DISPLAY_NAMES[version]
    return {"evaluation_type":"local_restricted_codegen_prompt_design_v2_first_pass",
        "created_at":datetime.now(timezone.utc).isoformat(), "dataset":metadata["dataset_path"],
        "dataset_sha256":metadata["dataset_sha256"], "sample_ids":sorted(expected_ids),
        "benchmark_samples_per_prompt":EXPECTED_SAMPLES, "total_expected_generations":EXPECTED_GENERATIONS,
        "model_key":MODEL_KEY, "model_display_name":MODEL_DISPLAY_NAME, "model_hf_spec":MODEL_HF_SPEC,
        "model_alias":MODEL_ALIAS, "provider":PROVIDER, "quantization":QUANTIZATION,
        "temperature":TEMPERATURE, "max_tokens":MAX_TOKENS, "model_calls_per_sample":1,
        "auto_correction_used":False, "prompt_metadata":prompts_metadata, "prompts":prompts,
        "evaluation_complete":all(value["complete"] for value in prompts.values())}


def _print_header() -> None:
    print("="*62); print("Local Restricted Code Generation Prompt Design Evaluation"); print("="*62)
    print(f"Model: {MODEL_DISPLAY_NAME}\nProvider: {PROVIDER}\nModel specification:\n{MODEL_HF_SPEC}")
    print("\nBenchmark samples: 40\n\nPrompt variants:")
    for version in PROMPT_VERSIONS: print(f"- {PROMPT_DISPLAY_NAMES[version]}")
    print(f"\nFirst-pass generations only: YES\nAuto-correction: NO\nTemperature: {TEMPERATURE}")
    print(f"Maximum generation tokens: {MAX_TOKENS}\nExpected generations: {EXPECTED_GENERATIONS}")


def _print_summary(summary: dict) -> None:
    print(f"\n{'Prompt':38} {'Samples':>7} {'Structure':>9} {'Request':>8} {'Execute':>8} {'Result':>8} {'Fully Correct':>13}")
    for version in PROMPT_VERSIONS:
        row=summary["prompts"][version]
        print(f"{PROMPT_DISPLAY_NAMES[version]:38} {row['scored_samples']:>7} {_percent(row['structure_validation']):>9} "
              f"{_percent(row['request_match']):>8} {_percent(row['execution']):>8} "
              f"{_percent(row['result_valid']):>8} {_percent(row['fully_correct']):>13}")
    print("\nFully Correct Accuracy by Task")
    print(f"{'Task':22} " + " ".join(f"{PROMPT_DISPLAY_NAMES[v]:>34}" for v in PROMPT_VERSIONS))
    for task in TASKS:
        print(f"{TASK_LABELS[task]:22} " + " ".join(f"{_percent(summary['prompts'][v]['fully_correct_by_task'][task]):>34}" for v in PROMPT_VERSIONS))
    print("\nEvaluation Integrity Check")
    print(f"{'Prompt':38} {'Expected':>8} {'Attempted':>9} {'Completed':>9} {'API Failures':>12} {'Scored':>7}")
    for version in PROMPT_VERSIONS:
        row=summary["prompts"][version]
        print(f"{PROMPT_DISPLAY_NAMES[version]:38} {row['expected_samples']:>8} {row['attempted_samples']:>9} "
              f"{row['completed_model_calls']:>9} {row['api_failures']:>12} {row['scored_samples']:>7}")
        if row["unscored_sample_ids"]: print("  Unscored: " + ", ".join(row["unscored_sample_ids"]))
    if not summary["evaluation_complete"]: print("\nWARNING: EVALUATION INCOMPLETE")


def check_only(*, port: int) -> int:
    metadata, samples = validate_samples(); model=ministral_model(); executable=find_llama_server()
    available=port_is_available(port); OUTPUT_DIR.mkdir(parents=True,exist_ok=True); writable=os.access(OUTPUT_DIR,os.W_OK)
    print(f"Dataset: {metadata['dataset_path']} ({len(samples)} samples)\nllama-server: {executable or 'NOT FOUND'}")
    print(f"Evaluation port {port}: {'available' if available else 'already in use'}")
    print(f"Output directory: {'writable' if writable else 'not writable'}")
    print(f"Model: {model.display_name}\nHF: {model.hf_spec}\nAlias: {model.alias}")
    return int(not(executable and available and writable))


def smoke(*, port: int, timeout_seconds: int) -> int:
    metadata,samples=validate_samples();sample={**samples[0],"_dataset_sha256":metadata["dataset_sha256"]};model=ministral_model()
    _print_header();failed=False
    with running_model_server(model,port=port,timeout_seconds=timeout_seconds,log_dir=OUTPUT_DIR):
        warm_up_model()
        for version in PROMPT_VERSIONS:
            row=evaluate_pair(sample,version);failed |= not row["api_call_success"]
            print(f"\n{PROMPT_DISPLAY_NAMES[version]}\nRaw output: {row['raw_model_output']}\nGenerated call: {row['generated_call']}")
            for label,field in (("Structure","structure_valid"),("Request","request_match"),("Execute","execute_valid"),("Result","result_valid"),("Fully Correct","fully_correct")): print(f"{label}: {row[field]}")
            print(f"Failure stage: {row['error_stage']}\nValidation error: {row['error_reason']}")
    return int(failed)


def run(*, port: int, timeout_seconds: int, fresh: bool, resume: bool) -> dict:
    metadata,samples=validate_samples();model=ministral_model();OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    if fresh:_clear_outputs()
    if CHECKPOINT.exists() and not(fresh or resume):raise RuntimeError("Local V2 checkpoint exists. Use --resume or --fresh.")
    prior=_read_checkpoint() if resume else [];_validate_checkpoint(prior,metadata["dataset_sha256"])
    completed={(row["model_key"],row["prompt_version"],row["sample_id"]) for row in prior if row.get("evaluation_scored")};rows=list(prior)
    _print_header();pending=any((MODEL_KEY,v,s["id"]) not in completed for v in PROMPT_VERSIONS for s in samples)
    if pending:
        with running_model_server(model,port=port,timeout_seconds=timeout_seconds,log_dir=OUTPUT_DIR):
            warm_up_model()
            for version in PROMPT_VERSIONS:
                print(f"Active prompt: {PROMPT_DISPLAY_NAMES[version]}")
                for sample in samples:
                    key=(MODEL_KEY,version,sample["id"])
                    if key in completed:continue
                    row=evaluate_pair({**sample,"_dataset_sha256":metadata["dataset_sha256"]},version)
                    _append_checkpoint(row);rows.append(row)
    rows=_latest(rows);_write_csv(PREDICTIONS,rows);_write_csv(FAILURES,[row for row in rows if row.get("error_stage")])
    summary=_summary(metadata,samples,rows);SUMMARY.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    _print_summary(summary);return summary


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only",action="store_true");mode.add_argument("--smoke",action="store_true")
    lifecycle=parser.add_mutually_exclusive_group();lifecycle.add_argument("--fresh",action="store_true");lifecycle.add_argument("--resume",action="store_true")
    parser.add_argument("--port",type=int,default=8081);parser.add_argument("--server-timeout-seconds",type=int,default=1800)
    args=parser.parse_args(argv)
    if not 1<=args.port<=65535:parser.error("--port must be between 1 and 65535")
    if args.server_timeout_seconds<=0:parser.error("--server-timeout-seconds must be positive")
    return args


def main(argv=None)->int:
    args=parse_args(argv)
    if args.check_only:return check_only(port=args.port)
    if args.smoke:return smoke(port=args.port,timeout_seconds=args.server_timeout_seconds)
    run(port=args.port,timeout_seconds=args.server_timeout_seconds,fresh=args.fresh,resume=args.resume);return 0


if __name__=="__main__":raise SystemExit(main())
