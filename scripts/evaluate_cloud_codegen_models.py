"""Offline first-pass evaluation of three forced cloud code-generation models."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.code_generation_prompt import CODE_GENERATION_PROMPT_VERSION,build_code_generation_messages
from llm.generated_code_verifier import (
    extract_restricted_assignment,
    to_dict as verification_to_dict,
    verify_and_execute_generated_code,
)
from llm.model_clients import (
    ModelCallResult,
    call_cloud_codegen_evaluation_model,
    get_cloud_codegen_evaluation_models,
    get_cloud_codegen_evaluation_runtime,
)
from scripts.athlete_router_evaluation_common import load_frontend_benchmark, safe_error


DATASET = ROOT / "evaluation/frontend_realistic_benchmark_60.json"
PRICING = ROOT / "evaluation/cloud_codegen_pricing_snapshot_2026-08-19.json"
ARTIFACTS = ROOT / "artifacts"
CHECKPOINT = ARTIFACTS / "cloud_codegen_model_checkpoint.jsonl"
PREDICTIONS = ARTIFACTS / "cloud_codegen_model_predictions.csv"
SUMMARY = ARTIFACTS / "cloud_codegen_model_summary.csv"
PER_TASK = ARTIFACTS / "cloud_codegen_model_per_task.csv"
FAILURES = ARTIFACTS / "cloud_codegen_model_failure_breakdown.csv"
REPORT = ARTIFACTS / "cloud_codegen_model_evaluation.json"
OVERALL_FIGURE = ARTIFACTS / "cloud_codegen_model_overall_accuracy.png"
TASK_FIGURE = ARTIFACTS / "cloud_codegen_model_per_task_accuracy.png"

TASKS = ("table1", "table2", "figure1", "correlation", "variance_analysis")
TASK_LABELS = {
    "table1": "Table 1", "table2": "Table 2", "figure1": "Figure 1",
    "correlation": "Correlation", "variance_analysis": "Variance Analysis",
}
MODEL_KEYS = ("gpt4_1", "gemini", "claude")
RUN_LABEL = None

OLD_V1_ARTIFACT_NAMES = (
    "cloud_codegen_model_predictions.csv",
    "cloud_codegen_model_summary.csv",
    "cloud_codegen_model_per_task.csv",
    "cloud_codegen_model_failure_breakdown.csv",
    "cloud_codegen_model_evaluation.json",
    "cloud_codegen_model_overall_accuracy.png",
    "cloud_codegen_model_per_task_accuracy.png",
    "cloud_codegen_model_checkpoint.jsonl",
    "cloud_codegen_model_predictions_pool_only_v1.csv",
    "cloud_codegen_model_summary_pool_only_v1.csv",
    "cloud_codegen_model_per_task_pool_only_v1.csv",
    "cloud_codegen_model_failure_breakdown_pool_only_v1.csv",
    "cloud_codegen_model_evaluation_pool_only_v1.json",
    "cloud_codegen_model_overall_accuracy_pool_only_v1.png",
    "cloud_codegen_model_per_task_accuracy_pool_only_v1.png",
    "cloud_codegen_model_checkpoint_pool_only_v1.jsonl",
)


def artifact_paths(run_label: str | None = None) -> dict[str, Path]:
    if run_label is not None and not re.fullmatch(r"[A-Za-z0-9_-]+",run_label):
        raise ValueError("run-label may contain only letters, numbers, underscores, and hyphens.")
    suffix=f"_{run_label}" if run_label else ""
    return {
        "checkpoint":ARTIFACTS/f"cloud_codegen_model_checkpoint{suffix}.jsonl",
        "predictions":ARTIFACTS/f"cloud_codegen_model_predictions{suffix}.csv",
        "summary":ARTIFACTS/f"cloud_codegen_model_summary{suffix}.csv",
        "per_task":ARTIFACTS/f"cloud_codegen_model_per_task{suffix}.csv",
        "failures":ARTIFACTS/f"cloud_codegen_model_failure_breakdown{suffix}.csv",
        "report":ARTIFACTS/f"cloud_codegen_model_evaluation{suffix}.json",
        "overall_figure":ARTIFACTS/f"cloud_codegen_model_overall_accuracy{suffix}.png",
        "task_figure":ARTIFACTS/f"cloud_codegen_model_per_task_accuracy{suffix}.png",
    }


def configure_run_label(run_label: str | None) -> None:
    global RUN_LABEL,CHECKPOINT,PREDICTIONS,SUMMARY,PER_TASK,FAILURES,REPORT,OVERALL_FIGURE,TASK_FIGURE
    paths=artifact_paths(run_label);RUN_LABEL=run_label
    CHECKPOINT=paths["checkpoint"];PREDICTIONS=paths["predictions"];SUMMARY=paths["summary"]
    PER_TASK=paths["per_task"];FAILURES=paths["failures"];REPORT=paths["report"]
    OVERALL_FIGURE=paths["overall_figure"];TASK_FIGURE=paths["task_figure"]


def load_evaluation_samples() -> tuple[dict, list[dict]]:
    metadata, samples = load_frontend_benchmark(DATASET)
    distribution = Counter(sample["requested_analysis"] for sample in samples)
    expected = {task: 8 for task in TASKS}
    if metadata["shared_benchmark_samples"] != 60 or len(samples) != 40:
        raise ValueError("Cloud code-generation evaluation requires the locked 60/40 dataset.")
    if dict(distribution) != expected:
        raise ValueError(f"Unexpected task distribution: {dict(distribution)}")
    if any(sample["requested_analysis"] in {"figure2", "individual_profile"} for sample in samples):
        raise ValueError("Figure 2 and individual profile are not eligible for this benchmark.")
    if any(sample["privacy_route"] in {"blocked", "local_edge"} for sample in samples):
        raise ValueError("Blocked and Local Edge samples are not eligible for this benchmark.")
    return metadata, samples


def _model_map():
    models = get_cloud_codegen_evaluation_models()
    if tuple(model.key for model in models) != MODEL_KEYS:
        raise RuntimeError("Cloud code-generation evaluation registry must contain exactly three models.")
    return {model.key: model for model in models}


def _json_value(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def evaluate_pair(
    sample: dict,
    model_key: str,
    *,
    caller: Callable[..., ModelCallResult] = call_cloud_codegen_evaluation_model,
) -> dict:
    config = _model_map()[model_key]
    messages = build_code_generation_messages(sample["prompt"])
    call = caller(model_key, messages, max_tokens=2048)
    raw = call.content or ""
    cleaned, candidate_count = extract_restricted_assignment(raw)
    verification = verify_and_execute_generated_code(
        cleaned if raw else None,
        user_request=sample["prompt"],
        requested_analysis=sample["requested_analysis"],
        requested_filters=sample.get("analysis_filters") or {},
    )
    verified = verification_to_dict(verification)
    return {
        "sample_id": sample["id"],
        "prompt_version":CODE_GENERATION_PROMPT_VERSION,
        "requested_analysis": sample["requested_analysis"],
        "prompt": sample["prompt"],
        "privacy_route": sample["privacy_route"],
        "analysis_filters": sample.get("analysis_filters") or {},
        "model_key": model_key,
        "model_display_name": config.display_name,
        "provider": config.provider,
        "requested_model": call.requested_model,
        "actual_model": call.actual_model,
        "raw_response": raw,
        "generated_code": cleaned if raw else None,
        "candidate_assignment_count": candidate_count,
        "generated_method": verified.get("generated_method"),
        "generated_arguments": verified.get("generated_arguments"),
        "expected_method": verified.get("expected_method"),
        "structure_validation_passed": bool(verified.get("structure_validation_passed")),
        "request_match_passed": bool(verified.get("request_match_passed")),
        "local_execution_passed": bool(verified.get("local_execution_passed")),
        "result_validation_passed": bool(verified.get("result_validation_passed")),
        "fully_correct": bool(verified.get("fully_correct")),
        "failure_stage": None if verified.get("fully_correct") else (
            verified.get("failure_stage") if call.success else "api_error"
        ),
        "validation_error": verified.get("validation_error") if call.success else safe_error(call.error),
        "request_mismatches": verified.get("request_mismatches") or [],
        "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens,
        "total_tokens": call.total_tokens,
        "latency_seconds": call.latency_seconds,
        "api_call_success": bool(call.success and call.content),
    }


def _checkpoint_rows() -> list[dict]:
    if not CHECKPOINT.exists():
        return []
    rows = []
    for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _validate_checkpoint_prompt_version(rows: list[dict]) -> None:
    if any(row.get("prompt_version") != CODE_GENERATION_PROMPT_VERSION for row in rows):
        raise RuntimeError("Checkpoint prompt version does not match current prompt version.")


def _append_checkpoint(row: dict) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _latest_rows(rows: list[dict]) -> list[dict]:
    latest = {}
    for row in rows:
        latest[(row["sample_id"], row["model_key"])] = row
    return list(latest.values())


def load_predictions_csv() -> list[dict]:
    if not PREDICTIONS.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {PREDICTIONS}")
    boolean_fields={"structure_validation_passed","request_match_passed",
        "local_execution_passed","result_validation_passed","fully_correct","api_call_success"}
    integer_fields={"candidate_assignment_count","input_tokens","output_tokens","total_tokens"}
    float_fields={"latency_seconds"};json_fields={"analysis_filters","generated_arguments","request_mismatches"}
    rows=[]
    with PREDICTIONS.open(newline="",encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            for field in boolean_fields:row[field]=str(row.get(field)).casefold()=="true"
            for field in integer_fields:row[field]=int(row[field]) if row.get(field) not in {None,""} else None
            for field in float_fields:row[field]=float(row[field]) if row.get(field) not in {None,""} else None
            for field in json_fields:row[field]=json.loads(row[field]) if row.get(field) else ([] if field=="request_mismatches" else {})
            for field in ("failure_stage","validation_error","generated_method","expected_method","actual_model"):
                if row.get(field)=="":row[field]=None
            rows.append(row)
    return rows


def _mismatch_category(messages: list[str]) -> str | None:
    text = " ".join(messages).casefold()
    if not text:
        return None
    checks = (
        ("wrong_method", ("expected method",)),
        ("wrong_variables", ("public domains", "predictors", "variables")),
        ("wrong_filters", ("expected filters",)),
        ("wrong_target", ("expected target",)),
        ("wrong_controls", ("control specifications", "expected controls")),
        ("wrong_numeric_parameter", ("iterations", "threshold", "max_athletes")),
    )
    for category, patterns in checks:
        if any(pattern in text for pattern in patterns):
            return category
    return "other_request_mismatch"


def _rate(count: int, denominator: int, formal: bool) -> float | None:
    return count / denominator if formal and denominator else None


def build_results(rows: list[dict], metadata: dict) -> tuple[list[dict], list[dict], list[dict], dict]:
    models = _model_map()
    pricing = json.loads(PRICING.read_text(encoding="utf-8"))
    overall, per_task, failure_rows = [], [], []
    for key in MODEL_KEYS:
        selected = [row for row in rows if row["model_key"] == key]
        completed = sum(bool(row["api_call_success"]) for row in selected)
        formal = len(selected) == 40 and completed == 40
        counts = {
            "structure": sum(bool(row["structure_validation_passed"]) for row in selected),
            "request": sum(bool(row["request_match_passed"]) for row in selected),
            "execution": sum(bool(row["local_execution_passed"]) for row in selected),
            "result": sum(bool(row["result_validation_passed"]) for row in selected),
            "correct": sum(bool(row["fully_correct"]) and bool(row["api_call_success"]) for row in selected),
        }
        input_values = [row["input_tokens"] for row in selected if row.get("input_tokens") is not None]
        output_values = [row["output_tokens"] for row in selected if row.get("output_tokens") is not None]
        latencies = [row["latency_seconds"] for row in selected if row.get("latency_seconds") is not None]
        cost = None
        if len(input_values) == len(selected) == len(output_values) and selected:
            prices = pricing["per_1m_tokens"][key]
            cost = sum(input_values) / 1_000_000 * prices["input"] + sum(output_values) / 1_000_000 * prices["output"]
        overall.append({
            "prompt_version":CODE_GENERATION_PROMPT_VERSION,
            "Model": models[key].display_name, "model_key": key, "Samples": len(selected),
            "Structure Valid": counts["structure"], "Structure Valid Rate": _rate(counts["structure"], 40, formal),
            "Request Match": counts["request"], "Request Match Rate": _rate(counts["request"], 40, formal),
            "Execution Success": counts["execution"], "Execution Success Rate": _rate(counts["execution"], 40, formal),
            "Result Valid": counts["result"], "Result Valid Rate": _rate(counts["result"], 40, formal),
            "Fully Correct": counts["correct"], "Fully Correct Accuracy": _rate(counts["correct"], 40, formal),
            "API Calls": len(selected), "Completed Model Responses": completed,
            "Input Tokens": sum(input_values) if len(input_values) == len(selected) and selected else None,
            "Output Tokens": sum(output_values) if len(output_values) == len(selected) and selected else None,
            "Average Latency Seconds": sum(latencies) / len(latencies) if latencies else None,
            "total_estimated_cost_usd": cost,
            "average_cost_per_request_usd": cost / 40 if cost is not None and formal else None,
            "cost_per_fully_correct_request_usd": cost / counts["correct"] if cost is not None and counts["correct"] else None,
            "cost_is_pricing_snapshot_estimate": True,
        })
        failures = Counter(row.get("failure_stage") or "success" for row in selected)
        mismatches = Counter(
            category for row in selected
            if (category := _mismatch_category(row.get("request_mismatches") or []))
        )
        failure_rows.append({"prompt_version":CODE_GENERATION_PROMPT_VERSION,
            "Model": models[key].display_name, **{
            name: failures.get(name, 0) for name in (
                "api_error", "format_validation", "request_validation",
                "local_execution", "result_validation", "success",
            )}, **mismatches})
        for task in TASKS:
            task_rows = [row for row in selected if row["requested_analysis"] == task]
            correct = sum(bool(row["fully_correct"]) and bool(row["api_call_success"]) for row in task_rows)
            per_task.append({"prompt_version":CODE_GENERATION_PROMPT_VERSION,
                "Task": TASK_LABELS[task], "Model": models[key].display_name,
                "model_key":key,
                "Correct": correct, "Total": len(task_rows),
                "Accuracy": _rate(correct, 8, len(task_rows)==8)})
    complete = len(rows) == 120 and all(
        sum(row["model_key"]==key for row in rows)==40
        and sum(row["model_key"]==key and bool(row.get("api_call_success")) for row in rows)==40
        for key in MODEL_KEYS
    )
    report = {
        "status": "complete" if complete else "incomplete",
        "evaluation_type": "cloud_llm_restricted_code_generation_first_pass",
        "dataset": metadata["dataset_path"], "dataset_sha256": metadata["dataset_sha256"],
        "evaluation_samples_per_model": 40, "models": 3,
        "total_model_generations": len(rows),
        "prompt_type": "shared_pool_based_restricted_code_prompt",
        "prompt_version":CODE_GENERATION_PROMPT_VERSION,
        "run_label":RUN_LABEL,
        "gold_contract_exposed_to_model": False, "llm_judge_used": False,
        "cost_is_pricing_snapshot_estimate": True,
        "overall_results": overall, "per_task_results": per_task,
        "failure_breakdown": failure_rows, "per_sample_results": rows,
    }
    return overall, per_task, failure_rows, report


def _write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    if not rows:
        return
    fields = columns or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _json_value(row.get(name)) if isinstance(row.get(name), (dict, list)) else row.get(name) for name in fields})


def _plot(overall: list[dict], per_task: list[dict]) -> None:
    import matplotlib.pyplot as plt
    models=_model_map();labels = [models[key].display_name for key in MODEL_KEYS]
    overall_by_key={row["model_key"]:row for row in overall}
    values = [(overall_by_key[key]["Fully Correct Accuracy"] or 0) * 100 for key in MODEL_KEYS]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#3b82f6", "#10b981", "#8b5cf6"])
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=3)
    ax.set_ylabel("Fully Correct Accuracy (%)"); ax.set_ylim(0, 100)
    ax.set_title("Cloud LLM Restricted Code Fully Correct Accuracy")
    ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(OVERALL_FIGURE, dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 6)); width=.24; x=range(len(TASKS))
    for model_index, key in enumerate(MODEL_KEYS):
        model=models[key].display_name
        lookup = {row["Task"]: (row["Accuracy"] or 0) * 100 for row in per_task if row["model_key"] == key}
        positions=[value+(model_index-1)*width for value in x]
        ax.bar(positions,[lookup[TASK_LABELS[task]] for task in TASKS],width,label=model)
    ax.set_xticks(list(x),[TASK_LABELS[task] for task in TASKS]); ax.set_ylim(0,100)
    ax.set_ylabel("Fully Correct Accuracy (%)"); ax.set_title("Fully Correct Accuracy by Analysis Task")
    ax.legend(); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(TASK_FIGURE,dpi=300); plt.close(fig)


def save_artifacts(rows: list[dict], metadata: dict, *, write_predictions: bool = True) -> dict:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    overall, per_task, failure_rows, report = build_results(rows, metadata)
    prediction_columns = [
        "sample_id", "prompt_version", "requested_analysis", "prompt", "privacy_route", "analysis_filters",
        "model_key", "model_display_name", "provider", "requested_model", "actual_model",
        "raw_response", "generated_code", "candidate_assignment_count", "generated_method",
        "generated_arguments", "expected_method", "structure_validation_passed",
        "request_match_passed", "local_execution_passed", "result_validation_passed",
        "fully_correct", "failure_stage", "validation_error", "request_mismatches",
        "input_tokens", "output_tokens", "total_tokens", "latency_seconds", "api_call_success",
    ]
    if write_predictions:
        _write_csv(PREDICTIONS, rows, prediction_columns)
    _write_csv(SUMMARY, overall)
    _write_csv(PER_TASK, per_task); _write_csv(FAILURES, failure_rows)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    _plot(overall, per_task)
    return report


def _percent(value) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def print_report(report: dict) -> None:
    print("=" * 60); print("Cloud LLM Restricted Code Generation Evaluation"); print("=" * 60)
    print(f"Active prompt version: {CODE_GENERATION_PROMPT_VERSION}")
    print("Dataset samples per model: 40\nModels: 3")
    print(f"Total first-pass model generations: {report['total_model_generations']}\n")
    print(f"{'Model':24} {'Struct':>8} {'Request':>8} {'Execute':>8} {'Result':>8} {'Fully Correct':>14}")
    for row in report["overall_results"]:
        print(f"{row['Model']:24} {_percent(row['Structure Valid Rate']):>8} {_percent(row['Request Match Rate']):>8} "
              f"{_percent(row['Execution Success Rate']):>8} {_percent(row['Result Valid Rate']):>8} {_percent(row['Fully Correct Accuracy']):>14}")
    print("\n" + "-" * 60); print("Fully Correct Accuracy by Task"); print("-" * 60)
    for task in TASK_LABELS.values():
        values=[_percent(row["Accuracy"]) for row in report["per_task_results"] if row["Task"]==task]
        print(f"{task:24} " + " ".join(f"{value:>10}" for value in values))
    print(f"\nStatus: {report['status']}")


def check_only(model_keys: list[str]) -> int:
    messages=[{"role":"user","content":"Reply only with OK."}]
    failed=False
    for key in model_keys:
        runtime=get_cloud_codegen_evaluation_runtime(key)
        print(f"{runtime['display_name']}:")
        print(f"  API key loaded: {'yes' if runtime['api_key_loaded'] else 'no'}")
        print(f"  model: {runtime['model']}")
        print(f"  base URL: {runtime['base_url'] or 'OpenAI default'}")
        if not runtime["api_key_loaded"]:
            print(f"  missing credential environment variable: {runtime['api_key_env']}")
            failed=True
            continue
        result=call_cloud_codegen_evaluation_model(key,messages,max_tokens=20)
        print(f"  access: {'available' if result.success else safe_error(result.error)}")
        failed |= not result.success
    return 1 if failed else 0


def smoke(model_keys: list[str]) -> int:
    _, samples=load_evaluation_samples(); sample=samples[0]
    for key in model_keys:
        row=evaluate_pair(sample,key)
        print(f"\n{row['model_display_name']}")
        print(f"requested_model={row['requested_model']}")
        print(f"actual_model={row['actual_model']}")
        print(f"provider={row['provider']}")
        print(f"generated_code={row['generated_code']}")
        print(f"api_success={row['api_call_success']}")
        print(f"api_error={row['validation_error'] if not row['api_call_success'] else None}")
        print(f"structure_validation={row['structure_validation_passed']}")
        print(f"request_match={row['request_match_passed']}")
        print(f"local_execution={row['local_execution_passed']}")
        print(f"result_validation={row['result_validation_passed']}")
        print(f"fully_correct={row['fully_correct']}")
        print(f"failure_stage={row['failure_stage']}")
    return 0


def cleanup_old_v1() -> list[Path]:
    """Delete only the explicitly retired cloud code-generation result files."""
    deleted=[]
    for name in OLD_V1_ARTIFACT_NAMES:
        path=ARTIFACTS/name
        if path.is_file():
            path.unlink()
            deleted.append(path)
            print(f"Deleted: {path}")
    if not deleted:
        print("No old V1 cloud code-generation artifacts found.")
    return deleted


def rebuild_report() -> dict:
    metadata,_=load_evaluation_samples();rows=load_predictions_csv();counts=Counter(row["model_key"] for row in rows)
    expected={key:40 for key in MODEL_KEYS}
    if dict(counts)!=expected:
        raise ValueError(f"Prediction CSV must contain exactly 40 rows per model; found {dict(counts)}")
    models=_model_map()
    for row in rows:row["model_display_name"]=models[row["model_key"]].display_name
    print("Prediction rows:")
    for key in MODEL_KEYS:print(f"{models[key].display_name}: {counts[key]}")
    report=save_artifacts(rows,metadata,write_predictions=False);print_report(report);return report


def run_formal(model_keys: list[str], *, resume: bool, fresh: bool) -> dict:
    print(f"Active prompt version: {CODE_GENERATION_PROMPT_VERSION}")
    metadata,samples=load_evaluation_samples(); ARTIFACTS.mkdir(parents=True,exist_ok=True)
    if fresh and CHECKPOINT.exists(): CHECKPOINT.unlink()
    if CHECKPOINT.exists() and not resume and not fresh:
        raise RuntimeError("Checkpoint exists. Use --resume or --fresh.")
    checkpoint=_checkpoint_rows() if resume else []
    if resume:
        _validate_checkpoint_prompt_version(checkpoint)
    successful={(row["sample_id"],row["model_key"]) for row in checkpoint if row.get("api_call_success")}
    rows=list(checkpoint)
    for key in model_keys:
        for sample in samples:
            pair=(sample["id"],key)
            if pair in successful: continue
            row=evaluate_pair(sample,key); _append_checkpoint(row); rows.append(row)
    selected=_latest_rows(rows)
    report=save_artifacts(selected,metadata); print_report(report); return report


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(); mode.add_argument("--check-only",action="store_true"); mode.add_argument("--smoke",action="store_true"); mode.add_argument("--rebuild-report",action="store_true"); mode.add_argument("--cleanup-old-v1",action="store_true")
    parser.add_argument("--models",nargs="+",choices=MODEL_KEYS,default=list(MODEL_KEYS))
    parser.add_argument("--run-label")
    parser.add_argument("--resume",action="store_true"); parser.add_argument("--fresh",action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args=parse_args(argv)
    if args.cleanup_old_v1:
        cleanup_old_v1();return 0
    configure_run_label(args.run_label)
    if args.check_only:return check_only(args.models)
    if args.smoke:return smoke(args.models)
    if args.rebuild_report:rebuild_report();return 0
    run_formal(args.models,resume=args.resume,fresh=args.fresh); return 0


if __name__=="__main__":
    raise SystemExit(main())
