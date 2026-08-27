"""Formal first-pass comparison of three llama.cpp Local code generators."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.code_generation_prompt import CODE_GENERATION_PROMPT_VERSION, build_code_generation_messages
from llm.generated_code_verifier import extract_restricted_assignment, to_dict as verification_to_dict, verify_and_execute_generated_code
from llm.model_clients import ModelCallResult, call_local_codegen_model
from scripts.athlete_router_evaluation_common import load_frontend_benchmark, safe_error

DATASET = ROOT / "evaluation/frontend_realistic_benchmark_60.json"
ARTIFACTS = ROOT / "artifacts"
CHECKPOINT = ARTIFACTS / "local_codegen_model_checkpoint.jsonl"
PREDICTIONS = ARTIFACTS / "local_codegen_model_predictions.csv"
SUMMARY = ARTIFACTS / "local_codegen_model_summary.csv"
PER_TASK = ARTIFACTS / "local_codegen_model_per_task.csv"
FAILURES = ARTIFACTS / "local_codegen_model_failure_breakdown.csv"
REPORT = ARTIFACTS / "local_codegen_model_evaluation.json"
OVERALL_FIGURE = ARTIFACTS / "local_codegen_model_overall_accuracy.png"
TASK_FIGURE = ARTIFACTS / "local_codegen_model_per_task_accuracy.png"
LATENCY_FIGURE = ARTIFACTS / "local_codegen_model_latency.png"
TASKS = ("table1", "table2", "figure1", "correlation", "variance_analysis")
TASK_LABELS = {"table1":"Table 1", "table2":"Table 2", "figure1":"Figure 1", "correlation":"Correlation", "variance_analysis":"Variance Analysis"}
MODEL_KEYS = ("ministral", "qwen", "llama")
MAX_GENERATION_TOKENS = 2048
LOCAL_COST_NOTE = ("Local models do not have a per-request external API price in this experiment. Local hardware, electricity, and energy costs are not measured; latency is reported as the local efficiency metric.")

@dataclass(frozen=True)
class LocalEvaluationModel:
    key: str
    display_name: str
    hf_spec: str
    alias: str
    provider: str = "openai_compatible"

LOCAL_MODELS = (
    LocalEvaluationModel("ministral", "Ministral-3-8B", "mistralai/Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M", "ministral-3-8b-local-eval"),
    LocalEvaluationModel("qwen", "Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M", "qwen2.5-coder-7b-local-eval"),
    LocalEvaluationModel("llama", "Llama-3.1-8B-Instruct", "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M", "llama-3.1-8b-local-eval"),
)

def model_map() -> dict[str, LocalEvaluationModel]:
    if tuple(model.key for model in LOCAL_MODELS) != MODEL_KEYS:
        raise RuntimeError("Local evaluation registry must contain ministral, qwen, and llama in order.")
    if len({model.alias for model in LOCAL_MODELS}) != len(LOCAL_MODELS):
        raise RuntimeError("Local evaluation aliases must be unique.")
    if any(":Q4_K_M" not in model.hf_spec for model in LOCAL_MODELS):
        raise RuntimeError("Every Local evaluation model must use Q4_K_M.")
    return {model.key:model for model in LOCAL_MODELS}

def load_evaluation_samples() -> tuple[dict, list[dict]]:
    metadata, samples = load_frontend_benchmark(DATASET)
    distribution = Counter(sample["requested_analysis"] for sample in samples)
    if metadata["shared_benchmark_samples"] != 60 or len(samples) != 40:
        raise ValueError("Local code-generation evaluation requires the locked 60/40 dataset.")
    if dict(distribution) != {task:8 for task in TASKS}:
        raise ValueError(f"Unexpected task distribution: {dict(distribution)}")
    if any(sample["requested_analysis"] in {"figure2", "individual_profile"} for sample in samples):
        raise ValueError("Figure 2 and individual profile are not eligible for this benchmark.")
    if any(sample["privacy_route"] in {"blocked", "local_edge"} for sample in samples):
        raise ValueError("Blocked and Local Edge samples are not eligible for this benchmark.")
    return metadata, samples

def find_llama_server() -> str | None:
    configured = os.getenv("LLAMA_SERVER_PATH")
    if configured:
        path = Path(configured)
        return str(path.resolve()) if path.is_file() else None
    return shutil.which("llama-server") or shutil.which("llama-server.exe")

def port_is_available(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False

def _ready_alias(port: int) -> set[str] | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return {str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict)}
    except (OSError, ValueError, json.JSONDecodeError):
        return None

@contextmanager
def temporary_local_runtime(model: LocalEvaluationModel, port: int):
    values = {"LLM_LOCAL_PROVIDER":model.provider, "LLM_LOCAL_MODEL":model.alias,
        "LLM_LOCAL_BASE_URL":f"http://127.0.0.1:{port}/v1", "LLM_LOCAL_API_KEY":"none"}
    previous = {name:os.environ.get(name) for name in values}; os.environ.update(values)
    try:
        yield
    finally:
        for name, old in previous.items():
            if old is None: os.environ.pop(name, None)
            else: os.environ[name] = old

@contextmanager
def running_model_server(model: LocalEvaluationModel, *, port: int, timeout_seconds: int,
                         log_dir: Path | None = None):
    executable = find_llama_server()
    if not executable:
        raise RuntimeError("llama-server was not found in PATH and LLAMA_SERVER_PATH is not a valid file.")
    if not port_is_available(port):
        raise RuntimeError(f"Evaluation port {port} is already in use; no process was stopped.")
    active_log_dir = Path(log_dir) if log_dir is not None else ARTIFACTS
    active_log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = active_log_dir / f"local_codegen_{model.key}_server_stdout.log"
    stderr_path = active_log_dir / f"local_codegen_{model.key}_server_stderr.log"
    command = [executable, "-hf", model.hf_spec, "--alias", model.alias, "--host", "127.0.0.1", "--port", str(port)]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    process = None
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        try:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, creationflags=flags)
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"{model.display_name} llama-server exited before readiness; see {stderr_path}.")
                aliases = _ready_alias(port)
                if aliases is not None and model.alias in aliases: break
                time.sleep(1)
            else:
                raise RuntimeError(f"{model.display_name} did not become ready within {timeout_seconds} seconds; see {stderr_path}.")
            with temporary_local_runtime(model, port):
                yield process
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=10)

def warm_up_model() -> ModelCallResult:
    result = call_local_codegen_model([{"role":"user", "content":"Reply only with OK."}], temperature=0.0, max_tokens=16)
    if not result.success:
        raise RuntimeError(f"Local model warm-up failed: {safe_error(result.error)}")
    return result

def evaluate_pair(sample: dict, model_key: str, *, caller: Callable[..., ModelCallResult] = call_local_codegen_model) -> dict:
    config = model_map()[model_key]
    call = caller(build_code_generation_messages(sample["prompt"]), temperature=0.0, max_tokens=MAX_GENERATION_TOKENS)
    raw = call.content or ""; cleaned, candidate_count = extract_restricted_assignment(raw)
    verified = verification_to_dict(verify_and_execute_generated_code(cleaned if raw else None,
        user_request=sample["prompt"], requested_analysis=sample["requested_analysis"],
        requested_filters=sample.get("analysis_filters") or {}, close_figures_after_execution=True))
    return {
        "sample_id":sample["id"], "prompt_version":CODE_GENERATION_PROMPT_VERSION,
        "requested_analysis":sample["requested_analysis"], "prompt":sample["prompt"],
        "privacy_route":sample["privacy_route"], "analysis_filters":sample.get("analysis_filters") or {},
        "model_key":model_key, "model_display_name":config.display_name, "model_hf_spec":config.hf_spec,
        "model_alias":config.alias, "provider":config.provider, "requested_model":call.requested_model,
        "actual_model":call.actual_model, "raw_response":raw, "generated_code":cleaned if raw else None,
        "candidate_assignment_count":candidate_count, "generated_method":verified.get("generated_method"),
        "generated_arguments":verified.get("generated_arguments"), "expected_method":verified.get("expected_method"),
        "structure_validation_passed":bool(verified.get("structure_validation_passed")),
        "request_match_passed":bool(verified.get("request_match_passed")),
        "local_execution_passed":bool(verified.get("local_execution_passed")),
        "result_validation_passed":bool(verified.get("result_validation_passed")),
        "fully_correct":bool(verified.get("fully_correct")),
        "failure_stage":None if verified.get("fully_correct") else (verified.get("failure_stage") if call.success else "api_error"),
        "validation_error":verified.get("validation_error") if call.success else safe_error(call.error),
        "request_mismatches":verified.get("request_mismatches") or [], "input_tokens":call.input_tokens,
        "output_tokens":call.output_tokens, "total_tokens":call.total_tokens, "latency_seconds":call.latency_seconds,
        "api_call_success":bool(call.success and call.content),
    }

def _mismatch_category(messages: list[str]) -> str | None:
    text = " ".join(messages).casefold()
    if not text: return None
    checks = (("wrong_method",("expected method",)), ("wrong_variables",("public domains","predictors","variables")),
        ("wrong_filters",("expected filters",)), ("wrong_target",("expected target",)),
        ("wrong_controls",("control specifications","expected controls")),
        ("wrong_numeric_parameter",("iterations","threshold","max_athletes")))
    for category, patterns in checks:
        if any(pattern in text for pattern in patterns): return category
    return "other_request_mismatch"

def _rate(count: int, denominator: int, formal: bool) -> float | None:
    return count / denominator if formal and denominator else None

def build_results(rows: list[dict], metadata: dict) -> tuple[list[dict], list[dict], list[dict], dict]:
    models = model_map(); overall, per_task, failure_rows = [], [], []
    for key in MODEL_KEYS:
        selected = [row for row in rows if row["model_key"] == key]
        completed = sum(bool(row.get("api_call_success")) for row in selected); formal = len(selected) == completed == 40
        counts = {"structure":sum(bool(row.get("structure_validation_passed")) for row in selected),
            "request":sum(bool(row.get("request_match_passed")) for row in selected),
            "execution":sum(bool(row.get("local_execution_passed")) for row in selected),
            "result":sum(bool(row.get("result_validation_passed")) for row in selected),
            "correct":sum(bool(row.get("fully_correct") and row.get("api_call_success")) for row in selected)}
        latencies = [float(row["latency_seconds"]) for row in selected if row.get("latency_seconds") is not None]
        overall.append({"prompt_version":CODE_GENERATION_PROMPT_VERSION, "Model":models[key].display_name,
            "model_key":key, "Samples":len(selected), "Structure Valid":counts["structure"],
            "Structure Valid Rate":_rate(counts["structure"],40,formal), "Request Match":counts["request"],
            "Request Match Rate":_rate(counts["request"],40,formal), "Execution Success":counts["execution"],
            "Execution Success Rate":_rate(counts["execution"],40,formal), "Result Valid":counts["result"],
            "Result Valid Rate":_rate(counts["result"],40,formal), "Fully Correct":counts["correct"],
            "Fully Correct Accuracy":_rate(counts["correct"],40,formal), "Completed Model Responses":completed,
            "Average Latency Seconds":statistics.mean(latencies) if latencies else None,
            "Median Latency Seconds":statistics.median(latencies) if latencies else None, "Cost Note":LOCAL_COST_NOTE})
        failures = Counter(row.get("failure_stage") or "success" for row in selected)
        mismatches = Counter(category for row in selected if (category := _mismatch_category(row.get("request_mismatches") or [])))
        failure_rows.append({"prompt_version":CODE_GENERATION_PROMPT_VERSION, "Model":models[key].display_name,
            **{name:failures.get(name,0) for name in ("api_error","format_validation","request_validation","local_execution","result_validation","success")},
            **{name:mismatches.get(name,0) for name in ("wrong_method","wrong_variables","wrong_filters","wrong_target","wrong_controls","wrong_numeric_parameter","other_request_mismatch")}})
        for task in TASKS:
            task_rows = [row for row in selected if row["requested_analysis"] == task]
            correct = sum(bool(row.get("fully_correct") and row.get("api_call_success")) for row in task_rows)
            per_task.append({"prompt_version":CODE_GENERATION_PROMPT_VERSION, "Task":TASK_LABELS[task],
                "Model":models[key].display_name, "model_key":key, "Correct":correct, "Total":len(task_rows),
                "Accuracy":_rate(correct,8,len(task_rows)==8)})
    complete = len(rows) == 120 and all(sum(row["model_key"]==key for row in rows)==40 and
        sum(row["model_key"]==key and bool(row.get("api_call_success")) for row in rows)==40 for key in MODEL_KEYS)
    report = {"status":"complete" if complete else "incomplete",
        "evaluation_type":"local_llm_restricted_code_generation_first_pass", "dataset":metadata["dataset_path"],
        "dataset_sha256":metadata["dataset_sha256"], "evaluation_samples_per_model":40, "models":3,
        "total_model_generations":len(rows), "prompt_version":CODE_GENERATION_PROMPT_VERSION,
        "generation_temperature":0.0, "max_generation_tokens":MAX_GENERATION_TOKENS,
        "one_model_loaded_at_a_time":True, "gold_contract_exposed_to_model":False,
        "llm_judge_used":False, "local_cost_note":LOCAL_COST_NOTE, "overall_results":overall,
        "per_task_results":per_task, "failure_breakdown":failure_rows, "per_sample_results":rows}
    return overall, per_task, failure_rows, report

def _write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    if not rows: return
    fields = columns or list(dict.fromkeys(name for row in rows for name in row))
    with path.open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore");writer.writeheader()
        for row in rows: writer.writerow({name:json.dumps(row.get(name),ensure_ascii=False,sort_keys=True) if isinstance(row.get(name),(dict,list)) else row.get(name) for name in fields})

def _plot(overall: list[dict], per_task: list[dict]) -> None:
    from sports import matplotlib_backend as _matplotlib_backend
    import matplotlib.pyplot as plt
    models=model_map();labels=[models[key].display_name for key in MODEL_KEYS];lookup={row["model_key"]:row for row in overall}
    values=[(lookup[key]["Fully Correct Accuracy"] or 0)*100 for key in MODEL_KEYS]
    fig,axis=plt.subplots(figsize=(9,5));bars=axis.bar(labels,values);axis.bar_label(bars,labels=[f"{value:.1f}%" for value in values],padding=3)
    axis.set_ylim(0,100);axis.set_ylabel("Fully Correct Accuracy (%)");axis.set_title("Local LLM Restricted Code Fully Correct Accuracy");fig.tight_layout();fig.savefig(OVERALL_FIGURE,dpi=300);plt.close(fig)
    fig,axis=plt.subplots(figsize=(12,6));width=.24;positions=range(len(TASKS))
    for index,key in enumerate(MODEL_KEYS):
        task_lookup={row["Task"]:(row["Accuracy"] or 0)*100 for row in per_task if row["model_key"]==key}
        axis.bar([position+(index-1)*width for position in positions],[task_lookup[TASK_LABELS[task]] for task in TASKS],width,label=models[key].display_name)
    axis.set_xticks(list(positions),[TASK_LABELS[task] for task in TASKS]);axis.set_ylim(0,100);axis.set_ylabel("Fully Correct Accuracy (%)");axis.legend();fig.tight_layout();fig.savefig(TASK_FIGURE,dpi=300);plt.close(fig)
    latency=[lookup[key]["Average Latency Seconds"] or 0 for key in MODEL_KEYS]
    fig,axis=plt.subplots(figsize=(9,5));axis.bar(labels,latency);axis.set_ylabel("Average Generation Latency (seconds)");fig.tight_layout();fig.savefig(LATENCY_FIGURE,dpi=300);plt.close(fig)

def save_artifacts(rows: list[dict], metadata: dict) -> dict:
    ARTIFACTS.mkdir(parents=True,exist_ok=True);overall,per_task,failures,report=build_results(rows,metadata)
    columns=["sample_id","prompt_version","requested_analysis","prompt","privacy_route","analysis_filters","model_key","model_display_name","model_hf_spec","model_alias","provider","requested_model","actual_model","raw_response","generated_code","candidate_assignment_count","generated_method","generated_arguments","expected_method","structure_validation_passed","request_match_passed","local_execution_passed","result_validation_passed","fully_correct","failure_stage","validation_error","request_mismatches","input_tokens","output_tokens","total_tokens","latency_seconds","api_call_success"]
    _write_csv(PREDICTIONS,rows,columns);_write_csv(SUMMARY,overall);_write_csv(PER_TASK,per_task);_write_csv(FAILURES,failures)
    REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str),encoding="utf-8");_plot(overall,per_task);return report

def checkpoint_rows() -> list[dict]:
    if not CHECKPOINT.exists(): return []
    return [json.loads(line) for line in CHECKPOINT.read_text(encoding="utf-8").splitlines() if line.strip()]

def validate_checkpoint(rows: list[dict]) -> None:
    models=model_map()
    for row in rows:
        if row.get("prompt_version") != CODE_GENERATION_PROMPT_VERSION: raise RuntimeError("Checkpoint prompt version does not match the current prompt version.")
        config=models.get(row.get("model_key"))
        if config and (row.get("model_hf_spec") != config.hf_spec or row.get("model_alias") != config.alias): raise RuntimeError("Checkpoint Local model specification does not match the current registry.")

def append_checkpoint(row: dict) -> None:
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    with CHECKPOINT.open("a",encoding="utf-8") as handle: handle.write(json.dumps(row,ensure_ascii=False,default=str)+"\n")

def latest_rows(rows: list[dict]) -> list[dict]:
    return list({(row["sample_id"],row["model_key"]):row for row in rows}.values())

def _percent(value) -> str: return "N/A" if value is None else f"{value*100:.1f}%"

def print_report(report: dict) -> None:
    print("="*60);print("Local LLM Restricted Code Generation Evaluation");print("="*60)
    print(f"Active prompt version: {CODE_GENERATION_PROMPT_VERSION}");print("Dataset samples per model: 40\nModels: 3");print(f"Total first-pass model generations: {report['total_model_generations']}\n")
    print(f"{'Model':30} {'Struct':>8} {'Request':>8} {'Execute':>8} {'Result':>8} {'Fully Correct':>14} {'Avg Latency':>12}")
    for row in report["overall_results"]:
        latency="N/A" if row["Average Latency Seconds"] is None else f"{row['Average Latency Seconds']:.2f}"
        print(f"{row['Model']:30} {_percent(row['Structure Valid Rate']):>8} {_percent(row['Request Match Rate']):>8} {_percent(row['Execution Success Rate']):>8} {_percent(row['Result Valid Rate']):>8} {_percent(row['Fully Correct Accuracy']):>14} {latency:>12}")
    print("\n"+"-"*60);print("Fully Correct Accuracy by Task");print("-"*60)
    for task in TASK_LABELS.values():
        values=[_percent(row["Accuracy"]) for row in report["per_task_results"] if row["Task"]==task];print(f"{task:24} "+" ".join(f"{value:>10}" for value in values))
    print(f"\nCost note: {LOCAL_COST_NOTE}\nStatus: {report['status']}")

def check_only(model_keys: list[str], *, port: int) -> int:
    metadata,samples=load_evaluation_samples();models=model_map();executable=find_llama_server();available=port_is_available(port)
    ARTIFACTS.mkdir(parents=True,exist_ok=True);writable=os.access(ARTIFACTS,os.W_OK)
    print(f"Dataset: {metadata['dataset_path']} ({metadata['shared_benchmark_samples']} shared / {len(samples)} eligible)")
    print(f"llama-server: {executable or 'NOT FOUND'}");print(f"Evaluation port {port}: {'available' if available else 'already in use'}");print(f"Artifacts directory: {'writable' if writable else 'not writable'}")
    for key in model_keys:
        model=models[key];print(f"{model.display_name}: {model.hf_spec} -> {model.alias}")
    return int(not(executable and available and writable))

def smoke(model_keys: list[str], *, port: int, timeout_seconds: int) -> int:
    _,samples=load_evaluation_samples();sample=samples[0];failed=False
    for key in model_keys:
        model=model_map()[key]
        with running_model_server(model,port=port,timeout_seconds=timeout_seconds): warm_up_model();row=evaluate_pair(sample,key)
        failed |= not row["api_call_success"];print(f"\n{row['model_display_name']}")
        for label,field in (("Structure Validation","structure_validation_passed"),("Request Match Validation","request_match_passed"),("Local Execution","local_execution_passed"),("Result Validation","result_validation_passed"),("Fully Correct","fully_correct")): print(f"{label}: {row[field]}")
        print(f"Generated code: {row['generated_code']}");print(f"Failure stage: {row['failure_stage']}");print(f"Validation error: {row['validation_error']}")
    return int(failed)

def run_formal(model_keys: list[str], *, port: int, timeout_seconds: int, fresh: bool, resume: bool) -> dict:
    metadata,samples=load_evaluation_samples();ARTIFACTS.mkdir(parents=True,exist_ok=True)
    if fresh and CHECKPOINT.exists(): CHECKPOINT.unlink()
    if CHECKPOINT.exists() and not fresh and not resume: raise RuntimeError("Local evaluation checkpoint exists. Use --resume or --fresh.")
    prior=checkpoint_rows() if resume else []
    if resume: validate_checkpoint(prior)
    successful={(row["sample_id"],row["model_key"]) for row in prior if row.get("api_call_success")};rows=list(prior)
    for key in model_keys:
        pending=[sample for sample in samples if (sample["id"],key) not in successful]
        if not pending: continue
        model=model_map()[key]
        with running_model_server(model,port=port,timeout_seconds=timeout_seconds):
            warm_up_model()
            for sample in pending:
                row=evaluate_pair(sample,key);append_checkpoint(row);rows.append(row)
    report=save_artifacts(latest_rows(rows),metadata);print_report(report);return report

def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group();mode.add_argument("--check-only",action="store_true");mode.add_argument("--smoke",action="store_true")
    lifecycle=parser.add_mutually_exclusive_group();lifecycle.add_argument("--fresh",action="store_true");lifecycle.add_argument("--resume",action="store_true")
    parser.add_argument("--models",nargs="+",choices=MODEL_KEYS,default=list(MODEL_KEYS));parser.add_argument("--port",type=int,default=8081);parser.add_argument("--server-timeout-seconds",type=int,default=1800);args=parser.parse_args(argv)
    if not 1<=args.port<=65535: parser.error("--port must be between 1 and 65535")
    if args.server_timeout_seconds<=0: parser.error("--server-timeout-seconds must be positive")
    return args

def main(argv=None) -> int:
    args=parse_args(argv)
    if args.check_only:return check_only(args.models,port=args.port)
    if args.smoke:return smoke(args.models,port=args.port,timeout_seconds=args.server_timeout_seconds)
    run_formal(args.models,port=args.port,timeout_seconds=args.server_timeout_seconds,fresh=args.fresh,resume=args.resume);return 0

if __name__=="__main__": raise SystemExit(main())
