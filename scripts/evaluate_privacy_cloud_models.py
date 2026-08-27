"""Formal evaluation of three cloud LLMs as the shared Privacy Assessor."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.env import load_local_env
from llm.model_clients import (
    ModelCallResult,
    call_cloud_privacy_evaluation_model,
    get_cloud_codegen_evaluation_models,
    get_cloud_codegen_evaluation_runtime,
)
from privacy.llm_privacy_assessor import (
    ASSESSMENT_RULES_VERSION,
    ASSESSMENT_SCHEMA_VERSION,
    build_privacy_assessment_messages,
    parse_privacy_assessment_response,
)
from privacy.llm_soft_gating_model import DEFAULT_MODEL_PATH, FEATURE_NAMES
from privacy.prism_router import trained_soft_gating_features


DATASET = ROOT / "evaluation/frontend_realistic_benchmark_60.json"
ARTIFACTS = ROOT / "artifacts"
CHECKPOINT = ARTIFACTS / "privacy_cloud_model_checkpoint.jsonl"
PREDICTIONS = ARTIFACTS / "privacy_cloud_model_predictions.csv"
SUMMARY = ARTIFACTS / "privacy_cloud_model_summary.csv"
PER_ROUTE = ARTIFACTS / "privacy_cloud_model_per_route.csv"
FEATURE_SUMMARY = ARTIFACTS / "privacy_cloud_model_feature_summary.csv"
REPORT = ARTIFACTS / "privacy_cloud_model_evaluation.json"
ROUTE_FIGURE = ARTIFACTS / "privacy_cloud_model_route_accuracy.png"
PER_ROUTE_FIGURE = ARTIFACTS / "privacy_cloud_model_per_route_accuracy.png"

MODEL_KEYS = ("gpt4_1", "gemini", "claude")
ROUTES = ("cloud", "collaboration", "local_edge", "blocked")
EXPECTED_DISTRIBUTION = {"cloud": 5, "collaboration": 35, "local_edge": 10, "blocked": 10}
PROMPT_VERSION = f"{ASSESSMENT_RULES_VERSION}|{ASSESSMENT_SCHEMA_VERSION}"
PRIVACY_EVALUATION_MAX_TOKENS = 1500


_COMPLETE_OUTER_FENCE = re.compile(
    r"^\s*```(?:json)?[ \t]*\r?\n(?P<content>[\s\S]*?)\r?\n```\s*$",
    re.IGNORECASE,
)


def normalize_privacy_evaluation_response(content: str) -> str:
    """Remove only one complete outer Markdown fence; never repair JSON."""
    if not isinstance(content, str):
        return content
    match = _COMPLETE_OUTER_FENCE.fullmatch(content)
    return match.group("content") if match else content


def is_plain_json_response(content: str | None) -> bool:
    return bool(isinstance(content, str) and content.strip().startswith("{")
        and not content.strip().startswith("```"))


def dataset_digest(payload: dict) -> str:
    clean = {key: value for key, value in payload.items() if key != "dataset_sha256"}
    encoded = json.dumps(clean, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_evaluation_samples(path: Path = DATASET) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "evaluation_status": "formal",
        "independent_evaluation": True,
        "used_for_training": False,
        "used_for_threshold_calibration": False,
        "locked": True,
        "sample_count": 60,
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise ValueError(f"Locked privacy benchmark requires {key}={expected!r}.")
    if payload.get("dataset_sha256") != dataset_digest(payload):
        raise ValueError("Privacy benchmark dataset SHA256 does not match its contents.")
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != 60:
        raise ValueError("Privacy cloud-model evaluation requires exactly 60 samples.")
    if any(not isinstance(row, dict) or not all(key in row for key in ("id", "prompt", "ground_truth_route")) for row in samples):
        raise ValueError("Privacy benchmark sample is missing a required field.")
    distribution = dict(Counter(row["ground_truth_route"] for row in samples))
    if distribution != EXPECTED_DISTRIBUTION:
        raise ValueError(f"Unexpected Ground Truth route distribution: {distribution}")
    return payload, samples


def model_map() -> dict:
    models = get_cloud_codegen_evaluation_models()
    if tuple(model.key for model in models) != MODEL_KEYS:
        raise RuntimeError("Privacy evaluation requires exactly gpt4_1, gemini, and claude.")
    return {model.key: model for model in models}


def route_assessment(assessment, *, gater: Callable[[list[float]], dict] = trained_soft_gating_features):
    features = [float(getattr(assessment, name)) for name in FEATURE_NAMES]
    if assessment.blocked_request:
        return features, True, None, "blocked"
    probabilities = gater(features)
    return features, False, probabilities, max(probabilities, key=probabilities.get)


def evaluate_pair(
    sample: dict,
    model_key: str,
    *,
    caller: Callable[..., ModelCallResult] = call_cloud_privacy_evaluation_model,
    gater: Callable[[list[float]], dict] = trained_soft_gating_features,
) -> dict:
    config = model_map()[model_key]
    prompt = str(sample["prompt"])
    messages = build_privacy_assessment_messages(prompt)
    call = caller(model_key, messages, max_tokens=PRIVACY_EVALUATION_MAX_TOKENS)
    normalized_response = normalize_privacy_evaluation_response(call.content) if call.content is not None else None
    base = {
        "sample_id": sample["id"], "prompt": prompt,
        "model_key": model_key, "model_display_name": config.display_name,
        "provider": call.provider, "requested_model": call.requested_model,
        "actual_model": call.actual_model, "input_tokens": call.input_tokens,
        "raw_model_response": call.content,
        "normalized_response": normalized_response,
        "plain_json_compliant": is_plain_json_response(call.content),
        "finish_reason": call.finish_reason,
        "output_tokens": call.output_tokens, "total_tokens": call.total_tokens,
        "latency_seconds": call.latency_seconds,
        "privacy_prompt_rules_version": ASSESSMENT_RULES_VERSION,
        "privacy_prompt_schema_version": ASSESSMENT_SCHEMA_VERSION,
        "privacy_prompt_version": PROMPT_VERSION,
        "gating_model_path": str(DEFAULT_MODEL_PATH.resolve()),
        "gating_feature_names": list(FEATURE_NAMES),
    }
    predicted_route = None
    try:
        if not call.success or not call.content:
            raise RuntimeError(call.error or "Privacy assessment API returned no content.")
        assessment = parse_privacy_assessment_response(
            normalized_response, requested_model=call.requested_model,
            actual_model=call.actual_model, provider=call.provider,
        )
        features, skipped, probabilities, predicted_route = route_assessment(assessment, gater=gater)
        result = {
            **base,
            **{name: features[index] for index, name in enumerate(FEATURE_NAMES)},
            "blocked_request": assessment.blocked_request,
            "analysis_type": assessment.analysis_type,
            "sensitive_categories": assessment.sensitive_categories,
            "explanation": assessment.explanation, "confidence": assessment.confidence,
            "soft_gating_skipped": skipped,
            "soft_gating_probabilities": probabilities,
            "predicted_route": predicted_route,
            "api_call_success": True, "parse_success": True, "error": None,
        }
    except Exception as exc:
        result = {
            **base, **{name: None for name in FEATURE_NAMES},
            "blocked_request": None, "analysis_type": None,
            "sensitive_categories": [], "explanation": None, "confidence": None,
            "soft_gating_skipped": None, "soft_gating_probabilities": None,
            "predicted_route": None,
            "api_call_success": bool(call.success and call.content),
            "parse_success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    # Ground Truth is read only after the independent prediction is complete.
    expected = sample["ground_truth_route"]
    result["ground_truth_route"] = expected
    result["route_correct"] = bool(result["api_call_success"] and predicted_route == expected)
    return result


def _binary_metrics(rows: list[dict]) -> dict:
    tp = sum(row.get("blocked_request") is True and row["ground_truth_route"] == "blocked" for row in rows)
    tn = sum(row.get("blocked_request") is False and row["ground_truth_route"] != "blocked" for row in rows)
    fp = sum(row.get("blocked_request") is True and row["ground_truth_route"] != "blocked" for row in rows)
    fn = sum(row.get("blocked_request") is not True and row["ground_truth_route"] == "blocked" for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": (tp + tn) / 60, "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_blocks": fp, "missed_blocks": fn}


def _multiclass_metrics(rows: list[dict]) -> dict:
    values = []
    for route in ROUTES:
        tp = sum(row.get("predicted_route") == route and row["ground_truth_route"] == route for row in rows)
        fp = sum(row.get("predicted_route") == route and row["ground_truth_route"] != route for row in rows)
        fn = sum(row.get("predicted_route") != route and row["ground_truth_route"] == route for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        values.append((precision, recall, f1))
    return {"balanced_accuracy": statistics.mean(value[1] for value in values),
        "macro_precision": statistics.mean(value[0] for value in values),
        "macro_recall": statistics.mean(value[1] for value in values),
        "macro_f1": statistics.mean(value[2] for value in values)}


def build_results(rows: list[dict], dataset: dict):
    models = model_map(); overall = []; per_route = []; feature_summary = []
    for key in MODEL_KEYS:
        selected = [row for row in rows if row["model_key"] == key]
        api_failures = sum(not bool(row.get("api_call_success")) for row in selected)
        parse_failures = sum(bool(row.get("api_call_success")) and not bool(row.get("parse_success"))
            for row in selected)
        valid_assessments = sum(bool(row.get("api_call_success") and row.get("parse_success"))
            for row in selected)
        has_fixed_denominator = len(selected) == 60
        correct = sum(bool(row.get("route_correct")) for row in selected)
        nonblocked = [row for row in selected if row["ground_truth_route"] != "blocked"]
        nonblocked_correct = sum(bool(row.get("route_correct")) for row in nonblocked)
        multi = _multiclass_metrics(selected); blocked = _binary_metrics(selected)
        overall.append({"model_key": key, "Model": models[key].display_name,
            "Samples": 60, "Completed Responses": len(selected) - api_failures,
            "API Failures": api_failures, "Parse Failures": parse_failures,
            "Valid Privacy Assessments": valid_assessments,
            "Invalid Privacy Assessments": 60 - valid_assessments if has_fixed_denominator else len(selected) - valid_assessments,
            "Exact Route Correct": correct, "Exact Route Accuracy": correct / 60 if has_fixed_denominator else None,
            "Balanced Accuracy": multi["balanced_accuracy"] if has_fixed_denominator else None,
            "Macro Precision": multi["macro_precision"] if has_fixed_denominator else None,
            "Macro Recall": multi["macro_recall"] if has_fixed_denominator else None,
            "Macro F1": multi["macro_f1"] if has_fixed_denominator else None,
            "Non-Blocked Samples": 50, "Non-Blocked Route Correct": nonblocked_correct,
            "Non-Blocked Route Accuracy": nonblocked_correct / 50 if has_fixed_denominator else None,
            "Blocked Ground Truth Samples": 10,
            "Blocked Classification Accuracy": blocked["accuracy"] if has_fixed_denominator else None,
            "Blocked Precision": blocked["precision"] if has_fixed_denominator else None,
            "Blocked Recall": blocked["recall"] if has_fixed_denominator else None,
            "Blocked F1": blocked["f1"] if has_fixed_denominator else None,
            "False Block Count": blocked["false_blocks"], "Missed Block Count": blocked["missed_blocks"]})
        for route in ROUTES:
            route_rows = [row for row in selected if row["ground_truth_route"] == route]
            route_correct = sum(bool(row.get("route_correct")) for row in route_rows)
            per_route.append({"model_key": key, "Model": models[key].display_name,
                "Ground Truth Route": route, "correct": route_correct,
                "total": EXPECTED_DISTRIBUTION[route],
                "accuracy": route_correct / EXPECTED_DISTRIBUTION[route] if has_fixed_denominator else None})
            valid_feature_rows = [row for row in route_rows if has_valid_feature_vector(row)]
            feature_summary.append({"model": models[key].display_name,
                "ground_truth_route": route, "total_samples": len(route_rows),
                "valid_feature_samples": len(valid_feature_rows),
                **{f"{name}_mean": (statistics.mean(float(row[name]) for row in valid_feature_rows)
                    if valid_feature_rows else None) for name in FEATURE_NAMES}})
    complete = len(rows) == 180 and all(sum(row["model_key"] == key for row in rows) == 60 for key in MODEL_KEYS)
    completed_api_calls = sum(bool(row.get("api_call_success")) for row in rows)
    valid_privacy_assessments = sum(bool(row.get("api_call_success") and row.get("parse_success")) for row in rows)
    report = {"status": "complete" if complete else "incomplete",
        "dataset": str(DATASET), "dataset_sha256": dataset["dataset_sha256"],
        "sample_count_per_model": 60, "model_count": 3,
        "total_model_generations": len(rows), "total_expected_calls": 180,
        "completed_api_calls": completed_api_calls,
        "valid_privacy_assessments": valid_privacy_assessments,
        "invalid_privacy_assessments": len(rows) - valid_privacy_assessments,
        "privacy_prompt_rules_version": ASSESSMENT_RULES_VERSION,
        "privacy_prompt_schema_version": ASSESSMENT_SCHEMA_VERSION,
        "shared_prompt": True, "shared_soft_gating": True, "soft_gating_retrained": False,
        "gating_model_path": str(DEFAULT_MODEL_PATH.resolve()),
        "gating_feature_names": list(FEATURE_NAMES),
        "overall_results": overall, "per_route_results": per_route,
        "feature_summary": feature_summary, "per_sample_results": rows}
    return overall, per_route, feature_summary, report


def has_valid_feature_vector(row: dict) -> bool:
    """True only when all shared gating features are finite numeric values."""
    values = [row.get(name) for name in FEATURE_NAMES]
    if any(isinstance(value, bool) or value is None for value in values):
        return False
    try:
        return all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        return False


def _json_cell(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    fields = columns or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _json_cell(row.get(field)) for field in fields})


def plot_results(overall: list[dict], per_route: list[dict]) -> None:
    from sports import matplotlib_backend as _matplotlib_backend
    import matplotlib.pyplot as plt
    models = model_map(); labels = [models[key].display_name for key in MODEL_KEYS]
    lookup = {row["model_key"]: row for row in overall}
    values = [(lookup[key]["Exact Route Accuracy"] or 0) * 100 for key in MODEL_KEYS]
    fig, axis = plt.subplots(figsize=(8, 5)); bars = axis.bar(labels, values)
    axis.bar_label(bars, labels=[f"{value:.1f}%" for value in values]); axis.set_ylim(0, 100)
    axis.set_ylabel("Exact Route Accuracy (%)"); fig.tight_layout(); fig.savefig(ROUTE_FIGURE, dpi=300); plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 5.5)); width = .25; positions = range(len(ROUTES))
    for index, key in enumerate(MODEL_KEYS):
        route_lookup = {row["Ground Truth Route"]: (row["accuracy"] or 0) * 100
            for row in per_route if row["model_key"] == key}
        axis.bar([position + (index - 1) * width for position in positions],
            [route_lookup[route] for route in ROUTES], width, label=models[key].display_name)
    axis.set_xticks(list(positions), [route.replace("_", " ").title() for route in ROUTES])
    axis.set_ylim(0, 100); axis.set_ylabel("Per-Route Accuracy (%)"); axis.legend()
    fig.tight_layout(); fig.savefig(PER_ROUTE_FIGURE, dpi=300); plt.close(fig)


def save_artifacts(rows: list[dict], dataset: dict) -> dict:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    overall, per_route, feature_summary, report = build_results(rows, dataset)
    write_csv(PREDICTIONS, rows); write_csv(SUMMARY, overall)
    write_csv(PER_ROUTE, per_route); write_csv(FEATURE_SUMMARY, feature_summary)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_results(overall, per_route)
    return report


def checkpoint_rows() -> list[dict]:
    if not CHECKPOINT.exists(): return []
    return [json.loads(line) for line in CHECKPOINT.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_checkpoint_version(rows: list[dict]) -> None:
    if any(row.get("privacy_prompt_version") != PROMPT_VERSION for row in rows):
        raise RuntimeError("Checkpoint Privacy Prompt version differs; use --fresh.")


def append_checkpoint(row: dict) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def latest_rows(rows: list[dict]) -> list[dict]:
    latest = {(row["sample_id"], row["model_key"]): row for row in rows}
    return list(latest.values())


def checkpoint_completed_pairs(rows: list[dict]) -> set[tuple[str, str]]:
    """Every recorded pair is final, including an API or schema failure."""
    return {(row["sample_id"], row["model_key"]) for row in rows}


def percent(value) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def print_report(report: dict) -> None:
    print("=" * 62); print("Privacy Assessor Cloud Model Comparison"); print("=" * 62)
    print("Dataset samples per model: 60\nShared Privacy Prompt: YES\nShared frozen 4D Soft Gating: YES\n")
    print(f"{'Model':24} {'Route Acc':>10} {'Nonblocked Acc':>16} {'Block Acc':>11}")
    for row in report["overall_results"]:
        print(f"{row['Model']:24} {percent(row['Exact Route Accuracy']):>10} "
            f"{percent(row['Non-Blocked Route Accuracy']):>16} {percent(row['Blocked Classification Accuracy']):>11}")
    print("\n" + "-" * 62); print("Per Route Accuracy"); print("-" * 62)
    print(f"{'Model':24} {'Cloud':>8} {'Collaboration':>16} {'Local Edge':>12} {'Blocked':>9}")
    for overall in report["overall_results"]:
        values = {row["Ground Truth Route"]: row["accuracy"] for row in report["per_route_results"]
            if row["model_key"] == overall["model_key"]}
        print(f"{overall['Model']:24} {percent(values['cloud']):>8} {percent(values['collaboration']):>16} "
            f"{percent(values['local_edge']):>12} {percent(values['blocked']):>9}")
    print(f"\nStatus: {report['status']}")


def check_only(keys: list[str]) -> int:
    failed = False
    messages = build_privacy_assessment_messages("Assess this general explanation request.")
    for key in keys:
        runtime = get_cloud_codegen_evaluation_runtime(key)
        if not runtime["api_key_loaded"]:
            print(f"{runtime['display_name']}: FAIL (missing {runtime['api_key_env']})"); failed = True; continue
        call = call_cloud_privacy_evaluation_model(
            key, messages, max_tokens=PRIVACY_EVALUATION_MAX_TOKENS
        )
        print(f"{runtime['display_name']}: {'PASS' if call.success else 'FAIL'} ({call.actual_model or call.error})")
        failed |= not call.success
    return int(failed)


def smoke(keys: list[str]) -> int:
    _, samples = load_evaluation_samples(); sample = samples[0]; failed = False
    for key in keys:
        row = evaluate_pair(sample, key); failed |= not row["parse_success"]
        print("-" * 60)
        print(f"Model: {row['model_display_name']}")
        print(f"Actual model: {row['actual_model']}")
        print(f"API call success: {row['api_call_success']}")
        print(f"Finish reason: {row['finish_reason']}")
        print(f"Input tokens: {row['input_tokens']}")
        print(f"Output tokens: {row['output_tokens']}")
        print(f"Total tokens: {row['total_tokens']}")
        print("Raw response:")
        print(row["raw_model_response"] if row["raw_model_response"] is not None else "None")
        print("\nNormalized response:")
        print(row["normalized_response"] if row["normalized_response"] is not None else "None")
        print(f"\nPlain JSON compliant: {row['plain_json_compliant']}")
        print(f"Parse success: {row['parse_success']}")
        print("Parser error:")
        print(row["error"] if row["error"] is not None else "None")
        if row["parse_success"]:
            print("Features: " + repr([row[name] for name in FEATURE_NAMES]))
            print(f"Blocked request: {row['blocked_request']}")
            print(f"Soft-gating probabilities: {row['soft_gating_probabilities']}")
            print(f"Predicted route: {row['predicted_route']}")
            print(f"Ground-truth route: {row['ground_truth_route']}")
            print(f"Result: {'correct' if row['route_correct'] else 'incorrect'}")
        print("-" * 60)
    return int(failed)


def run_formal(keys: list[str], *, fresh: bool, resume: bool) -> dict:
    dataset, samples = load_evaluation_samples(); ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if fresh and CHECKPOINT.exists(): CHECKPOINT.unlink()
    if CHECKPOINT.exists() and not fresh and not resume:
        raise RuntimeError("Checkpoint exists. Use --resume or --fresh.")
    prior = checkpoint_rows() if resume else []
    if resume: validate_checkpoint_version(prior)
    completed_pairs = checkpoint_completed_pairs(prior)
    rows = list(prior)
    for key in keys:
        for sample in samples:
            if (sample["id"], key) in completed_pairs: continue
            row = evaluate_pair(sample, key); append_checkpoint(row); rows.append(row)
    report = save_artifacts(latest_rows(rows), dataset); print_report(report); return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true"); mode.add_argument("--smoke", action="store_true")
    lifecycle = parser.add_mutually_exclusive_group()
    lifecycle.add_argument("--fresh", action="store_true"); lifecycle.add_argument("--resume", action="store_true")
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=list(MODEL_KEYS))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv); load_local_env()
    if args.check_only: return check_only(args.models)
    if args.smoke: return smoke(args.models)
    run_formal(args.models, fresh=args.fresh, resume=args.resume); return 0


if __name__ == "__main__":
    raise SystemExit(main())
