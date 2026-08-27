"""Shared helpers for the frontend benchmark and restricted code-generation evaluations."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

SHARED_BENCHMARK_SAMPLES = 60
ELIGIBLE_LLM_SAMPLES = 40
ELIGIBLE_ROUTES = {"cloud", "collaboration"}
ELIGIBLE_ROUTE_DISTRIBUTION = {"cloud": 5, "collaboration": 35}


def safe_error(value: object) -> str:
    text = str(value or "Unknown provider error")
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED_API_KEY]", text)
    text = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(api[_ -]?key[=: ]+)[^\s,;]+", r"\1[REDACTED]", text)
    return text[:800]


def _dataset_digest(payload: dict) -> str:
    clean = {key: value for key, value in payload.items() if key != "dataset_sha256"}
    encoded = json.dumps(
        clean, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_frontend_benchmark(dataset_path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != SHARED_BENCHMARK_SAMPLES:
        raise ValueError("Shared frontend benchmark must contain exactly 60 samples.")
    if payload.get("dataset_sha256") != _dataset_digest(payload):
        raise ValueError("Shared frontend benchmark digest does not match its contents.")
    if payload.get("used_for_training") is not False:
        raise ValueError("Shared frontend benchmark must not be used for training.")
    if payload.get("used_for_threshold_calibration") is not False:
        raise ValueError("Shared frontend benchmark must not be used for threshold calibration.")
    if payload.get("independent_evaluation") is not True:
        raise ValueError("Shared frontend benchmark must be an independent evaluation.")
    eligible = [sample for sample in samples if sample.get("llm_router_eligible") is True]
    if len(eligible) != ELIGIBLE_LLM_SAMPLES:
        raise ValueError("Shared frontend benchmark must contain exactly 40 LLM-router-eligible samples.")
    mapped = []
    prompts = []
    for index, sample in enumerate(eligible):
        missing = [field for field in ("id", "requested_analysis", "ground_truth_route") if field not in sample]
        prompt = sample.get("prompt") or sample.get("question")
        if missing or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Eligible sample {index} is missing required fields: {missing or ['prompt']}")
        privacy_route = str(sample["ground_truth_route"]).strip().casefold()
        if privacy_route not in ELIGIBLE_ROUTES:
            raise ValueError(f"Eligible sample {sample['id']} has invalid privacy route {privacy_route!r}.")
        mapped.append({
            **sample, "prompt": prompt, "privacy_route": privacy_route,
            "filters": sample.get("analysis_filters", {}),
            "difficulty": sample.get("difficulty", "realistic"),
            "requires_code": sample.get("requires_code", True),
        })
        prompts.append(" ".join(prompt.casefold().split()))
    if len(prompts) != len(set(prompts)):
        raise ValueError("Eligible frontend benchmark prompts must be unique.")
    distribution = {
        route: sum(row["privacy_route"] == route for row in mapped)
        for route in ("cloud", "collaboration")
    }
    if distribution != ELIGIBLE_ROUTE_DISTRIBUTION:
        raise ValueError(f"Unexpected eligible privacy-route distribution: {distribution}")
    return {
        "num_samples": ELIGIBLE_LLM_SAMPLES,
        "dataset_name": "Frontend-Realistic LLM Evaluation",
        "version": payload.get("schema_version"),
        "dataset_path": "evaluation/frontend_realistic_benchmark_60.json",
        "dataset_sha256": payload.get("dataset_sha256"),
        "shared_benchmark_samples": SHARED_BENCHMARK_SAMPLES,
        "eligible_llm_samples": ELIGIBLE_LLM_SAMPLES,
        "route_distribution": distribution,
    }, mapped
