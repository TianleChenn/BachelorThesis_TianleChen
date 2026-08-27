"""Shared objective code-generation evaluation for Cloud and Local models."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Callable

from llm.code_generation_prompt import (
    CODE_GENERATION_PROMPT_VERSION, RESTRICTED_CODE_MAX_TOKENS,
    build_code_generation_messages,
)
from llm.generated_code_verifier import (
    GeneratedCodeVerificationResult,
    clean_generated_code,
    sanitize_exception,
    verify_and_execute_generated_code,
)
from llm.model_clients import ModelCallResult, call_gemini_cloud_model, call_local_codegen_model


def preference_from_correctness(cloud_fully_correct: bool, local_fully_correct: bool) -> str:
    if local_fully_correct:
        return "local"
    if cloud_fully_correct:
        return "cloud"
    return "invalid"


def evaluate_model_candidate(
    sample: dict,
    caller: Callable[..., ModelCallResult],
    *,
    messages: list[dict[str, str]] | None = None,
    max_tokens: int = RESTRICTED_CODE_MAX_TOKENS,
) -> dict:
    """Generate from the shared prompt, then validate only after generation."""
    canonical_messages = messages if messages is not None else build_code_generation_messages(sample["prompt"])
    call = caller(deepcopy(canonical_messages), temperature=0.0, max_tokens=max_tokens)
    raw = call.content if call.success and call.content else None
    generated = clean_generated_code(raw or "") if raw else None
    try:
        verification = verify_and_execute_generated_code(
            generated,
            user_request=sample["prompt"],
            requested_analysis=sample["analysis_type"],
            requested_filters=sample.get("requested_filters") or {},
            close_figures_after_execution=True,
        )
    except Exception as exc:
        # Model/API failures are handled by their callers.  An unexpected
        # candidate validation/execution failure is recorded as an invalid
        # candidate so the resumable collection can continue.
        verification = GeneratedCodeVerificationResult(
            cleaned_code=generated,
            failure_stage="local_execution",
            validation_error=sanitize_exception(exc),
        )
    details = asdict(verification)
    details.pop("result", None)
    return {
        "model": call.requested_model,
        "requested_model": call.requested_model,
        "actual_model": call.actual_model,
        "raw_response": raw,
        "generated_code": generated,
        "fully_correct": bool(verification.fully_correct),
        "validation_details": details,
        "latency_seconds": call.latency_seconds,
        "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens,
        "finish_reason": call.finish_reason,
        "error": call.error,
    }


def evaluate_cloud_and_local(sample: dict) -> dict:
    # Controlled comparison: build once from the original request. No privacy
    # route or cloud_prompt transformation is accepted in this experiment.
    messages = build_code_generation_messages(sample["prompt"])
    cloud = evaluate_model_candidate(sample, call_gemini_cloud_model, messages=messages)
    local = evaluate_model_candidate(sample, call_local_codegen_model, messages=messages)
    return {
        "cloud": cloud,
        "local": local,
        "preference": preference_from_correctness(cloud["fully_correct"], local["fully_correct"]),
        "prompt_version": CODE_GENERATION_PROMPT_VERSION,
    }
