"""Prompt variants and prompt-only evaluation helpers for privacy ablation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from llm.model_clients import ModelCallResult, call_privacy_risk_model
from privacy.llm_minimal_4d_privacy_assessor import (
    FEATURE_KEYS,
    MINIMAL_4D_SYSTEM_PROMPT,
    parse_minimal_4d_response,
)
from privacy.llm_privacy_assessor import (
    PRIVACY_ASSESSMENT_SYSTEM_PROMPT,
    parse_privacy_assessment_response,
)
from privacy.llm_soft_gating_model import build_llm_gating_features
from privacy.prism_router import get_active_prism_gater_path, trained_soft_gating_features

PROMPT_VERSIONS = ("minimal", "defined", "full")
PRIVACY_ABLATION_MAX_TOKENS = 500

_FOUR_DIMENSION_DEFINITIONS = """

Definitions of the four privacy dimensions:

Privacy Risk Score:
Represents the overall privacy risk of the request. Low values indicate low privacy risk and high
values indicate high privacy risk.

Subject Scope:
Represents how specifically the request targets athletes. Requests about all athletes should generally
have low values. Requests about a subgroup should have intermediate values. Requests about one specific
athlete should have high values.

Data Sensitivity:
Represents the sensitivity of the requested athlete information. General performance information should
generally have lower values. Medical, genetic, mental-health, blood-related, or similarly sensitive
information should generally have higher values.

Disclosure Level:
Represents how much detailed information the request asks to reveal. Aggregate or derived statistical
results should generally have lower values. Exact measurements, raw records, or direct private values
should generally have higher values.
"""

# P1 and P3 are aliases to the production prompt constants, not rewritten copies.
MINIMAL_PRIVACY_PROMPT = MINIMAL_4D_SYSTEM_PROMPT
DEFINED_4D_PRIVACY_PROMPT = MINIMAL_4D_SYSTEM_PROMPT + _FOUR_DIMENSION_DEFINITIONS
FULL_PRIVACY_PROMPT = PRIVACY_ASSESSMENT_SYSTEM_PROMPT
PRIVACY_PROMPTS = {
    "minimal": MINIMAL_PRIVACY_PROMPT,
    "defined": DEFINED_4D_PRIVACY_PROMPT,
    "full": FULL_PRIVACY_PROMPT,
}


@dataclass(frozen=True)
class AblationPrivacyAssessment:
    privacy_risk_score: float
    subject_scope: float
    data_sensitivity: float
    disclosure_level: float
    blocked_request: bool


def prompt_sha256(prompt_version: str) -> str:
    return hashlib.sha256(PRIVACY_PROMPTS[prompt_version].encode("utf-8")).hexdigest()


def build_privacy_ablation_messages(prompt_version: str, user_request: str) -> list[dict[str, str]]:
    if prompt_version not in PRIVACY_PROMPTS:
        raise ValueError(f"Unknown privacy prompt version: {prompt_version}")
    return [
        {"role": "system", "content": PRIVACY_PROMPTS[prompt_version]},
        {"role": "user", "content": str(user_request).strip()},
    ]


def parse_privacy_ablation_response(prompt_version: str, content: str) -> AblationPrivacyAssessment:
    if prompt_version in {"minimal", "defined"}:
        parsed = parse_minimal_4d_response(content)
    elif prompt_version == "full":
        parsed = parse_privacy_assessment_response(content)
    else:
        raise ValueError(f"Unknown privacy prompt version: {prompt_version}")
    return AblationPrivacyAssessment(
        **{name: float(getattr(parsed, name)) for name in FEATURE_KEYS},
        blocked_request=bool(parsed.blocked_request),
    )


def route_ablation_assessment(assessment: AblationPrivacyAssessment) -> tuple[str, dict | None]:
    """Apply the production blocking override and exact active frozen Soft Gating."""
    if assessment.blocked_request:
        return "blocked", None
    probabilities = trained_soft_gating_features(build_llm_gating_features(assessment))
    return max(probabilities, key=probabilities.get), probabilities


def evaluate_privacy_prompt(
    prompt_version: str,
    user_request: str,
    *,
    caller=call_privacy_risk_model,
) -> dict:
    messages = build_privacy_ablation_messages(prompt_version, user_request)
    call: ModelCallResult = caller(
        messages, temperature=0.0, max_tokens=PRIVACY_ABLATION_MAX_TOKENS
    )
    base = {
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256(prompt_version),
        "requested_model": call.requested_model,
        "actual_model": call.actual_model,
        "provider": call.provider,
        "raw_model_output": call.content,
        "latency_seconds": call.latency_seconds,
        "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens,
        "api_call_success": bool(call.success and call.content),
        "api_error": call.error if not call.success else None,
        "model_error": None,
    }
    if not call.success or not call.content:
        return {**base, "assessment_success": False}
    if call.fallback_used:
        return {**base, "assessment_success": False,
                "model_error": "Model fallback is forbidden in prompt ablation."}
    try:
        assessment = parse_privacy_ablation_response(prompt_version, call.content)
        route, probabilities = route_ablation_assessment(assessment)
    except Exception as exc:
        return {**base, "assessment_success": False,
                "model_error": f"{type(exc).__name__}: {exc}"}
    return {
        **base,
        "assessment_success": True,
        **{name: getattr(assessment, name) for name in FEATURE_KEYS},
        "blocked_request": assessment.blocked_request,
        "soft_gating_probabilities": probabilities,
        "predicted_route": route,
        "gating_model_path": str(get_active_prism_gater_path()),
    }


def prompt_metadata() -> dict:
    return {
        version: {"sha256": prompt_sha256(version), "system_prompt": PRIVACY_PROMPTS[version]}
        for version in PROMPT_VERSIONS
    }
