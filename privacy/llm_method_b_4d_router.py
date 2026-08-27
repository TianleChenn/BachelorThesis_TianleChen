"""Independent minimal-LLM plus 4D Soft Gating router for comparison Method B."""

from __future__ import annotations
from dataclasses import dataclass
from privacy.llm_minimal_4d_privacy_assessor import assess_privacy_minimal_4d
from privacy.llm_soft_gating_model import FEATURE_NAMES, build_llm_gating_features
from privacy.prism_router import get_active_prism_gater_path, trained_soft_gating_features


@dataclass
class MethodBDecision:
    route: str | None
    success: bool
    error: str | None
    features: list[float] | None
    probabilities: dict[str, float] | None
    gating_source: str
    assessment_cache_used: bool = False
    blocked_request: bool = False
    blocked: bool = False
    reason: str | None = None


def get_method_b_gater_path():
    """Return Method C's exact active checkpoint path."""
    return get_active_prism_gater_path()


def route_with_method_b_4d_soft_gating(prompt: str) -> MethodBDecision:
    assessment = assess_privacy_minimal_4d(prompt)
    if not assessment.success:
        return MethodBDecision(None, False, assessment.error, None, None, "minimal_4d_assessment_failed", assessment.cache_used)
    features = build_llm_gating_features(assessment)
    if assessment.blocked_request:
        reason = "Blocked by the Method B minimal LLM assessment before Soft Gating."
        return MethodBDecision("blocked", True, None, features, None,
                               "method_b_minimal_llm_blocked_request", assessment.cache_used,
                               blocked_request=True, blocked=True, reason=reason)
    try:
        probabilities = trained_soft_gating_features(features)
        route = max(probabilities, key=probabilities.get)
        return MethodBDecision(route, True, None, features, probabilities,
                               "method_b_minimal_4d_shared_soft_gating", assessment.cache_used,
                               blocked_request=False, blocked=False)
    except Exception as exc:
        return MethodBDecision(None, False, f"{type(exc).__name__}: {exc}", features, None,
                               "method_b_soft_gating_failed", assessment.cache_used)
