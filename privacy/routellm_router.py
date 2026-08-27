"""Privacy-constrained runtime adapter for the project-specific athlete router."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from llm.athlete_strong_weak_router import AthleteStrongWeakRouter, predict_athlete_router
from llm.env import load_local_env
from llm.model_config import LOCAL_EDGE_GENERATOR_MODEL, ROUTELLM_STRONG_MODEL, ROUTELLM_WEAK_MODEL


@dataclass
class RouteLLMDecision:
    selected_model: str
    reason: str
    threshold: float | None
    router_name: str
    router_error: str | None
    router_prompt_source: str
    execution_strong_model: str
    execution_weak_model: str
    privacy_constrained: bool
    selected_tier: str = "none"
    execution_model: str | None = None
    athlete_router_probability: float | None = None
    athlete_router_threshold: float | None = None
    athlete_router_model_version: str | None = None
    estimated_relative_cost: float | None = None
    estimated_cost_saving: float | None = None
    routing_target: str = "code_generator"


def to_dict(obj) -> dict:
    return asdict(obj) if hasattr(obj, "__dataclass_fields__") else dict(obj)


def _local_decision(selected: str, reason: str, source: str) -> RouteLLMDecision:
    load_local_env()
    return RouteLLMDecision(
        selected_model=selected,
        reason=reason,
        threshold=None,
        router_name="not_applicable",
        router_error=None,
        router_prompt_source=source,
        execution_strong_model=os.getenv("LLM_STRONG_MODEL", ROUTELLM_STRONG_MODEL),
        execution_weak_model=os.getenv("LLM_WEAK_MODEL", ROUTELLM_WEAK_MODEL),
        privacy_constrained=True,
        execution_model=(
            os.getenv("LLM_LOCAL_MODEL_ID", LOCAL_EDGE_GENERATOR_MODEL)
            if selected == "local_template_high_privacy"
            else None
        ),
    )


def non_routellm_decision(privacy_route: str, source: str) -> RouteLLMDecision:
    """Describe Local Edge or Blocked execution without invoking the athlete router."""
    if privacy_route == "local_edge":
        return _local_decision(
            "local_template_high_privacy",
            "PRISM kept the request local. Strong/Weak routing was not applicable.",
            source,
        )
    if privacy_route == "blocked":
        return _local_decision(
            "none", "PRISM blocked the request before any model call.", source
        )
    raise ValueError(f"Strong/Weak routing is applicable to PRISM route: {privacy_route}")


def route_model(
    prompt: str,
    privacy_route: str,
    threshold: float | None = None,
    strong_cost: float = 1.0,
    weak_cost: float = 0.15,
    privacy_prompt: str | None = None,
    router_prompt_source: str | None = None,
    requested_analysis: str | None = None,
    difficulty: str | None = None,
    filters: dict | None = None,
    requires_code: bool = True,
    athlete_router: AthleteStrongWeakRouter | None = None,
) -> RouteLLMDecision:
    """Route Cloud/Collaboration requests with the saved text classifier."""
    source = router_prompt_source or (
        "prism_approved_prompt" if privacy_prompt is not None else "original_prompt"
    )
    if privacy_route in {"local_edge", "blocked"}:
        return non_routellm_decision(privacy_route, source)

    load_local_env()
    routing_prompt = privacy_prompt or prompt
    prediction = predict_athlete_router(
        routing_prompt,
        requested_analysis=requested_analysis,
        difficulty=difficulty,
        privacy_route=privacy_route,
        filters=filters or {},
        requires_code=requires_code,
        router_prompt_source=source,
        router=athlete_router,
    )
    calibrated_threshold = float(prediction["threshold"])
    probability = float(prediction["p_strong"])
    common = dict(
        threshold=calibrated_threshold,
        router_name="new_athlete_router",
        router_error=None,
        router_prompt_source=source,
        execution_strong_model=os.getenv("LLM_STRONG_MODEL", ROUTELLM_STRONG_MODEL),
        execution_weak_model=os.getenv("LLM_WEAK_MODEL", ROUTELLM_WEAK_MODEL),
        privacy_constrained=False,
        athlete_router_probability=probability,
        athlete_router_threshold=calibrated_threshold,
        athlete_router_model_version=prediction.get("model_version"),
    )
    reason = (
        "The New Athlete Router estimated that the strong model probability was "
        f"{probability:.6f}. The calibrated threshold was {calibrated_threshold:.6f}."
    )
    if prediction["selected_tier"] == "strong":
        return RouteLLMDecision(
            "strong_gpt4_1106_preview",
            reason,
            estimated_relative_cost=strong_cost,
            estimated_cost_saving=0.0,
            selected_tier="strong",
            execution_model=common["execution_strong_model"],
            **common,
        )
    return RouteLLMDecision(
        "weak_mixtral_8x7b",
        reason,
        estimated_relative_cost=weak_cost,
        estimated_cost_saving=max(0.0, strong_cost - weak_cost),
        selected_tier="weak",
        execution_model=common["execution_weak_model"],
        **common,
    )
