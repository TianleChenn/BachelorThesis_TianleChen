"""Privacy-constrained adapter for the cost-aware Cloud/Local router."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from llm.athlete_cloud_local_router import AthleteCloudLocalRouter
from llm.env import load_local_env


@dataclass
class CloudLocalDecision:
    selected_model: str
    reason: str
    threshold: float | None
    router_name: str
    router_error: str | None
    router_prompt_source: str
    privacy_constrained: bool
    selected_tier: str = "none"
    execution_model: str | None = None
    cloud_model_probability: float | None = None
    router_model_version: str | None = None
    routing_target: str = "code_generator"


def to_dict(obj) -> dict:
    return asdict(obj) if hasattr(obj, "__dataclass_fields__") else dict(obj)


def privacy_forced_decision(privacy_route: str, source: str) -> CloudLocalDecision:
    """Bypass cost-aware classification for Local Edge and Blocked routes."""
    load_local_env()
    if privacy_route == "local_edge":
        return CloudLocalDecision(
            selected_model="local_ministral", selected_tier="local",
            execution_model=os.getenv("LLM_LOCAL_MODEL", "Ministral-3-8B-Local"),
            reason="PRISM requires local processing; cost-aware classification was bypassed.",
            threshold=None, router_name="privacy_forced_local", router_error=None,
            router_prompt_source=source, privacy_constrained=True,
        )
    if privacy_route == "blocked":
        return CloudLocalDecision(
            selected_model="none", selected_tier="none", execution_model=None,
            reason="PRISM blocked the request before any model call.", threshold=None,
            router_name="not_applicable", router_error=None,
            router_prompt_source=source, privacy_constrained=True,
        )
    raise ValueError(f"Cost-aware routing is applicable to PRISM route: {privacy_route}")


def route_cloud_local(
    prompt: str,
    privacy_route: str,
    *,
    router_prompt_source: str = "original_prompt",
    athlete_router: AthleteCloudLocalRouter | None = None,
) -> CloudLocalDecision:
    """Classify the original request locally for Cloud/Collaboration routes."""
    if privacy_route in {"local_edge", "blocked"}:
        return privacy_forced_decision(privacy_route, router_prompt_source)
    if privacy_route not in {"cloud", "collaboration"}:
        raise ValueError(f"Unsupported privacy route: {privacy_route}")
    try:
        prediction = (athlete_router or AthleteCloudLocalRouter()).predict(
            prompt, router_prompt_source=router_prompt_source
        )
    except Exception as exc:
        load_local_env()
        return CloudLocalDecision(
            selected_model="local_ministral", selected_tier="local",
            execution_model=os.getenv("LLM_LOCAL_MODEL", "Ministral-3-8B-Local"),
            reason="Cloud/Local router unavailable; conservatively selected Local.",
            threshold=None, router_name="athlete_cloud_local_router", router_error=f"{type(exc).__name__}: {exc}",
            router_prompt_source=router_prompt_source, privacy_constrained=True,
        )
    probability = float(prediction["p_cloud"])
    threshold = float(prediction["threshold"])
    return CloudLocalDecision(
        selected_model=prediction["selected_model"],
        selected_tier=prediction["selected_tier"],
        execution_model=prediction["execution_model"],
        reason=(f"P_cloud={probability:.6f}; calibrated threshold={threshold:.6f}."),
        threshold=threshold,
        router_name=prediction["router_name"],
        router_error=None,
        router_prompt_source=router_prompt_source,
        privacy_constrained=False,
        cloud_model_probability=probability,
        router_model_version=prediction.get("model_version"),
    )
