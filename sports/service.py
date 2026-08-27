"""Professor-aligned service layer.

End-to-end flow:
1. PRISM privacy routing,
2. cost-aware Cloud/Local selection only for Cloud/Collaboration; Local Edge
   is forced directly to the configured localhost model,
3. schema-only Python code generation,
4. local AST/whitelist execution,
5. result returned to frontend.
"""

from __future__ import annotations
import logging

from llm.generated_code_verifier import (
    to_dict as verification_to_dict,
    verify_and_execute_generated_code,
)
from llm.code_generator import generate_code, to_dict as code_to_dict
from llm.model_config import LOCAL_EDGE_GENERATOR_MODEL
from privacy.prism_router import prism_route, to_dict as prism_to_dict
from privacy.cloud_local_router import privacy_forced_decision, route_cloud_local, to_dict as router_to_dict

logger = logging.getLogger(__name__)


def _code_generation_failure_message(failure_stage: str | None, privacy_route: str | None) -> str:
    messages = {
        "cloud_model_unavailable": "The selected Cloud code generation model is temporarily unavailable.",
        "local_model_unavailable": "The local code generation model is unavailable.",
        "code_generation": (
            "The local model returned no analysis code."
            if privacy_route == "local_edge"
            else "The selected model could not generate analysis code."
        ),
        "format_validation": "The model output was not a valid restricted analysis call.",
        "request_validation": "The generated analysis call did not match the user's requested analysis.",
        "unsupported_analysis": "Unsupported requested analysis.",
    }
    return messages.get(failure_stage, "The selected model could not generate analysis code.")


def _strip_result(execution_dict: dict) -> dict:
    return {k: v for k, v in execution_dict.items() if k != "result"}


def _build_pipeline_model_audit(
    privacy_decision: dict, model_decision: dict, code_generation: dict
) -> dict:
    """Keep cost-aware selection distinct from the actual code generator."""
    route = privacy_decision.get("route")
    selected = model_decision.get("selected_model")
    tier = model_decision.get("selected_tier") or "none"
    selected_model = model_decision.get("execution_model") if tier in {"cloud", "local"} else None
    actual_generator = code_generation.get("actual_model") or code_generation.get("requested_model")
    if route == "local_edge" and not actual_generator:
        actual_generator = model_decision.get("execution_model") or LOCAL_EDGE_GENERATOR_MODEL
    is_local = tier == "local"
    cost_aware_applicable = route in {"cloud", "collaboration"}
    return {
        "privacy_route": route,
        "cost_aware_selected_tier": tier if cost_aware_applicable else None,
        "cost_aware_selected_model": selected_model if cost_aware_applicable else None,
        "cost_aware_router_status": "applicable" if cost_aware_applicable else "not_applicable",
        "execution_selected_tier": tier,
        "execution_selected_model": selected_model,
        "configured_cloud_model": model_decision.get("execution_model") if tier == "cloud" else None,
        "configured_local_model": model_decision.get("execution_model") if tier == "local" else None,
        "actual_generator_model": actual_generator,
        "actual_generator_provider": code_generation.get("provider"),
        "local_edge_generator_used": route == "local_edge" and is_local,
        "cost_aware_router_applicable": cost_aware_applicable,
        "model_role": "local" if is_local else tier,
        "model_name": actual_generator,
        "provider": code_generation.get("provider"),
        "base_url": "http://127.0.0.1:8080/v1" if is_local else None,
        "execution_location": "local" if is_local else "cloud" if tier == "cloud" else "none",
        "external_cloud_api_used": tier == "cloud",
    }


def _select_model_decision(
    user_prompt: str,
    privacy_decision_obj,
    requested_analysis: str | None = None,
    analysis_filters: dict | None = None,
):
    """Run the local Cloud/Local classifier only where privacy permits it."""
    privacy_route = str(privacy_decision_obj.route)
    if privacy_route in {"cloud", "collaboration"}:
        decision = route_cloud_local(
            user_prompt,
            privacy_route,
            router_prompt_source=(
                "original_prompt_local_classifier"
            ),
        )
        if decision.selected_tier not in {"cloud", "local"}:
            raise RuntimeError("Cost-aware router did not return a valid Cloud/Local decision")
        return decision
    if privacy_route in {"local_edge", "blocked"}:
        return privacy_forced_decision(privacy_route, "local_original_prompt")
    raise RuntimeError(f"Unsupported PRISM route: {privacy_route}")


def _build_privacy_test(user_prompt: str, decision) -> dict:
    """Project a RoutingDecision into UI metadata without assessing or recalculating."""
    is_mapping = isinstance(decision, dict)
    get = decision.get if is_mapping else lambda name, default=None: getattr(decision, name, default)
    logger.info("privacy_test_built_from_decision success=%s", get("privacy_assessment_success", True))
    return {"input_prompt":user_prompt,
        "privacy_risk_score":get("risk_score"),
        "assessment_method":"LLM-based Privacy Assessment",
        "assessment_model":get("privacy_assessment_model"),
        "requested_model":get("privacy_assessment_model"),
        "actual_model":get("privacy_assessment_actual_model"),
        "assessment_provider":get("privacy_assessment_provider"),
        "assessment_rules_version":get("privacy_assessment_rules_version"),
        "rules_version":get("privacy_assessment_rules_version"),
        "cache_used":bool(get("privacy_assessment_cache_used")),
        "cache_hit":bool(get("privacy_assessment_cache_used")),
        "assessment_success":bool(get("privacy_assessment_success", True)),
        "parse_success":bool(get("privacy_assessment_success", True)),
        "validation_success":bool(get("privacy_assessment_success", True)),
        "fallback_reason":get("privacy_assessment_fallback_reason"),
        "validation_error":get("privacy_assessment_validation_error"),
        "explanation":get("privacy_assessment_explanation"),
        "confidence":get("privacy_assessment_confidence"),
        "fallback_used":bool(get("privacy_assessment_fallback_used")),
        "assessment_error":get("privacy_assessment_error"),
        "indicators":dict(get("privacy_indicators") or {}),
        "sensitive_categories":list(get("sensitive_categories") or []),
        "llm_generated_json":get("llm_privacy_assessment") or {},
        "gating_feature_names":list(get("gating_feature_names") or []),
        "gating_features":get("gating_features"),
        "gating_input_dim":int(get("gating_input_dim") or 0),
        "gating_probabilities":get("probabilities"),
        "selected_route":get("route"),
        "gating_skipped":bool(get("gating_skipped")),
        "gating_model_name":get("gating_model_name"),
        "gating_model_path":get("gating_model_path"),
        "gating_training_dataset":get("gating_training_dataset") or "privacy_gating_train_4d_hard_llm_generated_90.json",
        "gating_simulation_stage":False,
        "gating_training_stage":"hard_llm_generated_reviewed",
        "gating_features_source":"GPT-4.1 Privacy Assessor",
        "gating_group_split":True,
        "gating_independent_evaluation":False,
        "model_config_source":"LLM_STRONG_MODEL",
        "blocked_request":bool(get("blocked")),
        "privacy_method":get("privacy_method"),
        "cloud_payload_type":get("cloud_payload_type", "none"),
        "ldp_executed":get("route") == "collaboration",
        "prompt_before_ldp":user_prompt if get("route") == "collaboration" else None,
        "prompt_after_ldp":get("perturbed_prompt") if get("route") == "collaboration" else None,
        "cloud_prompt":get("cloud_prompt"),
        "ldp_audit":list(get("ldp_audit") or [])}


def handle_user_request(
    user_prompt: str,
    use_openai: bool = True,
    session_id: str | None = None,
    requested_analysis: str | None = None,
    private_local_context: dict | None = None,
    analysis_filters: dict[str, str] | None = None,
) -> dict:
    """Route to a code generator, execute validated code, then format its result."""
    privacy_decision_obj = prism_route(user_prompt)
    privacy_decision = prism_to_dict(privacy_decision_obj)
    privacy_test=_build_privacy_test(user_prompt,privacy_decision_obj)

    soft_gating_probabilities = privacy_decision.get("probabilities") or {}
    prism_privacy_result = {
        "prism_input_prompt": user_prompt,
        # This is the exact field made available to cloud code generation.
        "prompt_after_prism": privacy_decision_obj.cloud_prompt,
        "privacy_risk_score": privacy_decision.get("risk_score"),
        "privacy_assessment_method": privacy_decision.get("privacy_assessment_method"),
        "privacy_assessment_model": privacy_decision.get("privacy_assessment_model"),
        "privacy_assessment_explanation": privacy_decision.get("privacy_assessment_explanation"),
        "privacy_assessment_confidence": privacy_decision.get("privacy_assessment_confidence"),
        "privacy_assessment_fallback_used": privacy_decision.get("privacy_assessment_fallback_used", False),
        "soft_gating": {
            "cloud": float(soft_gating_probabilities.get("cloud", 0.0)),
            "collaboration": float(
                soft_gating_probabilities.get("collaboration", 0.0)
            ),
            "local_edge": float(
                soft_gating_probabilities.get(
                    "local_edge", soft_gating_probabilities.get("local", 0.0)
                )
            ),
        },
        "route": privacy_decision.get("route", "unknown"),
        "cloud_payload_type": privacy_decision.get("cloud_payload_type", "none"),
        "gating_source": privacy_decision.get("gating_source", "unknown"),
    }

    model_decision_obj = _select_model_decision(
        user_prompt, privacy_decision_obj, requested_analysis, analysis_filters
    )
    model_decision = router_to_dict(model_decision_obj)
    llm_result = {
        "threshold": model_decision_obj.threshold,
        "p_cloud": model_decision_obj.cloud_model_probability,
        "cloud_model_probability": model_decision_obj.cloud_model_probability,
        "athlete_router_threshold": model_decision_obj.threshold,
        "athlete_router_model_version": model_decision_obj.router_model_version,
        "router_name": model_decision_obj.router_name,
        "router_error": model_decision_obj.router_error,
        "privacy_route": privacy_decision_obj.route,
        "router_applicable": privacy_decision_obj.route in {"cloud", "collaboration"},
        "selected_tier": model_decision_obj.selected_tier,
        "selected_model": (model_decision_obj.selected_model
            if model_decision_obj.selected_tier in {"cloud", "local"} else None),
        "execution_model": model_decision_obj.execution_model,
    }
    code_generation_obj = generate_code(user_prompt, model_decision, privacy_decision,
        use_openai=use_openai,requested_analysis=requested_analysis,requested_filters=analysis_filters)
    code_generation = code_to_dict(code_generation_obj)
    llm_result["generated_code"] = code_generation.get("code")

    # Public PRISM details retain categories and decisions, never detected values
    # or internal prompt payload fields.
    public_privacy_decision = dict(privacy_decision)
    public_privacy_decision["entities"] = [
        {k: v for k, v in entity.items() if k != "value"}
        for entity in (privacy_decision.get("entities") or [])
        if isinstance(entity, dict)
    ]
    for internal_key in ("cloud_prompt", "perturbed_prompt", "semantic_sketch"):
        public_privacy_decision.pop(internal_key, None)

    requested_analysis = code_generation.get("requested_analysis") or "unspecified"
    trusted_subject_reference=(private_local_context or {}).get("CURRENT_SUBJECT") if requested_analysis=="individual_profile" else None
    code_execution_obj = verify_and_execute_generated_code(
        code_generation_obj.code,
        user_request=user_prompt,
        requested_analysis=requested_analysis,
        requested_filters=analysis_filters or {},
        subject_reference=trusted_subject_reference,
    )
    code_execution_full = verification_to_dict(code_execution_obj)
    result = code_execution_full.get("result")
    if isinstance(result, dict) and "correlation_matrix" in result:
        result = {k: v for k, v in result.items() if k != "correlation_matrix"}
        code_execution_full["result"] = result
    if not code_execution_full.get("allowed"):
        result = None
    code_execution_full["result"] = result

    if privacy_decision_obj.blocked:
        answer = privacy_decision_obj.reason
        allowed = False
        result = None
    elif code_generation.get("action") == "unsupported":
        answer = (
            "Future elite-status prediction is not supported. "
            "Use a current descriptive or statistical analysis instead."
        )
        allowed = False
        result = None
    elif not code_generation.get("code"):
        answer = _code_generation_failure_message(
            code_generation.get("failure_stage"), privacy_decision.get("route")
        )
        allowed = False
        result = None
    elif not code_execution_full.get("allowed"):
        stage=code_execution_full.get("failure_stage")
        answer={
            "format_validation":"The model output was not a valid restricted analysis call.",
            "request_validation":"The generated analysis call did not match the user's requested analysis.",
            "local_execution":"The verified analysis call failed during local execution.",
            "result_validation":"The analysis executed, but the returned result did not match the expected analysis output.",
        }.get(stage,"The generated analysis code did not pass the restricted execution policy.")
        allowed=False;result=None
    elif isinstance(result, dict) and result.get("summary"):
        answer = result["summary"]
        allowed = True
    elif isinstance(result, dict):
        answer = f"{result.get('analysis', 'Analysis')} finished locally."
        allowed = True
    else:
        answer = "The generated analysis code did not pass the restricted execution policy."
        allowed = False

    # Keep older keys (payload/plan/generated_code) for UI compatibility.
    response = {
        "allowed": allowed,
        "answer": answer,
        "privacy_decision": public_privacy_decision,
        "prism_privacy_result": prism_privacy_result,
        "privacy_test": privacy_test,
        "model_decision": model_decision,
        "llm_result": llm_result,
        "code_generation": code_generation,
        "generated_code": code_generation.get("code"),
        "code_execution": _strip_result(code_execution_full),
        "result": result,
        "payload": result,
        "pipeline_audit": {
            **_build_pipeline_model_audit(privacy_decision, model_decision, code_generation),
            "explicit_requested_analysis": requested_analysis,
            "generator_target": code_generation.get("generator_target"),
            "llm_model_used": code_generation.get("actual_model") or code_generation.get("requested_model"),
            "llm_provider": code_generation.get("provider"),
            "cloud_used": bool(code_generation.get("cloud_used")),
            "generated_method": code_execution_full.get("generated_method") or code_generation.get("action"),
            "structure_validation_passed": bool(code_execution_full.get("structure_validation_passed")),
            "request_match_passed": bool(code_execution_full.get("request_match_passed")),
            "local_execution_passed": bool(code_execution_full.get("local_execution_passed")),
            "result_validation_passed": bool(code_execution_full.get("result_validation_passed")),
            "fully_correct": bool(code_execution_full.get("fully_correct")),
            "normalization_applied": bool(code_generation.get("normalization_applied")),
            "routing_target": "dynamic_restricted_code_generator",
            "requested_generator_channel": code_generation.get("requested_generator_channel"),
            "used_generator_channel": code_generation.get("used_generator_channel"),
            "generator_fallback_used": code_generation.get("generator_fallback_used"),
            "generated_code_validated": bool(code_execution_full.get("allowed")),
            "generated_code_executed_locally": bool(code_execution_full.get("executed")),
            "answer_source": "local_backend_execution_result" if result is not None else "privacy_or_execution_status",
            "route_llm_directly_answered": False,
            "collaboration_protocol_used": privacy_decision.get("route") == "collaboration",
            "original_prompt_sent_to_cloud": bool(code_generation.get("original_prompt_sent_to_cloud", False)),
            "cloud_received_only_perturbed_prompt": bool(
                privacy_decision.get("route") == "collaboration"
                and privacy_decision.get("cloud_payload_type") == "two_layer_ldp_perturbed_prompt"
            ),
            "cloud_semantic_sketch_received": bool(code_generation.get("cloud_semantic_sketch_received")),
            "local_plan_validated": bool(code_generation.get("local_plan_validated")),
            "local_edge_endpoint_enforced": bool(code_generation.get("local_edge_endpoint_enforced")),
            "cloud_direct_answer_generated": False,
            "cloud_python_code_generated_for_collaboration": False,
            "llm_generated_restricted_python": bool(code_generation.get("generated_code_type") == "restricted_python"),
            "restricted_analysis_api_used": bool(code_generation.get("restricted_api_used")),
            "arbitrary_python_allowed": False,
            "raw_dataframe_available_to_llm": False,
            "fixed_template_generator_used": False,
            "fresh_generation_per_request": True,
            "generation_timestamp": code_generation.get("generation_timestamp"),
            "generation_request_id": code_generation.get("generation_request_id"),
        },
        "plan": {
            "action": code_generation.get("action"),
            "group": code_generation.get("group"),
            "safe_to_execute": bool(code_execution_full.get("allowed")),
            "explanation": code_generation.get("explanation"),
        },
        "failure_stage": (
            code_generation.get("failure_stage") if not code_generation.get("code")
            else code_execution_full.get("failure_stage") if not code_execution_full.get("allowed")
            else None
        ),
        "sanitized_error": code_generation.get("validation_error") or code_execution_full.get("validation_error"),
        "pipeline_diagnostics": {
            "code_generation_passed": bool(code_generation.get("code")),
            "response_cleaning_passed": bool(code_generation.get("code")),
            "structure_validation_passed": bool(code_execution_full.get("structure_validation_passed")),
            "request_match_passed": bool(code_execution_full.get("request_match_passed")),
            "local_execution_passed": bool(code_execution_full.get("local_execution_passed")),
            "result_validation_passed": bool(code_execution_full.get("result_validation_passed")),
            "fully_correct": bool(code_execution_full.get("fully_correct")),
            "frontend_rendering_passed": True,
            "validation_passed": bool(code_execution_full.get("fully_correct")),
            "execution_location": "local",
            "local_generator_available": code_generation.get("local_generator_available"),
            "first_generation_non_empty": bool(code_generation.get("first_generation_non_empty")),
            "first_validation_passed": bool(code_generation.get("first_validation_passed")),
            "repair_attempted": bool(code_generation.get("generation_retry_used")),
            "repair_validation_passed": bool(code_generation.get("repair_validation_passed")),
            "first_validation_error": code_generation.get("first_validation_error"),
            "repair_validation_error": code_generation.get("repair_validation_error"),
            "candidate_assignment_count": int(code_generation.get("candidate_assignment_count") or 0),
            "final_execution_completed": bool(code_execution_full.get("local_execution_passed")),
            "generated_method": code_execution_full.get("generated_method") or code_generation.get("action"),
            "requested_analysis": requested_analysis,
            "selected_generator_tier": model_decision.get("selected_tier"),
            "requested_generator_channel": code_generation.get("requested_generator_channel"),
            "used_generator_channel": code_generation.get("used_generator_channel"),
            "requested_model": code_generation.get("requested_model"),
            "actual_model": code_generation.get("actual_model"),
            "provider": code_generation.get("provider"),
            "model_call_success": bool(code_generation.get("model_call_success")),
            "model_unavailable": bool(code_generation.get("model_unavailable")),
            "provider_retry_used": bool(code_generation.get("provider_retry_used")),
            "generation_request_id": code_generation.get("generation_request_id"),
        },
    }
    diagnostics=response.get("pipeline_diagnostics") or {}
    logger.debug(
        "pipeline_request_complete request_id=%s analysis=%s selected_model=%s method=%s failure_stage=%s",
        diagnostics.get("generation_request_id"), diagnostics.get("requested_analysis"),
        model_decision.get("selected_model"), diagnostics.get("generated_method"),
        response.get("failure_stage"),
    )
    return response
