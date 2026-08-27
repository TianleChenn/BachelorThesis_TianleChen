from unittest.mock import patch

import pytest

from llm.analysis_request_contracts import build_request_contract, render_request_contract
from llm.code_generator import generate_code
from llm.model_clients import ModelCallResult
from sports.service import _code_generation_failure_message


FIGURE1_REQUEST = (
    "Generate Figure 1 for all athletes using all eight public domains, "
    "expertise_value as target, elite_status as the analysis grouping field, "
    "correlation threshold 0.15, and 1000 comparison-group variance samples."
)
FIGURE1_CODE = render_request_contract(build_request_contract("figure1"))
INVALID_FIGURE1_CODE = FIGURE1_CODE.replace("analysis.figure1(", "analysis.figure2(")
CLOUD_DECISION = {"selected_model": "cloud_gemini", "selected_tier": "cloud"}
CLOUD_PRIVACY = {"route": "cloud", "blocked": False}


def _call(
    content: str | None,
    *,
    success: bool,
    unavailable: bool,
    error: str | None = None,
    requested_model: str = "gemini-3.5-flash",
) -> ModelCallResult:
    return ModelCallResult(
        content=content,
        requested_model=requested_model,
        actual_model=requested_model if success else None,
        provider="openai_compatible",
        success=success,
        unavailable=unavailable,
        fallback_used=False,
        error=error,
        endpoint="chat.completions",
    )


def _generate_cloud():
    return generate_code(
        FIGURE1_REQUEST,
        CLOUD_DECISION,
        CLOUD_PRIVACY,
        requested_analysis="figure1",
    )


def test_gemini_success_on_first_attempt_never_calls_local():
    success = _call(FIGURE1_CODE, success=True, unavailable=False)
    with patch("llm.code_generator.call_gemini_cloud_model", return_value=success) as cloud, \
         patch("llm.code_generator.call_local_codegen_model") as local:
        result = _generate_cloud()

    cloud.assert_called_once()
    local.assert_not_called()
    assert result.code == FIGURE1_CODE
    assert result.requested_generator_channel == "cloud_restricted_code_generator"
    assert result.used_generator_channel == "cloud_restricted_code_generator"
    assert result.generator_fallback_used is False
    assert result.provider_retry_used is False


def test_gemini_503_then_success_retries_without_local_fallback():
    responses = [
        _call(None, success=False, unavailable=True, error="HTTP 503: sanitized"),
        _call(FIGURE1_CODE, success=True, unavailable=False),
    ]
    with patch("llm.code_generator.call_gemini_cloud_model", side_effect=responses) as cloud, \
         patch("llm.code_generator.call_local_codegen_model") as local:
        result = _generate_cloud()

    assert cloud.call_count == 2
    assert cloud.call_args_list[0].args[0] == cloud.call_args_list[1].args[0]
    local.assert_not_called()
    assert result.code == FIGURE1_CODE
    assert result.provider_retry_used is True
    assert result.generator_fallback_used is False
    assert result.used_generator_channel == "cloud_restricted_code_generator"


def test_two_gemini_503s_fallback_once_to_successful_local_generator():
    cloud_unavailable = _call(
        None, success=False, unavailable=True, error="HTTP 503: high demand"
    )
    local_success = _call(
        FIGURE1_CODE,
        success=True,
        unavailable=False,
        requested_model="Ministral-3-8B-Local",
    )
    with patch(
        "llm.code_generator.call_gemini_cloud_model", return_value=cloud_unavailable
    ) as cloud, patch(
        "llm.code_generator.call_local_codegen_model", return_value=local_success
    ) as local:
        result = _generate_cloud()

    assert cloud.call_count == 2
    local.assert_called_once()
    assert result.code == FIGURE1_CODE
    assert result.requested_generator_channel == "cloud_restricted_code_generator"
    assert result.used_generator_channel == "local_restricted_code_generator"
    assert result.generator_fallback_used is True
    assert result.generator_target == "local_restricted_generator"
    assert result.cloud_used is True
    assert result.provider_retry_used is True
    assert result.requested_model == "Ministral-3-8B-Local"
    assert result.cloud_provider_failure == "HTTP 503: high demand"
    assert result.local_fallback_failure is None


def test_two_gemini_503s_and_unavailable_local_return_distinct_failures():
    cloud_unavailable = _call(
        None, success=False, unavailable=True, error="HTTP 503: high demand"
    )
    local_unavailable = _call(
        None,
        success=False,
        unavailable=True,
        error="APIConnectionError: local endpoint unavailable",
        requested_model="Ministral-3-8B-Local",
    )
    with patch(
        "llm.code_generator.call_gemini_cloud_model", return_value=cloud_unavailable
    ) as cloud, patch(
        "llm.code_generator.call_local_codegen_model", return_value=local_unavailable
    ) as local:
        result = _generate_cloud()

    assert cloud.call_count == 2
    local.assert_called_once()
    assert result.code is None
    assert result.failure_stage == "local_model_unavailable"
    assert result.requested_generator_channel == "cloud_restricted_code_generator"
    assert result.used_generator_channel == "local_restricted_code_generator"
    assert result.generator_fallback_used is True
    assert result.generator_target == "local_restricted_generator"
    assert result.cloud_used is True
    assert result.provider_retry_used is True
    assert result.cloud_provider_failure == "HTTP 503: high demand"
    assert result.local_fallback_failure == "APIConnectionError: local endpoint unavailable"


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_nonavailability_http_failures_never_call_local(status):
    rejected = _call(
        None,
        success=False,
        unavailable=False,
        error=f"HTTP {status}: sanitized",
    )
    with patch("llm.code_generator.call_gemini_cloud_model", return_value=rejected) as cloud, \
         patch("llm.code_generator.call_local_codegen_model") as local:
        result = _generate_cloud()

    cloud.assert_called_once()
    local.assert_not_called()
    assert result.code is None
    assert result.failure_stage == "code_generation"
    assert result.provider_retry_used is False
    assert result.generator_fallback_used is False


def test_cloud_validation_failure_uses_repair_and_never_switches_to_local():
    responses = [
        _call(INVALID_FIGURE1_CODE, success=True, unavailable=False),
        _call(FIGURE1_CODE, success=True, unavailable=False),
    ]
    with patch("llm.code_generator.call_gemini_cloud_model", side_effect=responses) as cloud, \
         patch("llm.code_generator.call_local_codegen_model") as local:
        result = _generate_cloud()

    assert cloud.call_count == 2
    local.assert_not_called()
    assert result.code == FIGURE1_CODE
    assert result.generation_retry_used is True
    assert result.provider_retry_used is False
    assert result.generator_fallback_used is False
    assert result.used_generator_channel == "cloud_restricted_code_generator"


def test_collaboration_fallback_reuses_only_the_privacy_approved_cloud_prompt():
    cloud_unavailable = _call(
        None, success=False, unavailable=True, error="HTTP 503: high demand"
    )
    local_success = _call(
        FIGURE1_CODE,
        success=True,
        unavailable=False,
        requested_model="Ministral-3-8B-Local",
    )
    privacy = {
        "route": "collaboration",
        "blocked": False,
        "cloud_prompt": "PRIVACY_APPROVED_GENERATION_PROMPT",
    }
    with patch(
        "llm.code_generator.call_gemini_cloud_model", return_value=cloud_unavailable
    ) as cloud, patch(
        "llm.code_generator.call_local_codegen_model", return_value=local_success
    ) as local:
        result = generate_code(
            FIGURE1_REQUEST,
            CLOUD_DECISION,
            privacy,
            requested_analysis="figure1",
        )

    assert result.code == FIGURE1_CODE
    assert result.privacy_applied_to_cloud_prompt is True
    assert cloud.call_args_list[0].args[0] == local.call_args.args[0]
    assert "PRIVACY_APPROVED_GENERATION_PROMPT" in repr(local.call_args.args[0])
    assert FIGURE1_REQUEST not in repr(local.call_args.args[0])


def test_direct_local_unavailable_keeps_existing_local_retry_behavior():
    unavailable = _call(
        None,
        success=False,
        unavailable=True,
        error="APIConnectionError: sanitized",
        requested_model="Ministral-3-8B-Local",
    )
    with patch("llm.code_generator.call_local_codegen_model", return_value=unavailable) as local, \
         patch("llm.code_generator.call_gemini_cloud_model") as cloud:
        result = generate_code(
            "Run a local correlation.",
            {},
            {"route": "local_edge", "blocked": False},
            requested_analysis="correlation",
        )

    assert result.code is None
    assert result.failure_stage == "local_model_unavailable"
    assert result.provider_retry_used is True
    assert local.call_count == 2
    cloud.assert_not_called()


def test_failure_messages_keep_cloud_and_local_providers_distinct():
    assert _code_generation_failure_message("cloud_model_unavailable", "cloud") == (
        "The selected Cloud code generation model is temporarily unavailable."
    )
    assert _code_generation_failure_message("local_model_unavailable", "local_edge") == (
        "The local code generation model is unavailable."
    )
