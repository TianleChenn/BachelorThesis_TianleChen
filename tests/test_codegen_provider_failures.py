from unittest.mock import patch

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


def test_cloud_unavailable_is_not_labeled_as_local():
    unavailable = _call(None, success=False, unavailable=True, error="APIConnectionError: sanitized")
    with patch("llm.code_generator.call_gemini_cloud_model", return_value=unavailable) as cloud, \
         patch("llm.code_generator.call_local_codegen_model") as local:
        result = generate_code(
            FIGURE1_REQUEST,
            CLOUD_DECISION,
            CLOUD_PRIVACY,
            requested_analysis="figure1",
        )

    assert result.code is None
    assert result.failure_stage == "cloud_model_unavailable"
    assert result.failure_stage != "local_model_unavailable"
    assert result.provider_retry_used is True
    assert cloud.call_count == 2
    local.assert_not_called()


def test_local_unavailable_keeps_local_failure_stage():
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


def test_transient_cloud_failure_retries_gemini_without_local_fallback():
    responses = [
        _call(None, success=False, unavailable=True, error="HTTP 503: sanitized"),
        _call(FIGURE1_CODE, success=True, unavailable=False),
    ]
    with patch("llm.code_generator.call_gemini_cloud_model", side_effect=responses) as cloud, \
         patch("llm.code_generator.call_local_codegen_model") as local:
        result = generate_code(
            FIGURE1_REQUEST,
            CLOUD_DECISION,
            CLOUD_PRIVACY,
            requested_analysis="figure1",
        )

    assert cloud.call_count == 2
    assert cloud.call_args_list[0].args[0] == cloud.call_args_list[1].args[0]
    local.assert_not_called()
    assert result.code == FIGURE1_CODE
    assert result.failure_stage is None
    assert result.provider_retry_used is True
    assert result.generation_retry_used is False
    assert result.generator_fallback_used is False


def test_permanent_cloud_failure_retries_once_and_never_calls_local():
    unavailable = _call(None, success=False, unavailable=True, error="HTTP 503: sanitized")
    with patch("llm.code_generator.call_gemini_cloud_model", return_value=unavailable) as cloud, \
         patch("llm.code_generator.call_local_codegen_model") as local:
        result = generate_code(
            FIGURE1_REQUEST,
            CLOUD_DECISION,
            CLOUD_PRIVACY,
            requested_analysis="figure1",
        )

    assert cloud.call_count == 2
    local.assert_not_called()
    assert result.code is None
    assert result.failure_stage == "cloud_model_unavailable"
    assert result.provider_retry_used is True
    assert result.generation_retry_used is False
    assert result.generator_fallback_used is False


def test_non_transient_cloud_failure_is_not_transport_retried():
    bad_request = _call(None, success=False, unavailable=False, error="HTTP 400: sanitized")
    with patch("llm.code_generator.call_gemini_cloud_model", return_value=bad_request) as cloud, \
         patch("llm.code_generator.call_local_codegen_model") as local:
        result = generate_code(
            FIGURE1_REQUEST,
            CLOUD_DECISION,
            CLOUD_PRIVACY,
            requested_analysis="figure1",
        )

    cloud.assert_called_once()
    local.assert_not_called()
    assert result.code is None
    assert result.failure_stage == "code_generation"
    assert result.provider_retry_used is False


def test_failure_messages_keep_cloud_and_local_providers_distinct():
    assert _code_generation_failure_message("cloud_model_unavailable", "cloud") == (
        "The selected Cloud code generation model is temporarily unavailable."
    )
    assert _code_generation_failure_message("local_model_unavailable", "local_edge") == (
        "The local code generation model is unavailable."
    )
