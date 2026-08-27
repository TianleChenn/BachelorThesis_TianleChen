from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from frontend import build_llm_selection_label
from llm.analysis_request_contracts import build_request_contract,render_request_contract
from llm.code_generation_prompt import build_code_generation_messages
from llm.generated_code_verifier import inspect_generated_code,verify_and_execute_generated_code
from llm.code_generator import _generate, generate_code
from llm.local_model_provider import LocalModelProvider, get_local_model_status
from llm.model_clients import ModelCallResult


def _call(content, success=True, unavailable=False, error=None):
    return ModelCallResult(content, "local", "local" if success else None, "transformers", success, unavailable, False, error)


def test_local_prompt_uses_the_shared_generic_messages():
    messages=build_code_generation_messages("Calculate Pearson correlations.")
    assert "requested_analysis" not in repr(messages)
    assert "result = analysis.<method>(...)" in messages[0]["content"]


def test_unsupported_analysis_stops_before_model_call():
    with patch("llm.code_generator._call_for_channel") as provider:
        result = generate_code("request", {}, {"route": "local_edge"}, requested_analysis="unknown")
    assert result.failure_stage == "unsupported_analysis"
    provider.assert_not_called()


def test_model_unavailable_and_empty_generation_have_distinct_stages():
    with patch("llm.code_generator._call_for_channel", return_value=_call(None, False, True, "unavailable")):
        assert _generate("local", "request", {}, "id", "correlation", "local")[4] == "local_model_unavailable"
    with patch("llm.code_generator._call_for_channel", return_value=_call(None, False, False, "empty")):
        assert _generate("local", "request", {}, "id", "correlation", "local")[4] == "code_generation"


def test_markdown_code_is_cleaned_and_missing_assignment_is_repaired_once():
    code = render_request_contract(build_request_contract("correlation"))
    calls = iter([_call("analysis.correlation()"), _call(f"```python\n{code}\n```")])
    with patch("llm.code_generator._call_for_channel", side_effect=lambda *_args, **_kwargs: next(calls)) as local:
        generated, _, error, repaired, _, metadata = _generate("local", "request", {}, "id", "correlation", "local")
    assert error is None and repaired is True and generated[0] == code
    assert metadata["first_validation_passed"] is False
    assert metadata["repair_validation_passed"] is True
    assert local.call_count == 2


def test_correlation_repair_uses_assistant_turn_and_exact_verifier_feedback():
    code = render_request_contract(build_request_contract("correlation"))
    invalid = code.replace("method='pearson')", "method='pearson', visualization=False)")
    repaired = code.replace("method='pearson')", "method='pearson', visualization=True)")
    calls = iter([_call(invalid), _call(repaired)])
    with patch("llm.code_generator.call_local_codegen_model", side_effect=lambda *_args, **_kwargs: next(calls)) as local, \
         patch("llm.code_generator.call_gemini_cloud_model") as cloud:
        result = generate_code(
            "Run correlation for the selected cohort using all standardized domains.",
            {}, {"route":"local_edge", "blocked":False}, requested_analysis="correlation",
        )

    assert result.code == repaired
    assert result.generation_retry_used is True
    assert result.first_validation_passed is False
    assert result.repair_validation_passed is True
    assert local.call_count == 2
    cloud.assert_not_called()
    repair_messages = local.call_args.args[0]
    assert [message["role"] for message in repair_messages] == ["system", "user", "assistant", "user"]
    assert repair_messages[2]["content"] == invalid
    assert "Expected visualization=True but generated False." in repair_messages[3]["content"]


def test_provider_request_error_is_not_unavailable_or_a_fallback():
    class BadRequestError(Exception):
        status_code = 400
        code = "bad_request"

    fake_client = Mock()
    fake_client.chat.completions.create.side_effect = BadRequestError("invalid chat roles")
    with patch("openai.OpenAI", return_value=fake_client):
        from llm.model_clients import _call
        result = _call(
            [{"role":"user", "content":"request"}], "local", "openai_compatible",
            "none", "http://127.0.0.1:8080/v1", 0.0, 100,
        )
    assert result.success is False
    assert result.unavailable is False
    assert result.fallback_used is False
    assert "http_status=400" in result.error


def test_two_invalid_generations_never_access_local_data():
    calls = iter([_call("analysis.correlation()"), _call("analysis.correlation()")])
    with patch("llm.code_generator._call_for_channel", side_effect=lambda *_args, **_kwargs: next(calls)):
        generated = _generate("local", "request", {}, "id", "correlation", "local")[0]
    with patch("llm.generated_code_verifier.RestrictedAnalysisAPI") as api:
        execution = verify_and_execute_generated_code(None if generated is None else generated[0],
            user_request="request",requested_analysis="correlation")
    assert execution.executed is False
    api.assert_not_called()


def test_high_risk_never_calls_cloud_models_and_keeps_requested_analysis():
    local = _call(render_request_contract(build_request_contract("correlation")))
    with patch("llm.code_generator.call_local_codegen_model", return_value=local) as local_call, \
         patch("llm.code_generator.call_gemini_cloud_model") as cloud:
        result = generate_code("private correlation", {}, {"route": "local_edge"}, requested_analysis="correlation", requested_filters={})
    assert result.requested_analysis == "correlation" and result.code
    local_call.assert_called_once(); cloud.assert_not_called()


def test_individual_profile_uses_token_and_public_result_hides_ids():
    code = render_request_contract(build_request_contract("individual_profile"))
    assert "subject_token='CURRENT_SUBJECT'" in code
    result = verify_and_execute_generated_code(code,user_request="profile",
        requested_analysis="individual_profile",subject_reference="Athlete_003")
    assert result.executed and "Athlete_003" not in repr(result.result) and "CURRENT_SUBJECT" not in repr(result.result)


def test_pipeline_overview_shows_actual_local_model_and_frontend_guards_success_message():
    assert build_llm_selection_label("local_edge", None, None, "internal-model") == "Local — internal-model"
    source = Path("frontend.py").read_text(encoding="utf-8")
    success = source.index('response["answer"] = "Anonymous standardized profile generated locally."')
    guard = source.rfind('if response.get("allowed") and response.get("result") is not None:', 0, success)
    assert guard != -1


def test_health_status_api_is_structured_without_forcing_a_download():
    status = get_local_model_status(load_model=False)
    assert isinstance(status.available, bool)
    assert status.model_id
    assert hasattr(status, "device") and hasattr(status, "error")
