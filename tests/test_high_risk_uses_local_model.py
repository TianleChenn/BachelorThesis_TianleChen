from unittest.mock import patch

from llm.analysis_request_contracts import build_request_contract, render_request_contract
from llm.code_generator import generate_code
from llm.model_clients import ModelCallResult


def _call(content, success=True):
    return ModelCallResult(content, "Ministral-3-8B-Local", "Ministral-3-8B-Local" if success else None,
                           "openai_compatible", success, not success, False,
                           None if success else "local unavailable")


def test_local_edge_forces_local_and_never_cloud():
    code = render_request_contract(build_request_contract("correlation"))
    with patch("llm.code_generator.call_local_codegen_model", return_value=_call(code)) as local, \
         patch("llm.code_generator.call_gemini_cloud_model") as cloud:
        result = generate_code("safe local correlation", {}, {"route":"local_edge", "blocked":False},
                               requested_analysis="correlation")
    assert result.code and result.provider == "openai_compatible"
    assert result.generator_target == "local_restricted_generator" and result.cloud_used is False
    assert result.original_prompt_sent_to_cloud is False
    assert result.local_edge_endpoint_enforced is True
    assert "safe local correlation" in repr(local.call_args.args[0])
    local.assert_called_once(); cloud.assert_not_called()


def test_local_edge_failure_has_no_cloud_fallback():
    with patch("llm.code_generator.call_local_codegen_model", return_value=_call(None, False)) as local, \
         patch("llm.code_generator.call_gemini_cloud_model") as cloud:
        result = generate_code("safe local correlation", {}, {"route":"local_edge", "blocked":False},
                               requested_analysis="correlation")
    assert result.code is None and result.failure_stage == "local_model_unavailable"
    assert result.cloud_used is False and result.original_prompt_sent_to_cloud is False
    assert result.local_edge_endpoint_enforced is True
    assert local.call_count == 2; cloud.assert_not_called()
