from unittest.mock import patch

from llm.analysis_request_contracts import build_request_contract,render_request_contract
from llm.generated_code_verifier import inspect_generated_code
from llm.code_generator import _generate
from llm.local_model_provider import LocalModelProvider
from llm.model_clients import ModelCallResult


CODE = render_request_contract(build_request_contract("individual_profile"))


def clean(value):
    return LocalModelProvider.clean_restricted_model_output(value)


def test_only_fence_lines_are_removed():
    assert clean(f"```python\n{CODE}\n```") == CODE
    assert clean("python\n" + CODE) != CODE
    assert clean(CODE.replace("result = ", "", 1)) != CODE
    assert clean(CODE + "\nExplanation") != CODE


def test_unsafe_outputs_are_rejected_by_shared_validator():
    invalid = [CODE.replace("CURRENT_SUBJECT", "Athlete_003"), CODE.replace("individual_profile", "correlation"), CODE + "\n" + CODE]
    assert all(not inspect_generated_code(clean(value),user_request="profile",
        requested_analysis="individual_profile",requested_filters={}).request_match_passed
        for value in invalid)


def test_invalid_first_output_gets_one_repair_then_stops():
    call = ModelCallResult("analysis.individual_profile()", "qwen", "qwen", "transformers", True, False, False, None)
    with patch("llm.code_generator._call_for_channel", return_value=call) as model_call:
        generated, _, error, repaired, stage, _ = _generate("local", "local profile", {}, "id", "individual_profile", "qwen")
    assert generated is None and error and repaired is True and stage == "format_validation"
    assert model_call.call_count == 2
