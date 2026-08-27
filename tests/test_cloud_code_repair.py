from unittest.mock import patch

import pytest

from llm.analysis_request_contracts import build_request_contract,render_request_contract
from llm.code_generator import _generate
from llm.generated_code_verifier import clean_generated_code,extract_restricted_assignment,inspect_generated_code,verify_and_execute_generated_code
from llm.model_clients import ModelCallResult


CORRELATION = render_request_contract(build_request_contract("correlation"))


@pytest.mark.parametrize("response", [
    CORRELATION,
    f"```python\n{CORRELATION}\n```",
    f"Here is the requested restricted call:\n{CORRELATION}\nThis completes the request.",
])
def test_exactly_one_one_or_multiline_candidate_is_safely_extracted(response):
    cleaned,count=extract_restricted_assignment(response)
    assert count==1
    assert cleaned==CORRELATION
    assert inspect_generated_code(cleaned,user_request="correlation",
        requested_analysis="correlation",requested_filters={}).request_match_passed


@pytest.mark.parametrize("extra", [
    CORRELATION,
    "import os",
    "x = 1",
    "# comment",
])
def test_multiple_or_additional_python_content_is_not_extracted(extra):
    response=f"{CORRELATION}\n{extra}"
    assert not inspect_generated_code(clean_generated_code(response),user_request="correlation",
        requested_analysis="correlation",requested_filters={}).request_match_passed


def test_cloud_generation_repairs_once_and_executes_repaired_code():
    calls=iter([
        ModelCallResult("variables = []", "strong", "strong", "test", True, False, False, None),
        ModelCallResult(CORRELATION, "strong", "strong", "test", True, False, False, None),
    ])
    with patch("llm.code_generator._call_for_channel",side_effect=lambda *_args,**_kwargs:next(calls)) as model:
        generated,_,error,repaired,_,metadata=_generate("strong","correlation",{"requested_filters":{}},"id","correlation","strong")
    assert error is None and repaired is True and generated[0]==CORRELATION
    assert metadata["first_validation_error"]
    assert metadata["repair_validation_error"] is None
    assert model.call_count==2


def test_failed_cloud_repair_remains_blocked():
    invalid=ModelCallResult("variables = []", "strong", "strong", "test", True, False, False, None)
    with patch("llm.code_generator._call_for_channel",return_value=invalid) as model:
        generated,_,error,repaired,stage,_=_generate("strong","variance",{"requested_filters":{}},"id","variance_analysis","strong")
    assert generated is None and error and repaired is True and stage=="format_validation"
    assert model.call_count==2


def test_gpt_style_variance_prose_is_extracted_validated_and_executed():
    response=f"Here is the code you requested:\n```python\n{render_request_contract(build_request_contract('variance_analysis'))}\n```\nDone."
    code=clean_generated_code(response)
    assert inspect_generated_code(code,user_request="variance",
        requested_analysis="variance_analysis",requested_filters={}).request_match_passed
    execution=verify_and_execute_generated_code(code,user_request="variance",
        requested_analysis="variance_analysis",requested_filters={})
    assert execution.executed is True
