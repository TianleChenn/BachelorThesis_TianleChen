from pathlib import Path
from unittest.mock import patch

from llm.analysis_request_contracts import build_request_contract,render_request_contract
from llm.code_generation_prompt import build_code_generation_messages
from llm.generated_code_verifier import inspect_generated_code,verify_and_execute_generated_code
from llm.code_generator import _generate
from llm.model_clients import ModelCallResult


CODE=render_request_contract(build_request_contract("individual_profile"))


def test_current_subject_is_the_only_accepted_generated_token():
    assert inspect_generated_code(CODE,user_request="profile",requested_analysis="individual_profile",
        requested_filters={}).request_match_passed
    old=CODE.replace("CURRENT_SUBJECT","LOCAL_ATHLETE_REFERENCE")
    invalid_id=CODE.replace("CURRENT_SUBJECT","Athlete_003")
    assert inspect_generated_code(old,user_request="profile",requested_analysis="individual_profile",
        requested_filters={}).failure_stage=="request_validation"
    assert inspect_generated_code(invalid_id,user_request="profile",requested_analysis="individual_profile",
        requested_filters={}).failure_stage=="format_validation"


def test_local_prompt_requires_current_subject():
    prompt=repr(build_code_generation_messages(
        "Generate a protected profile for CURRENT_SUBJECT."
    ))
    assert "CURRENT_SUBJECT" in prompt
    assert "LOCAL_ATHLETE_REFERENCE" not in prompt


def test_frontend_keeps_real_id_only_in_private_local_context():
    source=Path("frontend.py").read_text(encoding="utf-8")
    assert 'private_local_context={"CURRENT_SUBJECT": subject_reference}' in source
    individual_prompt=source[source.index('prompt = (',source.index('def page_individual')):source.index('response = run_request(',source.index('def page_individual'))]
    assert "CURRENT_SUBJECT" in individual_prompt
    assert "subject_reference" not in individual_prompt
    assert "anonymous" not in individual_prompt.lower()
    assert "locally" not in individual_prompt.lower()
    assert 'for {athlete_id}' not in source


def test_restricted_api_resolves_token_without_exposing_identifier():
    result=verify_and_execute_generated_code(CODE,user_request="profile",
        requested_analysis="individual_profile",subject_reference="Athlete_003")
    assert result.local_execution_passed
    rendered=repr(result.result)
    assert "Athlete_003" not in rendered
    assert "body_weight" not in rendered
    assert "vitamin_b12" not in rendered


def test_old_token_gets_one_local_repair_then_stops():
    old=CODE.replace("CURRENT_SUBJECT","LOCAL_ATHLETE_REFERENCE")
    call=ModelCallResult(old,"qwen","qwen","transformers",True,False,False,None)
    with patch("llm.code_generator._call_for_channel",return_value=call) as model_call:
        generated,_,error,repaired,stage,_=_generate("local","profile",{},"id","individual_profile","qwen")
    assert generated is None and error and repaired is True and stage=="request_validation"
    assert model_call.call_count==2


def test_active_provider_replaces_real_ids_with_current_subject():
    source=Path("llm/local_model_provider.py").read_text(encoding="utf-8")
    assert '"CURRENT_SUBJECT", str(message.get("content") or "")' in source
    assert "requested_analysis" not in source
