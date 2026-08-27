from unittest.mock import Mock,patch

from llm.code_generation_pools import ANALYSIS_METHOD_POOL,ALLOWED_VALUE_POOL
from llm.code_generation_prompt import CODE_GENERATION_PROMPT_VERSION,build_code_generation_messages
from llm.code_generator import _generate
from llm.model_clients import ModelCallResult
from sports.config import PREDICTORS
from sports.restricted_analysis_api import ALLOWED_METHODS,METHOD_ARGUMENT_SCHEMAS


REQUEST=("Calculate pairwise Pearson correlations among all eight public domains "
    "for junior national team athletes.")


def test_prompt_v3_typed_schema_instructions_and_version():
    assert CODE_GENERATION_PROMPT_VERSION=="pool_typed_schema_v3"
    system=build_code_generation_messages(REQUEST)[0]["content"]
    assert "every explicit requirement" in system
    assert "use the corresponding allowed value" in system
    for method in ANALYSIS_METHOD_POOL:assert method in system
    for domain in PREDICTORS:assert domain in system
    for forbidden in ("Required contract","requested_analysis","required_filters",
            "Expected method","Expected arguments","request_mismatches"):
        assert forbidden not in system


def test_pool_integrity():
    assert len(ANALYSIS_METHOD_POOL)==7
    assert set(ANALYSIS_METHOD_POOL)==set(ALLOWED_METHODS)==set(METHOD_ARGUMENT_SCHEMAS)
    assert ALLOWED_VALUE_POOL["public_domains"]==list(PREDICTORS)
    assert not {"athlete_id","vitamin_b12","body_weight"} & set(ALLOWED_VALUE_POOL["public_domains"])
    assert ALLOWED_VALUE_POOL["table2_group_values"]==["all"]
    assert ANALYSIS_METHOD_POOL["table2"]["arguments"]["group"]=={
        "type":"string","values_from":"table2_group_values"}
    controls=ANALYSIS_METHOD_POOL["table1"]["arguments"]["controls"]
    assert controls["type"]=="list[list[string]]"
    assert "Each inner list represents one model specification" in controls["semantics"]
    filters=ANALYSIS_METHOD_POOL["correlation"]["arguments"]["filters"]
    assert filters["type"]=="dictionary"
    assert "Use {} when no cohort restriction" in filters["semantics"]


def test_prompt_contains_all_methods_and_no_hidden_gold_answer():
    messages=build_code_generation_messages(REQUEST)
    text=repr(messages)
    for method in ANALYSIS_METHOD_POOL:
        assert method in text
    for forbidden in ["requested_analysis","required_filters","Required contract",
            "Expected method","request_mismatches","{'national_team': 'Junior'}",
            "result = analysis.correlation("]:
        assert forbidden not in text
    assert "result = analysis.<method>(...)" in text


def test_same_messages_are_used_for_gpt_gemini_and_claude_adapters():
    captured={}
    def adapter(provider):
        def call(_channel,messages):
            captured[provider]=messages
            return ModelCallResult(None,provider,None,provider,False,True,False,"stop")
        return call
    for provider in ("gpt","gemini","claude"):
        with patch("llm.code_generator._call_for_channel",side_effect=adapter(provider)):
            _generate("strong",REQUEST,{"requested_filters":{"national_team":"Junior"}},
                provider,"correlation",provider)
    assert captured["gpt"]==captured["gemini"]==captured["claude"]


def test_hidden_metadata_does_not_change_model_messages():
    first=build_code_generation_messages(REQUEST)
    second=build_code_generation_messages(REQUEST)
    assert first==second


def test_semantic_repair_includes_local_verifier_feedback_in_chat_sequence():
    calls=[]
    def reject(_channel,messages):
        calls.append(messages)
        return ModelCallResult("result = analysis.table1()","gpt","gpt","test",True,False,False,None)
    with patch("llm.code_generator._call_for_channel",side_effect=reject):
        _generate("strong",REQUEST,{"requested_filters":{"national_team":"Junior"}},
            "id","correlation","gpt")
    assert len(calls)==2
    retry=calls[1]
    assert [message["role"] for message in retry] == ["system","user","assistant","user"]
    assert retry[2]["content"] == "result = analysis.table1()"
    assert "failed local validation" in retry[3]["content"]
    assert "Validation error: Expected method='correlation' but generated 'table1'." in retry[3]["content"]
    assert "Athlete_" not in retry[3]["content"]


def test_frontend_and_evaluation_do_not_define_duplicate_prompt_builders():
    from pathlib import Path
    generator=Path("llm/code_generator.py").read_text(encoding="utf-8")
    assert "build_code_generation_messages(prompt)" in generator
    for path in Path("evaluation").glob("*.py"):
        assert "def build_code_generation_messages" not in path.read_text(encoding="utf-8")
