import inspect
from types import SimpleNamespace

from llm.code_generation_pools import ANALYSIS_METHOD_POOL, ALLOWED_VALUE_POOL
from llm.code_generation_prompt import build_code_generation_messages
from llm.code_generation_prompt_design_v2 import (
    PROMPT_VERSIONS,
    build_codegen_prompt_design_v2_messages,
    prompt_sha256,
)
from llm.model_clients import ModelCallResult
from scripts import evaluate_codegen_prompt_design_v2 as runner


def _system(version: str) -> str:
    return build_codegen_prompt_design_v2_messages(version, "request")[0]["content"]


def test_basic_interface_contains_real_method_signatures():
    system = _system("basic_interface")
    for method, details in ANALYSIS_METHOD_POOL.items():
        assert f"result = analysis.{method}(" in system
        for parameter in details["arguments"]:
            assert f"{parameter}=..." in system


def test_basic_interface_does_not_contain_allowed_value_pool():
    system = _system("basic_interface")
    assert "ALLOWED VALUE POOL" not in system and "Allowed values" not in system
    for domain in ALLOWED_VALUE_POOL["public_domains"]:
        assert domain not in system
    for value in ("pearson", "spearman", "CURRENT_SUBJECT", "selected_cohort"):
        assert value not in system


def test_defined_contains_same_interfaces_allowed_values_and_short_definitions():
    basic = _system("basic_interface"); defined = _system("defined")
    assert defined.startswith(basic)
    for method, details in ANALYSIS_METHOD_POOL.items():
        assert f"result = analysis.{method}(" in defined
        assert details["description"] in defined
    for domain in ALLOWED_VALUE_POOL["public_domains"]:
        assert domain in defined
    for value in ("pearson", "spearman", "age", "sex", "selected_cohort"):
        assert value in defined
    assert "Concise parameter definitions" in defined


def test_defined_omits_full_request_preservation_rules():
    defined = _system("defined")
    for full_only in ("every explicit requirement", "preserve every explicitly requested",
                      "prefer that value over an API default", "silently check"):
        assert full_only not in defined


def test_full_is_exactly_the_production_prompt():
    request = "Calculate correlations for all athletes."
    assert build_codegen_prompt_design_v2_messages("full", request) == build_code_generation_messages(request)


def test_all_variants_receive_the_identical_user_request():
    request = "Run Table 1 for female athletes."
    user_messages = [build_codegen_prompt_design_v2_messages(version, request)[1]
                     for version in PROMPT_VERSIONS]
    assert user_messages == [user_messages[0]] * 3


def test_prompt_hashes_are_distinct_cache_dimensions():
    hashes = [prompt_sha256(version) for version in PROMPT_VERSIONS]
    assert len(set(hashes)) == 3 and all(len(value) == 64 for value in hashes)


def test_v2_uses_one_explicit_model_configuration_for_every_prompt(monkeypatch):
    calls = []
    def fake_call(messages, **kwargs):
        calls.append((kwargs, messages))
        return ModelCallResult(None, "gemini-3.5-flash", None, "openai_compatible",
                               False, True, False, "offline")
    monkeypatch.setattr(runner, "call_gemini_cloud_model", fake_call)
    for version in PROMPT_VERSIONS:
        runner.call_v2_model(build_codegen_prompt_design_v2_messages(version, "same"), max_tokens=2048)
    configurations = [call[0] for call in calls]
    assert configurations == [configurations[0]] * 3
    assert configurations[0] == {"temperature": 0.0, "max_tokens": 2048}


def test_v2_is_fixed_to_gemini_35_flash():
    assert runner.FIXED_MODEL_KEY == "gemini"
    assert runner.FIXED_MODEL_DISPLAY_NAME == "Gemini 3.5 Flash"
    assert "--model" not in inspect.getsource(runner.main)


def test_v2_benchmark_has_exactly_same_40_ids_for_every_prompt():
    _, samples = runner.load_evaluation_samples()
    ids = [sample["id"] for sample in samples]
    assert len(ids) == len(set(ids)) == 40
    assert {version: ids for version in PROMPT_VERSIONS} == {
        version: ids for version in ("basic_interface", "defined", "full")}


def test_v2_is_first_pass_without_auto_correction(monkeypatch):
    calls = []
    def caller(messages, **kwargs):
        calls.append((messages, kwargs))
        return ModelCallResult("result = analysis.table1()", "gpt-4.1", "gpt-4.1",
                               "test", True, False, False, None)
    monkeypatch.setattr(runner, "_model_config", lambda key: SimpleNamespace(display_name="GPT", provider="OpenAI"))
    monkeypatch.setattr(runner, "_verification_without_known_statsmodels_noise", lambda *a, **k: object())
    monkeypatch.setattr(runner, "verification_to_dict", lambda value: {
        "structure_validation_passed": True, "request_match_passed": True,
        "local_execution_passed": True, "result_validation_passed": True,
        "fully_correct": True, "request_mismatches": [],
    })
    sample = {"id": "one", "prompt": "Run Table 1", "requested_analysis": "table1",
              "analysis_filters": {}}
    result = runner.evaluate_pair(sample, "basic_interface", caller=caller)
    assert len(calls) == 1 and result["generation_attempts"] == 1
    assert result["auto_correction_used"] is False
    assert result["model_key"] == "gemini"


def test_incomplete_v2_does_not_score_with_39_denominator():
    rows = [{"sample_id": f"sample_{i}", "task": "table1", "api_call_success": True,
             "evaluation_scored": True, "fully_correct": True, "structure_valid": True,
             "request_match": True, "execute_valid": True, "result_valid": True,
             "error_stage": None} for i in range(39)]
    summary = runner.summarize_prompt(rows, {f"sample_{i}" for i in range(40)})
    assert summary["scored_samples"] == 39 and summary["fully_correct"] is None
    assert summary["unscored_sample_ids"] == ["sample_39"]
