import inspect
from collections import Counter
from types import SimpleNamespace

import pytest

from llm.code_generation_prompt import build_code_generation_messages
from llm.code_generation_prompt_design_v2 import PROMPT_VERSIONS, build_codegen_prompt_design_v2_messages
from llm.model_clients import ModelCallResult
from scripts import evaluate_local_codegen_prompt_design_v2 as evaluation


def test_exactly_three_prompts_and_required_local_display_names():
    assert PROMPT_VERSIONS == ("basic_interface", "defined", "full")
    assert evaluation.PROMPT_DISPLAY_NAMES == {
        "basic_interface":"Simple", "defined":"Medium",
        "full":"Restricted Code Generation Prompt"}


def test_loader_returns_same_40_samples_and_eight_per_task():
    metadata,samples=evaluation.validate_samples()
    assert metadata["dataset_path"]=="evaluation/frontend_realistic_benchmark_60.json"
    assert len(samples)==len({row["id"] for row in samples})==40
    assert Counter(row["requested_analysis"] for row in samples)=={
        task:8 for task in evaluation.TASKS}


def test_full_prompt_is_exactly_production_prompt():
    request="Run Table 1 for all athletes."
    assert build_codegen_prompt_design_v2_messages("full",request)==build_code_generation_messages(request)


def test_fixed_experiment_settings_and_separate_namespace():
    assert evaluation.MODEL_KEY=="ministral"
    assert evaluation.MODEL_DISPLAY_NAME=="Ministral-3-8B"
    assert evaluation.MODEL_HF_SPEC=="mistralai/Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M"
    assert evaluation.MODEL_ALIAS=="ministral-3-8b-local-eval"
    assert evaluation.QUANTIZATION=="Q4_K_M"
    assert evaluation.TEMPERATURE==0.0 and evaluation.MAX_TOKENS==2048
    assert evaluation.EXPECTED_GENERATIONS==120
    assert evaluation.OUTPUT_DIR.name=="local_prompt_design_v2"
    assert evaluation.OUTPUT_DIR != evaluation.ROOT/"evaluation/results/prompt_design_v2"
    assert "--models" not in inspect.getsource(evaluation.parse_args)


def test_evaluate_pair_is_one_first_pass_and_closes_figures(monkeypatch):
    calls=[];verification_calls=[]
    def caller(messages,**kwargs):
        calls.append((messages,kwargs))
        return ModelCallResult("result = analysis.table1()",evaluation.MODEL_ALIAS,
            evaluation.MODEL_ALIAS,"openai_compatible",True,False,False,None)
    monkeypatch.setattr(evaluation,"_verification_without_known_statsmodels_noise",
        lambda *args,**kwargs:(verification_calls.append((args,kwargs)) or object()))
    monkeypatch.setattr(evaluation,"verification_to_dict",lambda value:{
        "structure_validation_passed":True,"request_match_passed":True,
        "local_execution_passed":True,"result_validation_passed":True,
        "fully_correct":True,"request_mismatches":[]})
    sample={"id":"one","prompt":"Run Table 1","requested_analysis":"table1",
        "analysis_filters":{},"_dataset_sha256":"digest"}
    row=evaluation.evaluate_pair(sample,"basic_interface",caller=caller)
    assert len(calls)==1 and calls[0][1]=={"temperature":0.0,"max_tokens":2048}
    assert verification_calls[0][1]["close_figures_after_execution"] is True
    assert row["generation_attempts"]==1 and row["auto_correction_used"] is False
    assert row["evaluation_scored"] is True


def test_non_ministral_returned_alias_is_rejected(monkeypatch):
    def caller(messages,**kwargs):
        return ModelCallResult("result = analysis.table1()", "qwen", "qwen",
            "openai_compatible",True,False,False,None)
    monkeypatch.setattr(evaluation,"_verification_without_known_statsmodels_noise",lambda *a,**k:object())
    monkeypatch.setattr(evaluation,"verification_to_dict",lambda value:{})
    sample={"id":"one","prompt":"Run Table 1","requested_analysis":"table1",
        "analysis_filters":{},"_dataset_sha256":"digest"}
    row=evaluation.evaluate_pair(sample,"basic_interface",caller=caller)
    assert row["api_call_success"] is False and row["evaluation_scored"] is False
    assert row["error_stage"]=="API_ERROR" and "alias" in row["error_reason"]


def _complete_rows():
    return [{"sample_id":f"sample-{i}","model_key":"ministral","prompt_version":version,
        "task":task,"api_call_success":True,"evaluation_scored":True,
        "structure_valid":True,"request_match":True,"execute_valid":True,
        "result_valid":True,"fully_correct":True,"error_stage":None}
        for version in PROMPT_VERSIONS for task in evaluation.TASKS for i in range(8)]


def test_summary_calculation_for_complete_mocked_rows():
    metadata,samples=evaluation.validate_samples();rows=_complete_rows()
    # Use real benchmark IDs so integrity is part of the assertion.
    for version in PROMPT_VERSIONS:
        selected=[row for row in rows if row["prompt_version"]==version]
        for row,sample in zip(selected,samples):row["sample_id"]=sample["id"]
    summary=evaluation._summary(metadata,samples,rows)
    assert summary["total_expected_generations"]==120
    assert summary["evaluation_complete"] is True
    assert {key:value["display_name"] for key,value in summary["prompt_metadata"].items()}==evaluation.PROMPT_DISPLAY_NAMES
    assert all(value["scored_samples"]==40 and value["fully_correct"]==1
        for value in summary["prompts"].values())


def test_checkpoint_rejects_model_dataset_or_prompt_changes():
    metadata,_=evaluation.validate_samples();model=evaluation.ministral_model()
    version="basic_interface"
    row={"model_key":model.key,"model_hf_spec":model.hf_spec,"model_alias":model.alias,
        "dataset_sha256":metadata["dataset_sha256"],"prompt_version":version,
        "prompt_sha256":evaluation.prompt_sha256(version)}
    evaluation._validate_checkpoint([row],metadata["dataset_sha256"])
    with pytest.raises(RuntimeError,match="model specification"):
        evaluation._validate_checkpoint([{**row,"model_alias":"other"}],metadata["dataset_sha256"])
    with pytest.raises(RuntimeError,match="dataset SHA256"):
        evaluation._validate_checkpoint([row],"other")
    with pytest.raises(RuntimeError,match="prompt SHA256"):
        evaluation._validate_checkpoint([{**row,"prompt_sha256":"other"}],metadata["dataset_sha256"])


def test_cli_defaults_to_one_ministral_server_on_port_8081():
    args=evaluation.parse_args(["--check-only"])
    assert args.port==8081 and args.server_timeout_seconds==1800
    assert not hasattr(args,"models")
