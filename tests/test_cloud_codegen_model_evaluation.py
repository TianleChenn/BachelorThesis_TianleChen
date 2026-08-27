from collections import Counter
import pytest

from llm.analysis_request_contracts import build_request_contract,render_request_contract
from llm.code_generation_prompt import CODE_GENERATION_PROMPT_VERSION,build_code_generation_messages
from llm.model_clients import ModelCallResult,get_cloud_codegen_evaluation_models
from scripts import evaluate_cloud_codegen_models as benchmark


def successful_call(code, captured=None):
    def call(model_key,messages,*,max_tokens):
        if captured is not None:captured[model_key]=messages
        return ModelCallResult(code,model_key,model_key,"test",True,False,False,None,
            input_tokens=100,output_tokens=20,total_tokens=120,latency_seconds=.1)
    return call


def correlation_sample():
    _,samples=benchmark.load_evaluation_samples()
    return next(sample for sample in samples
        if sample["requested_analysis"]=="correlation" and not sample["analysis_filters"])


def test_locked_dataset_and_task_distribution():
    metadata,samples=benchmark.load_evaluation_samples()
    assert metadata["shared_benchmark_samples"]==60
    assert len(samples)==40
    assert Counter(row["requested_analysis"] for row in samples)=={
        "table1":8,"table2":8,"figure1":8,"correlation":8,"variance_analysis":8,
    }
    assert not any(row["privacy_route"] in {"blocked","local_edge"} for row in samples)


def test_registry_contains_exactly_three_configured_models():
    assert CODE_GENERATION_PROMPT_VERSION=="pool_typed_schema_v3"
    models=get_cloud_codegen_evaluation_models()
    assert [model.key for model in models]==["gpt4_1","gemini","claude"]
    assert [model.default_model for model in models]==[
        "gpt-4.1","gemini-3.5-flash","claude-sonnet-5",
    ]


def test_run_label_creates_optional_separate_paths_from_default_v2():
    current=benchmark.artifact_paths();v2=benchmark.artifact_paths("completeness_v2")
    assert current["report"].name=="cloud_codegen_model_evaluation.json"
    assert v2["report"].name=="cloud_codegen_model_evaluation_completeness_v2.json"
    assert v2["checkpoint"].name=="cloud_codegen_model_checkpoint_completeness_v2.jsonl"
    assert set(current.values()).isdisjoint(v2.values())
    with pytest.raises(ValueError):benchmark.artifact_paths("bad label")


def test_all_providers_receive_identical_messages_without_hidden_gold():
    sample=correlation_sample();captured={}
    code=render_request_contract(build_request_contract("correlation"))
    caller=successful_call(code,captured)
    for key in benchmark.MODEL_KEYS:benchmark.evaluate_pair(sample,key,caller=caller)
    assert captured["gpt4_1"]==captured["gemini"]==captured["claude"]
    text=repr(captured["gpt4_1"])
    assert "requested_analysis" not in text
    assert "analysis_filters" not in text
    assert "AnalysisRequestContract" not in text
    assert "result = analysis.correlation(" not in text
    assert captured["gpt4_1"]==build_code_generation_messages(sample["prompt"])


def test_correct_code_is_fully_correct_and_wrong_code_is_not():
    sample=correlation_sample()
    correct=render_request_contract(build_request_contract("correlation"))
    passed=benchmark.evaluate_pair(sample,"gpt4_1",caller=successful_call(correct))
    assert passed["fully_correct"] is True
    assert passed["prompt_version"]=="pool_typed_schema_v3"
    wrong=render_request_contract(build_request_contract("table1"))
    failed=benchmark.evaluate_pair(sample,"gpt4_1",caller=successful_call(wrong))
    assert failed["structure_validation_passed"] is True
    assert failed["request_match_passed"] is False
    assert failed["fully_correct"] is False


def test_api_error_never_counts_as_correct():
    sample=correlation_sample()
    def fail(model_key,messages,*,max_tokens):
        return ModelCallResult(None,model_key,None,"test",False,True,False,"unavailable")
    row=benchmark.evaluate_pair(sample,"gpt4_1",caller=fail)
    assert row["failure_stage"]=="api_error"
    assert row["api_call_success"] is False
    assert row["fully_correct"] is False


def synthetic_rows(count_per_model=40,api_success=True):
    rows=[]
    for key in benchmark.MODEL_KEYS:
        for index in range(count_per_model):
            task=benchmark.TASKS[index%5]
            rows.append({"sample_id":f"{task}-{index}","model_key":key,
                "requested_analysis":task,"api_call_success":api_success,
                "structure_validation_passed":api_success,"request_match_passed":api_success,
                "local_execution_passed":api_success,"result_validation_passed":api_success,
                "fully_correct":api_success,"failure_stage":None if api_success else "api_error",
                "request_mismatches":[],"input_tokens":100 if api_success else None,
                "output_tokens":20 if api_success else None,"latency_seconds":.1})
    return rows


def test_summary_denominators_are_40_and_8():
    metadata,_=benchmark.load_evaluation_samples()
    overall,per_task,_,report=benchmark.build_results(synthetic_rows(),metadata)
    assert all(row["Samples"]==40 for row in overall)
    assert all(row["Total"]==8 for row in per_task)
    assert report["status"]=="complete"
    assert report["prompt_version"]=="pool_typed_schema_v3"
    assert report["total_model_generations"]==120
    assert all(row["prompt_version"]=="pool_typed_schema_v3" for row in overall)
    assert all(row["prompt_version"]=="pool_typed_schema_v3" for row in per_task)


def test_report_requires_120_rows_and_counts_api_errors_as_incorrect():
    metadata,_=benchmark.load_evaluation_samples()
    _,_,_,short=benchmark.build_results(synthetic_rows(39),metadata)
    assert short["status"]=="incomplete"
    rows=synthetic_rows();rows[0]["api_call_success"]=False
    _,_,_,failed=benchmark.build_results(rows,metadata)
    assert failed["status"]=="incomplete"
    assert failed["overall_results"][0]["Fully Correct Accuracy"] is None


def test_cleanup_allowlist_contains_only_named_cloud_codegen_artifacts(tmp_path,monkeypatch):
    monkeypatch.setattr(benchmark,"ARTIFACTS",tmp_path)
    targets=[tmp_path/name for name in benchmark.OLD_V1_ARTIFACT_NAMES]
    for path in targets:path.write_text("old",encoding="utf-8")
    unrelated=tmp_path/"privacy_benchmark_predictions.csv"
    unrelated.write_text("keep",encoding="utf-8")
    assert set(benchmark.cleanup_old_v1())==set(targets)
    assert unrelated.exists()


def test_resume_checkpoint_rejects_different_or_missing_prompt_version():
    benchmark._validate_checkpoint_prompt_version([
        {"prompt_version":"pool_typed_schema_v3"},
    ])
    with pytest.raises(RuntimeError,match=(
            "Checkpoint prompt version does not match current prompt version")):
        benchmark._validate_checkpoint_prompt_version([
            {"prompt_version":"pool_"+"completeness_v2"},
        ])
    with pytest.raises(RuntimeError,match=(
            "Checkpoint prompt version does not match current prompt version")):
        benchmark._validate_checkpoint_prompt_version([{}])
