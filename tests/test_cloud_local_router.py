import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import joblib
import matplotlib.pyplot as plt

from llm.analysis_request_contracts import build_request_contract, render_request_contract
from llm.athlete_cloud_local_router import AthleteCloudLocalRouter
from llm.code_generation_prompt import (
    CODE_GENERATION_PROMPT_VERSION, RESTRICTED_CODE_MAX_TOKENS,
    build_code_generation_messages,
)
from llm.code_generator import generate_code
from llm.model_clients import ModelCallResult, call_gemini_cloud_model, get_local_codegen_runtime
from privacy.cloud_local_router import route_cloud_local
from scripts.cloud_local_evaluation_common import (
    evaluate_cloud_and_local, evaluate_model_candidate, preference_from_correctness,
)


class _Pipeline:
    def __init__(self, probability): self.probability = probability
    def predict_proba(self, prompts): return [[1-self.probability, self.probability] for _ in prompts]


class _WrongClassPipeline(_Pipeline):
    classes_ = [1, 0]


def _router(tmp_path: Path, probability: float, threshold: float = .5):
    model = tmp_path / "router.joblib"
    metadata = tmp_path / "router.json"
    joblib.dump(_Pipeline(probability), model)
    import hashlib
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    metadata.write_text(json.dumps({"status":"trained", "router_type":"project_specific_cloud_local_router",
                                    "model_sha256":digest, "threshold":threshold}), encoding="utf-8")
    return AthleteCloudLocalRouter(model, metadata)


def _result(content=None, success=True):
    return ModelCallResult(content, "model", "model" if success else None, "openai_compatible",
                           success, not success, False, None if success else "failed", "chat.completions")


def test_prompt_version_is_locked():
    assert CODE_GENERATION_PROMPT_VERSION == "pool_typed_schema_v3"
    assert RESTRICTED_CODE_MAX_TOKENS >= 1024


def test_shared_prompt_states_predefined_regression_contracts():
    system = build_code_generation_messages("request")[0]["content"]
    assert 'controls=[[], ["sex"], ["age"], ["sex", "age"]]' in system
    assert 'Table 2 must use all eight public-domain predictors, group="all"' in system


def test_preference_policy():
    assert preference_from_correctness(True, True) == "local"
    assert preference_from_correctness(True, False) == "cloud"
    assert preference_from_correctness(False, False) == "invalid"
    assert preference_from_correctness(False, True) == "local"


def test_cloud_probability_threshold_decision(tmp_path):
    assert _router(tmp_path, .7, .6).predict("request")["selected_tier"] == "cloud"
    assert _router(tmp_path, .2, .6).predict("request")["selected_tier"] == "local"


def test_cloud_probability_rejects_wrong_class_order(tmp_path):
    model = tmp_path / "router.joblib"
    metadata = tmp_path / "router.json"
    joblib.dump(_WrongClassPipeline(.7), model)
    import hashlib
    metadata.write_text(json.dumps({"status":"trained", "router_type":"project_specific_cloud_local_router",
        "model_sha256":hashlib.sha256(model.read_bytes()).hexdigest(), "threshold":.5}), encoding="utf-8")
    try: AthleteCloudLocalRouter(model, metadata).predict("request")
    except ValueError as exc: assert "classes [0, 1]" in str(exc)
    else: raise AssertionError("Invalid classifier class order was accepted")


def test_local_edge_bypasses_classifier():
    router = Mock()
    decision = route_cloud_local("private", "local_edge", athlete_router=router)
    assert decision.selected_tier == "local"
    router.predict.assert_not_called()


def test_nonlocal_local_endpoint_is_rejected():
    with patch.dict(os.environ, {"LLM_LOCAL_BASE_URL":"https://api.mistral.ai/v1"}, clear=False):
        try: get_local_codegen_runtime()
        except RuntimeError as exc: assert "localhost" in str(exc)
        else: raise AssertionError("Remote Local endpoint was accepted")


def test_gemini_cloud_config_uses_gemini_environment():
    env = {"LLM_GEMINI_PROVIDER":"openai_compatible", "LLM_GEMINI_MODEL":"gemini-test",
           "LLM_GEMINI_BASE_URL":"https://example.invalid/v1", "LLM_GEMINI_API_KEY":"secret"}
    with patch.dict(os.environ, env, clear=False), patch("llm.model_clients._call", return_value=_result("ok")) as call:
        call_gemini_cloud_model([{"role":"user", "content":"request"}])
    assert call.call_args.args[1:5] == ("gemini-test", "openai_compatible", "secret", "https://example.invalid/v1")
    assert call.call_args.args[5] is None


def test_gemini_runtime_ignores_explicit_temperature_and_uses_provider_default():
    with patch("llm.model_clients._call", return_value=_result("ok")) as call:
        call_gemini_cloud_model(
            [{"role": "user", "content": "request"}],
            temperature=0.7,
        )
    assert call.call_args.args[5] is None


def test_objective_evaluator_uses_shared_restricted_messages():
    sample = {"prompt":"correlate all domains", "analysis_type":"correlation", "requested_filters":{}}
    code = render_request_contract(build_request_contract("correlation"))
    caller = Mock(return_value=_result(code))
    evaluate_model_candidate(sample, caller)
    assert caller.call_args.args[0] == build_code_generation_messages(sample["prompt"])


def test_complete_long_table1_response_is_preserved_and_finish_reason_recorded():
    sample = {"prompt":"build Table 1", "analysis_type":"table1", "requested_filters":{}}
    code = render_request_contract(build_request_contract("table1"))
    response = _result(code)
    response.finish_reason = "stop"
    caller = Mock(return_value=response)
    evaluated = evaluate_model_candidate(sample, caller)
    assert evaluated["raw_response"] == code
    assert evaluated["generated_code"] == code
    assert evaluated["fully_correct"] is True
    assert evaluated["validation_details"]["structure_validation_passed"] is True
    assert evaluated["validation_details"]["request_match_passed"] is True
    assert evaluated["finish_reason"] == "stop"
    assert caller.call_args.kwargs["max_tokens"] == RESTRICTED_CODE_MAX_TOKENS


def test_repeated_figure1_candidate_validation_does_not_accumulate_figures():
    sample = {"prompt":"build Figure 1", "analysis_type":"figure1", "requested_filters":{}}
    code = render_request_contract(build_request_contract("figure1"))
    initial = set(plt.get_fignums())

    for _ in range(12):
        evaluated = evaluate_model_candidate(sample, Mock(return_value=_result(code)))
        assert evaluated["fully_correct"] is True

    assert set(plt.get_fignums()) == initial


def test_candidate_execution_exception_is_recorded_without_aborting_evaluation():
    sample = {"prompt":"build Figure 1", "analysis_type":"figure1", "requested_filters":{}}
    code = render_request_contract(build_request_contract("figure1"))
    with patch(
        "scripts.cloud_local_evaluation_common.verify_and_execute_generated_code",
        side_effect=RuntimeError("Fail to allocate bitmap"),
    ):
        evaluated = evaluate_model_candidate(sample, Mock(return_value=_result(code)))

    assert evaluated["fully_correct"] is False
    assert evaluated["validation_details"]["failure_stage"] == "local_execution"
    assert "Fail to allocate bitmap" in evaluated["validation_details"]["validation_error"]


def test_controlled_comparison_uses_identical_original_request_messages():
    sample = {
        "prompt":"original experimental correlation request",
        "cloud_prompt":"privacy transformed text must be ignored",
        "analysis_type":"correlation", "requested_filters":{},
    }
    code = render_request_contract(build_request_contract("correlation"))
    cloud = Mock(return_value=_result(code))
    local = Mock(return_value=_result(code))
    with patch("scripts.cloud_local_evaluation_common.call_gemini_cloud_model", cloud), \
         patch("scripts.cloud_local_evaluation_common.call_local_codegen_model", local):
        evaluate_cloud_and_local(sample)
    expected = build_code_generation_messages(sample["prompt"])
    assert cloud.call_args.args[0] == expected
    assert local.call_args.args[0] == expected
    assert "privacy transformed" not in json.dumps(cloud.call_args.args[0])


def test_blocked_request_bypasses_all_models():
    with patch("llm.code_generator._call_for_channel") as caller:
        result = generate_code("blocked", {}, {"route":"blocked", "blocked":True})
    caller.assert_not_called()
    assert result.code_source == "blocked"


def test_cloud_and_local_selections_call_explicit_tiers():
    code = render_request_contract(build_request_contract("correlation"))
    for tier in ("cloud", "local"):
        with patch("llm.code_generator._call_for_channel", return_value=_result(code)) as caller:
            result = generate_code("correlation", {"selected_tier":tier, "selected_model":f"{tier}_model"},
                                   {"route":"cloud"}, requested_analysis="correlation", requested_filters={})
        assert result.code
        assert caller.call_args.args[0] == tier
        assert caller.call_args.args[1] == build_code_generation_messages("correlation")


def test_collaboration_cloud_never_receives_original_prompt():
    original = "original private correlation request"
    approved = "perturbed approved correlation request"
    code = render_request_contract(build_request_contract("correlation"))
    captured = []
    def caller(tier, messages):
        captured.extend(messages)
        return _result(code)
    with patch("llm.code_generator._call_for_channel", side_effect=caller):
        result = generate_code(original, {"selected_tier":"cloud", "selected_model":"cloud_gemini"},
                               {"route":"collaboration", "cloud_prompt":approved},
                               requested_analysis="correlation", requested_filters={})
    assert result.code
    payload = json.dumps(captured)
    assert approved in payload and original not in payload


def test_collaboration_local_uses_original_request_without_privacy_transform():
    original = "original local correlation request"
    approved = "perturbed cloud-only correlation request"
    code = render_request_contract(build_request_contract("correlation"))
    with patch("llm.code_generator._call_for_channel", return_value=_result(code)) as caller:
        result = generate_code(original, {"selected_tier":"local", "selected_model":"local_ministral"},
                               {"route":"collaboration", "cloud_prompt":approved},
                               requested_analysis="correlation", requested_filters={})
    assert result.code
    messages = caller.call_args.args[1]
    assert original in json.dumps(messages) and approved not in json.dumps(messages)
