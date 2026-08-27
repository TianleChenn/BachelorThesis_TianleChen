from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from llm.code_generation_prompt import CODE_GENERATION_PROMPT_VERSION
from scripts import evaluate_local_codegen_models as evaluation


def synthetic_rows(*, successful=True):
    rows = []
    tasks = [task for task in evaluation.TASKS for _ in range(8)]
    for key in evaluation.MODEL_KEYS:
        for index, task in enumerate(tasks):
            rows.append({
                "sample_id": f"sample-{index}", "model_key": key,
                "requested_analysis": task, "api_call_success": successful,
                "structure_validation_passed": successful,
                "request_match_passed": successful,
                "local_execution_passed": successful,
                "result_validation_passed": successful,
                "fully_correct": successful, "failure_stage": None if successful else "api_error",
                "request_mismatches": [], "latency_seconds": float(index + 1),
            })
    return rows


def test_registry_is_exact_and_aliases_are_unique_q4_k_m():
    models = evaluation.model_map()
    assert tuple(models) == ("ministral", "qwen", "llama")
    assert len({model.alias for model in models.values()}) == 3
    assert all(model.hf_spec.endswith(":Q4_K_M") for model in models.values())
    assert models["ministral"].alias == "ministral-3-8b-local-eval"
    assert models["qwen"].alias == "qwen2.5-coder-7b-local-eval"
    assert models["llama"].alias == "llama-3.1-8b-local-eval"


def test_loader_uses_shared_frontend_60_and_returns_exact_40_distribution():
    metadata, samples = evaluation.load_evaluation_samples()
    assert evaluation.DATASET == evaluation.ROOT / "evaluation/frontend_realistic_benchmark_60.json"
    assert metadata["shared_benchmark_samples"] == 60
    assert len(samples) == 40
    assert Counter(row["requested_analysis"] for row in samples) == {task: 8 for task in evaluation.TASKS}
    assert not any(row["requested_analysis"] in {"figure2", "individual_profile"} for row in samples)
    assert not any(row["privacy_route"] in {"blocked", "local_edge"} for row in samples)


def test_all_output_artifacts_are_local_codegen_names():
    paths = (evaluation.CHECKPOINT, evaluation.PREDICTIONS, evaluation.SUMMARY,
        evaluation.PER_TASK, evaluation.FAILURES, evaluation.REPORT,
        evaluation.OVERALL_FIGURE, evaluation.TASK_FIGURE, evaluation.LATENCY_FIGURE)
    assert all(path.name.startswith("local_codegen_") for path in paths)
    assert not any("cloud_codegen_model" in path.name for path in paths)


def test_complete_120_rows_calculate_rates_and_latency():
    metadata, _ = evaluation.load_evaluation_samples()
    overall, per_task, failures, report = evaluation.build_results(synthetic_rows(), metadata)
    assert report["status"] == "complete"
    assert report["total_model_generations"] == 120
    assert report["generation_temperature"] == 0.0
    assert report["max_generation_tokens"] == 2048
    assert all(row["Samples"] == row["Completed Model Responses"] == 40 for row in overall)
    assert all(row["Fully Correct Accuracy"] == 1 for row in overall)
    assert all(row["Median Latency Seconds"] == 20.5 for row in overall)
    assert len(per_task) == 15 and all(row["Total"] == 8 and row["Accuracy"] == 1 for row in per_task)
    assert all(row["success"] == 40 for row in failures)


def test_incomplete_or_failed_calls_are_not_marked_complete():
    metadata, _ = evaluation.load_evaluation_samples()
    rows = synthetic_rows()
    rows[0]["api_call_success"] = False
    rows[0]["fully_correct"] = False
    _, _, _, report = evaluation.build_results(rows, metadata)
    assert report["status"] == "incomplete"
    assert report["overall_results"][0]["Completed Model Responses"] == 39
    assert report["overall_results"][0]["Fully Correct Accuracy"] is None
    _, _, _, report = evaluation.build_results(rows[:-1], metadata)
    assert report["status"] == "incomplete"


def test_checkpoint_validates_prompt_physical_model_and_resume_identity():
    model = evaluation.model_map()["ministral"]
    row = {"sample_id":"one", "model_key":"ministral",
        "prompt_version":CODE_GENERATION_PROMPT_VERSION,
        "model_hf_spec":model.hf_spec, "model_alias":model.alias,
        "api_call_success":True}
    evaluation.validate_checkpoint([row])
    assert evaluation.latest_rows([row, {**row, "api_call_success":False}])[0]["api_call_success"] is False
    with pytest.raises(RuntimeError, match="prompt version"):
        evaluation.validate_checkpoint([{**row, "prompt_version":"old"}])
    with pytest.raises(RuntimeError, match="model specification"):
        evaluation.validate_checkpoint([{**row, "model_alias":"different"}])


def test_occupied_evaluation_port_fails_without_starting_or_stopping_process():
    model = evaluation.model_map()["ministral"]
    with patch.object(evaluation, "find_llama_server", return_value="llama-server.exe"), \
         patch.object(evaluation, "port_is_available", return_value=False), \
         patch.object(evaluation.subprocess, "Popen") as popen:
        with pytest.raises(RuntimeError, match="already in use"):
            with evaluation.running_model_server(model, port=8081, timeout_seconds=1):
                pass
    popen.assert_not_called()


def test_server_command_is_localhost_only_and_owned_process_is_terminated():
    model = evaluation.model_map()["qwen"]
    process = Mock()
    process.poll.return_value = None
    with patch.object(evaluation, "find_llama_server", return_value="llama-server.exe"), \
         patch.object(evaluation, "port_is_available", return_value=True), \
         patch.object(evaluation, "_ready_alias", return_value={model.alias}), \
         patch.object(evaluation.subprocess, "Popen", return_value=process) as popen, \
         patch.dict(evaluation.os.environ, {}, clear=False):
        original_url = evaluation.os.environ.get("LLM_LOCAL_BASE_URL")
        with evaluation.running_model_server(model, port=8081, timeout_seconds=1):
            assert evaluation.os.environ["LLM_LOCAL_MODEL"] == model.alias
            assert evaluation.os.environ["LLM_LOCAL_BASE_URL"] == "http://127.0.0.1:8081/v1"
        assert evaluation.os.environ.get("LLM_LOCAL_BASE_URL") == original_url
    command = popen.call_args.args[0]
    assert command[command.index("-hf") + 1] == model.hf_spec
    assert command[command.index("--alias") + 1] == model.alias
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "8081"
    assert "0.0.0.0" not in command
    process.terminate.assert_called_once()
    process.wait.assert_called_once_with(timeout=30)


def test_cli_defaults_keep_evaluation_on_separate_port():
    args = evaluation.parse_args(["--check-only"])
    assert args.port == 8081
    assert args.server_timeout_seconds == 1800
    assert args.models == ["ministral", "qwen", "llama"]
