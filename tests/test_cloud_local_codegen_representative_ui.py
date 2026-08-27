import ast
from pathlib import Path

import frontend


SOURCE = Path("frontend.py").read_text(encoding="utf-8")


def _function_source(name):
    tree = ast.parse(SOURCE)
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return ast.get_source_segment(SOURCE, node) or ""


def _rows(model_keys):
    rows = []
    for sample_id, outcomes in (
        ("same-1", dict.fromkeys(model_keys, True)),
        ("same-2", {key: index == 0 for index, key in enumerate(model_keys)}),
    ):
        for model_key, correct in outcomes.items():
            rows.append({
                "sample_id": sample_id,
                "prompt": f"Prompt for {sample_id}",
                "model_key": model_key,
                "generated_code": f"result = analysis.correlation(model='{model_key}')",
                "structure_validation_passed": True,
                "request_match_passed": correct,
                "local_execution_passed": correct,
                "result_validation_passed": correct,
                "fully_correct": correct,
            })
    return rows


def test_cloud_example_is_one_deterministic_request_for_all_three_models():
    rows = _rows(("gpt4_1", "gemini", "claude"))
    first = frontend._select_cloud_codegen_representative_example(rows)
    second = frontend._select_cloud_codegen_representative_example(rows)
    assert first == second
    assert first["sample_id"] == "same-2"
    assert set(first["models"]) == {"gpt4_1", "gemini", "claude"}
    assert len({row["sample_id"] for row in first["models"].values()}) == 1


def test_cloud_validation_table_has_all_models_and_saved_statuses():
    example = frontend._select_cloud_codegen_representative_example(
        _rows(("gpt4_1", "gemini", "claude"))
    )
    table = frontend._codegen_example_table_data(
        example, frontend._CLOUD_CODEGEN_MODEL_SPECS
    ).set_index("Validation Measure")
    assert list(table.columns) == ["GPT-4.1", "Gemini 3.5 Flash", "Claude Sonnet 5"]
    assert table.loc["Fully Correct", "GPT-4.1"] == "PASS"
    assert table.loc["Fully Correct", "Gemini 3.5 Flash"] == "FAIL"
    assert table.loc["Execution", "Gemini 3.5 Flash"] == "NOT RUN"


def test_cloud_and_local_render_the_same_example_for_validation_and_code():
    cloud = _function_source("_render_cloud_codegen_evaluation")
    local = _function_source("_render_local_codegen_evaluation")
    assert "Representative Cloud Model Evaluation Example" in cloud
    assert "_codegen_example_table_data(example" in cloud
    assert "_render_representative_codegen_calls(example" in cloud
    assert "Representative Local Model Evaluation Example" in local
    assert "_local_model_example_table_data(example)" in local
    assert "_render_representative_codegen_calls(example" in local
    assert "raw_response" not in cloud
    assert "raw_response" not in local


def test_generated_code_renderer_uses_cleaned_code_only_and_never_calls_models():
    renderer = _function_source("_render_representative_codegen_calls")
    assert 'result.get("generated_code")' in renderer
    assert "Representative Generated Code" in renderer
    assert "raw_response" not in renderer
    for forbidden in (
        "subprocess",
        "os.system",
        "call_cloud_model",
        "call_local_model",
        "run_evaluation",
        "verify_and_execute_generated_code",
    ):
        assert forbidden not in renderer


def test_saved_artifacts_supply_complete_cloud_and_local_examples():
    cloud_report = frontend._load_json_artifact(
        "artifacts/cloud_codegen_model_evaluation.json"
    )
    local_report = frontend._load_json_artifact(
        "artifacts/local_codegen_model_evaluation.json"
    )
    cloud = frontend._select_cloud_codegen_representative_example(
        cloud_report.get("per_sample_results") or []
    )
    local = frontend._select_local_model_comparison_example(
        local_report.get("per_sample_results") or []
    )
    assert set(cloud["models"]) == {"gpt4_1", "gemini", "claude"}
    assert set(local["models"]) == {"ministral", "qwen", "llama"}
    assert all(row.get("generated_code") for row in cloud["models"].values())
    assert all(row.get("generated_code") for row in local["models"].values())


def test_both_cloud_charts_use_fixed_blue_green_purple_mapping(monkeypatch):
    expected = {
        "GPT-4.1": "#3b82f6",
        "Gemini 3.5 Flash": "#10b981",
        "Claude Sonnet 5": "#8b5cf6",
    }
    assert frontend._CLOUD_CODEGEN_MODEL_COLORS == expected
    assert frontend._CLOUD_CODEGEN_TASK_LABELS == {
        "Table 1": "Logistic Regression",
        "Table 2": "Multiple Linear Regression",
        "Figure 1": "Network Analysis",
    }
    report = {
        "overall_results": [
            {"Model": model, "Fully Correct Accuracy": value}
            for model, value in zip(expected, (0.55, 0.65, 0.50))
        ],
        "per_task_results": [
            {"Task": "Table 1", "Model": model, "Accuracy": value}
            for model, value in zip(expected, (0.50, 0.625, 0.25))
        ],
    }
    captured = []
    monkeypatch.setattr(
        frontend.st,
        "altair_chart",
        lambda chart, **kwargs: captured.append(chart.to_dict()),
    )
    frontend._render_cloud_codegen_accuracy_charts(report)

    def color_scales(value):
        scales = []
        if isinstance(value, dict):
            color = value.get("color")
            if isinstance(color, dict) and isinstance(color.get("scale"), dict):
                scales.append(color["scale"])
            for child in value.values():
                scales.extend(color_scales(child))
        elif isinstance(value, list):
            for child in value:
                scales.extend(color_scales(child))
        return scales

    assert len(captured) == 2
    for spec in captured:
        scales = color_scales(spec)
        assert scales
        assert all(scale["domain"] == list(expected) for scale in scales)
        assert all(scale["range"] == list(expected.values()) for scale in scales)
    assert captured[0]["title"] == "Cloud LLM Restricted Code Fully Correct Accuracy"
    assert captured[1]["title"] == "Fully Correct Accuracy by Analysis Task"
    assert captured[0]["height"] == frontend._MODEL_COMPARISON_CHART_HEIGHT
    assert captured[1]["height"] == frontend._MODEL_COMPARISON_CHART_HEIGHT
    assert captured[0]["layer"][0]["encoding"]["x"]["axis"]["labelAngle"] == 0
    assert captured[1]["encoding"]["x"]["axis"]["labelAngle"] == 0
    assert "Logistic Regression" in str(captured[1])
    assert "Table 1" not in str(captured[1])
