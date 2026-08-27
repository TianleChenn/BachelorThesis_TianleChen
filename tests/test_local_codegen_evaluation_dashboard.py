import ast
from pathlib import Path

import frontend


SOURCE = Path("frontend.py").read_text(encoding="utf-8")


def _function_source(name):
    tree = ast.parse(SOURCE)
    node = next(
        item for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    return ast.get_source_segment(SOURCE, node) or ""


def test_local_model_evaluation_is_rendered_after_cloud_evaluation():
    page = _function_source("_render_cloud_local_evaluation_tab")
    assert page.index("_render_cloud_codegen_evaluation(app_root)") < page.index(
        "_render_local_codegen_evaluation(app_root)"
    ) < page.index("Representative Cost-aware Routing Examples")


def test_local_evaluation_reads_saved_artifact_without_executing_models():
    renderer = _function_source("_render_local_codegen_evaluation")
    assert "Local Model Comparison for Restricted Code Generation" in renderer
    assert 'app_root / "artifacts/local_codegen_model_evaluation.json"' in renderer
    assert "_load_json_artifact" in renderer
    assert "python -m scripts.evaluate_local_codegen_models --resume" in renderer
    for forbidden in (
        "subprocess",
        "os.system",
        "run_evaluation",
        "call_local_model",
        "llama-server",
        "verify_and_execute_generated_code",
    ):
        assert forbidden not in renderer


def test_local_evaluation_has_required_models_metrics_and_task_labels():
    assert frontend._LOCAL_CODEGEN_MODELS == (
        "Ministral-3-8B",
        "Qwen2.5-Coder-7B-Instruct",
        "Llama-3.1-8B-Instruct",
    )
    assert [label for label, _ in frontend._LOCAL_CODEGEN_OVERALL_METRICS] == [
        "Structure",
        "Request Match",
        "Execution",
        "Result",
        "Fully Correct",
    ]
    assert list(frontend._LOCAL_CODEGEN_TASK_LABELS.values()) == [
        "Logistic Regression",
        "Multiple Linear Regression",
        "Network Analysis",
        "Correlation Analysis",
        "Variance Analysis",
    ]


def test_chart_values_are_derived_from_report_rates():
    report = {
        "overall_results": [{
            "Model": "Ministral-3-8B",
            "Structure Valid Rate": 0.91,
            "Request Match Rate": 0.82,
            "Execution Success Rate": 0.73,
            "Result Valid Rate": 0.64,
            "Fully Correct Accuracy": 0.55,
            "Fully Correct": 22,
            "Samples": 40,
        }],
        "per_task_results": [{
            "Task": "Table 1",
            "Model": "Ministral-3-8B",
            "Accuracy": 0.375,
        }],
    }

    overall = frontend._local_codegen_overall_chart_data(report)
    assert overall["Accuracy (%)"].tolist() == [91.0, 82.0, 73.0, 64.0, 55.0]
    fully_correct = frontend._local_codegen_fully_correct_chart_data(report)
    assert fully_correct.to_dict("records") == [{
        "Model": "Ministral-3-8B",
        "Fully Correct Accuracy (%)": 55.0,
    }]
    task = frontend._local_codegen_task_chart_data(report)
    assert task.to_dict("records") == [{
        "Analysis Task": "Logistic Regression",
        "Model": "Ministral-3-8B",
        "Fully Correct Accuracy (%)": 37.5,
    }]
    overall_table = frontend._local_codegen_overall_table_data(report)
    assert overall_table.to_dict("records") == [{
        "Model": "Ministral-3-8B",
        "Fully Correct / 40": "22 / 40",
        "Fully Correct Accuracy": "55.0%",
        "Structure Valid Rate": "91.0%",
        "Request Match Rate": "82.0%",
        "Execution Rate": "73.0%",
        "Result Valid Rate": "64.0%",
    }]


def test_local_layout_matches_cloud_style_and_includes_representative_example():
    renderer = _function_source("_render_local_codegen_evaluation")
    assert "Prompt Version:" in renderer
    assert "_local_codegen_overall_table_data" in renderer
    assert "Local LLM Restricted Code Fully Correct Accuracy" in renderer
    assert "_local_codegen_fully_correct_chart_data" in renderer
    assert "_render_simple_fully_correct_chart" in renderer
    assert "Overall Restricted Code Generation Performance" not in renderer
    grouped_chart = _function_source("_render_grouped_accuracy_chart")
    assert "Fully Correct Accuracy by Analysis Task" in grouped_chart
    assert "labelAngle=0" in grouped_chart
    assert "labelOverlap=False" in grouped_chart
    assert 'orient="top-right"' in grouped_chart
    assert "_LOCAL_CODEGEN_MODEL_COLORS" in grouped_chart
    assert "Representative Local Model Evaluation Example" in renderer
    assert "_select_local_model_comparison_example" in renderer
    assert "_local_model_example_table_data" in renderer


def test_both_local_charts_use_the_same_fixed_model_colors(monkeypatch):
    expected = {
        "Ministral-3-8B": "#3b82f6",
        "Qwen2.5-Coder-7B-Instruct": "#10b981",
        "Llama-3.1-8B-Instruct": "#8b5cf6",
    }
    assert frontend._LOCAL_CODEGEN_MODEL_COLORS == expected

    captured = []
    monkeypatch.setattr(
        frontend.st,
        "altair_chart",
        lambda chart, **kwargs: captured.append(chart.to_dict()),
    )
    overall = frontend._local_codegen_fully_correct_chart_data({
        "overall_results": [
            {"Model": model, "Fully Correct Accuracy": value}
            for model, value in zip(expected, (0.55, 0.175, 0.025))
        ]
    })
    task = frontend._local_codegen_task_chart_data({
        "per_task_results": [
            {"Task": "Table 1", "Model": model, "Accuracy": value}
            for model, value in zip(expected, (0.625, 0.0, 0.0))
        ]
    })
    frontend._render_simple_fully_correct_chart(overall)
    frontend._render_grouped_accuracy_chart(
        task,
        category_field="Analysis Task",
        value_field="Fully Correct Accuracy (%)",
        category_order=list(frontend._LOCAL_CODEGEN_TASK_LABELS.values()),
    )

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


def test_overall_local_chart_model_labels_are_horizontal_and_not_truncated():
    source = _function_source("_render_simple_fully_correct_chart")
    assert "labelAngle=0" in source
    assert "labelOverlap=False" in source
    assert "labelLimit=260" in source
    assert "labelAngle=-15" not in source


def test_all_local_model_comparison_charts_use_shared_height():
    simple = _function_source("_render_simple_fully_correct_chart")
    grouped = _function_source("_render_grouped_accuracy_chart")
    assert "_MODEL_COMPARISON_CHART_HEIGHT" in simple
    assert "_MODEL_COMPARISON_CHART_HEIGHT" in grouped
    assert frontend._MODEL_COMPARISON_CHART_HEIGHT == 350


def test_representative_example_is_same_request_and_selected_deterministically():
    rows = []
    for sample_id, outcomes in (
        ("request-1", {"ministral": False, "qwen": False, "llama": False}),
        ("request-2", {"ministral": True, "qwen": False, "llama": False}),
        ("request-3", {"ministral": True, "qwen": False, "llama": False}),
    ):
        for model_key, fully_correct in outcomes.items():
            rows.append({
                "sample_id": sample_id,
                "prompt": f"Prompt for {sample_id}",
                "model_key": model_key,
                "structure_validation_passed": fully_correct,
                "request_match_passed": fully_correct,
                "local_execution_passed": fully_correct,
                "result_validation_passed": fully_correct,
                "fully_correct": fully_correct,
            })

    first = frontend._select_local_model_comparison_example(rows)
    second = frontend._select_local_model_comparison_example(rows)
    assert first == second
    assert first["sample_id"] == "request-2"
    assert {row["sample_id"] for row in first["models"].values()} == {"request-2"}
    assert set(first["models"]) == {"ministral", "qwen", "llama"}


def test_representative_validation_table_uses_saved_stage_fields():
    example = {
        "models": {
            "ministral": {
                "structure_validation_passed": True,
                "request_match_passed": True,
                "local_execution_passed": True,
                "result_validation_passed": True,
                "fully_correct": True,
            },
            "qwen": {
                "structure_validation_passed": True,
                "request_match_passed": False,
                "local_execution_passed": False,
                "result_validation_passed": False,
                "fully_correct": False,
            },
            "llama": {
                "structure_validation_passed": False,
                "request_match_passed": False,
                "local_execution_passed": False,
                "result_validation_passed": False,
                "fully_correct": False,
            },
        }
    }
    table = frontend._local_model_example_table_data(example).set_index(
        "Validation Measure"
    )
    assert table.loc["Fully Correct", "Ministral-3-8B"] == "PASS"
    assert table.loc["Request Match", "Qwen2.5-Coder-7B-Instruct"] == "FAIL"
    assert table.loc["Execution", "Qwen2.5-Coder-7B-Instruct"] == "NOT RUN"
    assert table.loc["Request Match", "Llama-3.1-8B-Instruct"] == "NOT RUN"


def test_current_artifact_has_a_complete_preferred_representative_request():
    report = frontend._load_json_artifact(
        "artifacts/local_codegen_model_evaluation.json"
    )
    example = frontend._select_local_model_comparison_example(
        report.get("per_sample_results") or []
    )
    assert example is not None
    assert set(example["models"]) == {"ministral", "qwen", "llama"}
    assert example["models"]["ministral"]["fully_correct"] is True
    assert (
        example["models"]["qwen"]["fully_correct"] is False
        or example["models"]["llama"]["fully_correct"] is False
    )
    assert len({row["sample_id"] for row in example["models"].values()}) == 1


def test_cloud_evaluation_renderer_remains_separate():
    cloud_renderer = _function_source("_render_cloud_codegen_evaluation")
    assert "Cloud Model Comparison for Restricted Code Generation" in cloud_renderer
    assert "Cloud LLM Code Generation Evaluation" not in cloud_renderer
    assert "local_codegen_model_evaluation.json" not in cloud_renderer
    assert "_render_local_codegen_evaluation" not in cloud_renderer
