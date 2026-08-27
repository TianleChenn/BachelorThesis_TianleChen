import ast
import json
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


def test_final_privacy_page_uses_exact_offline_artifacts_only():
    renderer = _function_source("_render_privacy_evaluation_tab")
    for path in (
        "artifacts/thesis_evaluation/privacy_cloud_model_evaluation.json",
        "artifacts/privacy_benchmark_metrics.json",
        "artifacts/controlled_per_level_accuracy.csv",
        "artifacts/controlled_privacy_feature_summary.json",
        "artifacts/privacy_benchmark_per_route.csv",
        "artifacts/privacy_benchmark_confusion_matrices.csv",
        "evaluation/frontend_realistic_benchmark_60.json",
        "evaluation/privacy_controlled_benchmark.json",
    ):
        assert path in renderer
    assert 'app_root / "artifacts/privacy_cloud_model_evaluation.json"' not in renderer
    assert "Saved thesis Privacy Assessor evaluation snapshot was not found." in renderer
    assert "evaluate_privacy_cloud_models.py --resume" not in renderer
    assert "privacy_methods_frontend60_comparison.json" not in renderer
    for forbidden in (
        "st.button",
        "subprocess",
        "os.system",
        "call_privacy",
        "call_cloud",
        "run_privacy_benchmark(",
        "evaluate_privacy_cloud_models(",
        "train_soft_gating(",
    ):
        assert forbidden not in renderer


def test_four_main_privacy_modules_are_separated_by_dividers():
    renderer = _function_source("_render_privacy_evaluation_tab")
    assert renderer.count("st.divider()") == 3
    assessor = renderer.index("_render_privacy_assessor_evaluation(assessor)")
    method = renderer.index("_render_method_benchmark_comparison")
    controlled = renderer.index("_render_controlled_benchmark_analysis")
    features = renderer.index("_render_controlled_privacy_features")
    divider_positions = []
    start = 0
    while True:
        position = renderer.find("st.divider()", start)
        if position < 0:
            break
        divider_positions.append(position)
        start = position + 1
    assert assessor < divider_positions[0] < method
    assert method < divider_positions[1] < controlled
    assert controlled < divider_positions[2] < features


def test_privacy_assessor_section_uses_frozen_count_snapshot():
    renderer = _function_source("_render_privacy_assessor_evaluation")
    frames = _function_source("_privacy_assessor_snapshot_frames")
    assert "Privacy Assessor Model Comparison" in renderer
    assert "Saved thesis evaluation snapshot on the 60-request Independent Benchmark." in renderer
    assert "Ground Truth distribution: Cloud 5, Collaboration 35, Local Edge 10, Blocked 10." in renderer
    assert 'report.get("benchmark")' in frames
    assert 'report.get("models")' in frames
    assert 'result.get("route_correct")' in frames
    assert "Blocked Recall" in renderer
    assert "Blocked Classification Accuracy" not in renderer
    assert "Representative Privacy Assessor Example" not in renderer
    for model in ("GPT-4.1", "Gemini 3.5 Flash", "Claude Sonnet 5"):
        assert model in SOURCE
    for hidden in ("raw_model_response", "confidence", "input_tokens", "output_tokens"):
        assert hidden not in renderer


def test_thesis_privacy_assessor_snapshot_counts_drive_expected_percentages():
    path = Path("artifacts/thesis_evaluation/privacy_cloud_model_evaluation.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["snapshot"] == "Bachelor Thesis Section 5.2.2"
    assert report["read_only_snapshot"] is True
    assert report["benchmark"] == {
        "name": "Independent Benchmark",
        "total_samples": 60,
        "ground_truth_counts": {
            "Cloud": 5,
            "Collaboration": 35,
            "Local Edge": 10,
            "Blocked": 10,
        },
    }

    table, overall, per_route = frontend._privacy_assessor_snapshot_frames(report)
    assert table.to_dict("records") == [
        {"Model": "GPT-4.1", "Correct": "28 / 60", "Exact": "46.7%",
         "Non-blocked": "36.0%", "Blocked Recall": "100.0%"},
        {"Model": "Gemini 3.5 Flash", "Correct": "35 / 60", "Exact": "58.3%",
         "Non-blocked": "50.0%", "Blocked Recall": "100.0%"},
        {"Model": "Claude Sonnet 5", "Correct": "34 / 60", "Exact": "56.7%",
         "Non-blocked": "48.0%", "Blocked Recall": "100.0%"},
    ]
    route_values = {
        (row["Privacy Route"], row["Privacy Assessor"]): round(row["Accuracy (%)"], 1)
        for row in per_route.to_dict("records")
    }
    assert route_values == {
        ("Cloud", "GPT-4.1"): 100.0,
        ("Collaboration", "GPT-4.1"): 22.9,
        ("Local Edge", "GPT-4.1"): 50.0,
        ("Blocked", "GPT-4.1"): 100.0,
        ("Cloud", "Gemini 3.5 Flash"): 0.0,
        ("Collaboration", "Gemini 3.5 Flash"): 45.7,
        ("Local Edge", "Gemini 3.5 Flash"): 90.0,
        ("Blocked", "Gemini 3.5 Flash"): 100.0,
        ("Cloud", "Claude Sonnet 5"): 20.0,
        ("Collaboration", "Claude Sonnet 5"): 37.1,
        ("Local Edge", "Claude Sonnet 5"): 100.0,
        ("Blocked", "Claude Sonnet 5"): 100.0,
    }
    overall_values = {
        (row["Metric"], row["Privacy Assessor"]): round(row["Accuracy (%)"], 1)
        for row in overall.to_dict("records")
    }
    assert all(
        overall_values[("Blocked Recall", model)] == 100.0
        for model in frontend._PRIVACY_ASSESSOR_COLORS
    )
    for hard_coded in ("46.7", "58.3", "56.7", "22.9", "45.7", "37.1"):
        assert hard_coded not in _function_source("_privacy_assessor_snapshot_frames")


def test_privacy_assessor_example_is_deterministic_and_uses_same_sample():
    rows = []
    for sample_id, routes in (
        ("same-route", ("cloud", "cloud", "cloud")),
        ("different-route", ("cloud", "local_edge", "collaboration")),
    ):
        for model_key, route in zip(("gpt4_1", "gemini", "claude"), routes):
            rows.append({
                "sample_id": sample_id,
                "model_key": model_key,
                "prompt": f"Prompt {sample_id}",
                "ground_truth_route": "cloud",
                "predicted_route": route,
            })
    first = frontend._select_privacy_assessor_example(rows)
    second = frontend._select_privacy_assessor_example(rows)
    assert first == second
    assert first["sample_id"] == "different-route"
    assert set(first["models"]) == {"gpt4_1", "gemini", "claude"}
    assert len({row["sample_id"] for row in first["models"].values()}) == 1


def test_method_comparison_uses_independent_and_controlled_saved_metrics():
    renderer = _function_source("_render_method_benchmark_comparison")
    assert "Overall Exact Route Accuracy" in renderer
    assert "Exact Route Accuracy by Benchmark" not in renderer
    assert 'metrics.get("independent")' in renderer
    assert 'metrics.get("controlled")' in renderer
    assert "exact_route_accuracy" in renderer
    for method in ("Method A", "Method B", "Method C"):
        assert method in SOURCE
    for hard_coded in ("36.7%", "8.3%", "43.3%", "62.5%", "18.8%", "90.6%"):
        assert hard_coded not in renderer


def test_benchmark_summary_has_separate_counts_and_plain_language_descriptions():
    renderer = _function_source("_render_method_benchmark_comparison")
    assert 'st.columns(2)' in renderer
    assert "Independent Benchmark" in renderer
    assert "Controlled Benchmark" in renderer
    assert (
        "Contains varied analysis requests with different tasks, athlete groups, "
        in renderer
    )
    assert "filters, and privacy conditions." in renderer
    assert "Keeps the analysis task similar while privacy conditions change" in renderer
    assert "systematically from L0 (Cloud) to L3 (Blocked)." in renderer
    assert "Independent route distribution:" not in renderer


def test_controlled_level_chart_uses_saved_accuracy_percent_and_four_levels():
    renderer = _function_source("_render_controlled_benchmark_analysis")
    assert "accuracy_percent" in renderer
    for level in ("L0 Cloud", "L1 Collaboration", "L2 Local Edge", "L3 Blocked"):
        assert level in renderer
    frame = frontend._load_csv_artifact("artifacts/controlled_per_level_accuracy.csv")
    assert not frame.empty
    assert {"method", "privacy_level", "accuracy_percent", "level_label"} <= set(frame)
    assert set(frame["method"]) == {"Method A", "Method B", "Method C"}


def test_four_feature_tabs_share_one_sorted_chart_helper_and_method_colors():
    renderer = _function_source("_render_controlled_privacy_features")
    helper = _function_source("_render_controlled_privacy_feature_chart")
    for field in (
        "privacy_risk_score_mean",
        "subject_scope_mean",
        "data_sensitivity_mean",
        "disclosure_level_mean",
    ):
        assert field in renderer
    assert "st.tabs" in renderer
    assert "sorted(rows" in helper
    assert '["L0", "L1", "L2", "L3"]' in helper
    assert "_PRIVACY_METHOD_COLORS" in helper
    assert frontend._PRIVACY_METHOD_COLORS == {
        "Method A": "#3b82f6",
        "Method B": "#f59e0b",
        "Method C": "#10b981",
    }


def test_privacy_assessor_colors_are_fixed_across_charts():
    assert frontend._PRIVACY_ASSESSOR_COLORS == {
        "GPT-4.1": "#3b82f6",
        "Gemini 3.5 Flash": "#10b981",
        "Claude Sonnet 5": "#8b5cf6",
    }
    renderer = _function_source("_render_privacy_assessor_evaluation")
    assert renderer.count("colors=_PRIVACY_ASSESSOR_COLORS") == 2
    assert renderer.count("show_value_labels=True") == 2


def test_privacy_grouped_chart_labels_are_horizontal():
    renderer = _function_source("_privacy_grouped_chart")
    assert "labelAngle=0" in renderer
    assert "labelOverlap=False" in renderer


def test_privacy_grouped_chart_keeps_zero_bar_and_formats_its_value_label(monkeypatch):
    captured = []
    monkeypatch.setattr(
        frontend.st,
        "altair_chart",
        lambda chart, **kwargs: captured.append(chart.to_dict()),
    )
    frontend._privacy_grouped_chart(
        frontend.pd.DataFrame([
            {"Privacy Route": "Cloud", "Privacy Assessor": "GPT-4.1", "Accuracy (%)": 100.0},
            {"Privacy Route": "Cloud", "Privacy Assessor": "Gemini 3.5 Flash", "Accuracy (%)": 0.0},
            {"Privacy Route": "Cloud", "Privacy Assessor": "Claude Sonnet 5", "Accuracy (%)": 20.0},
        ]),
        x_field="Privacy Route",
        y_field="Accuracy (%)",
        series_field="Privacy Assessor",
        category_order=["Cloud"],
        colors=frontend._PRIVACY_ASSESSOR_COLORS,
        show_value_labels=True,
    )
    assert len(captured) == 1
    assert "0.0%" in str(captured[0])
    assert "100.0%" in str(captured[0])
    assert len(captured[0]["layer"]) == 2


def test_technical_details_are_collapsed_and_use_saved_secondary_metrics():
    renderer = _function_source("_render_privacy_technical_details")
    assert 'st.expander("Technical Details", expanded=False)' in renderer
    for metric in (
        "safety_aware_accuracy",
        "under_protection_rate",
        "over_protection_rate",
        "mean_route_distance",
    ):
        assert metric in renderer
    assert "Per-route Precision / Recall / F1" in renderer
    assert "Confusion Matrices" in renderer
