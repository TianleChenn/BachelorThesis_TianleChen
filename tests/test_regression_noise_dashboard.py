from __future__ import annotations

import ast
from pathlib import Path


RENDERER = Path("ui/result_renderer.py").read_text(encoding="utf-8")
FRONTEND = Path("frontend.py").read_text(encoding="utf-8")


def _noise_renderer_source() -> str:
    tree = ast.parse(RENDERER)
    return "\n".join(
        ast.get_source_segment(RENDERER, node) or ""
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and "noise" in node.name.lower()
    )


def test_table2_noise_dashboard_uses_a_mean_table_and_one_stability_metric():
    renderer = _noise_renderer_source()
    assert "Mean Perturbed Table 2-style Linear Regression" in RENDERER
    assert "Linear Regression Noise Stability Evaluation" in RENDERER
    assert "Average Standardized Coefficient Difference" in RENDERER
    assert "mean_perturbed_result" in renderer
    assert "stability_figure" in renderer
    assert renderer.count("st.metric(") == 1


def test_table2_noise_dashboard_does_not_restore_removed_technical_content():
    renderer = _noise_renderer_source()
    for removed in (
        "Change in Model Fit",
        "Original model fit",
        "Average model fit after noise",
        "Technical details",
        "Full-precision numerical results",
        "Original vs Mean Perturbed Comparison",
    ):
        assert removed not in renderer


def test_original_table2_model_diagnostics_remain_before_noise_results():
    final_start = FRONTEND.index("def _render_final_analysis_result")
    final_end = FRONTEND.index("\ndef show_pipeline_response", final_start)
    final_result = FRONTEND[final_start:final_end]
    start = FRONTEND.index("def show_pipeline_response")
    end = FRONTEND.index("\ndef run_request", start)
    pipeline = FRONTEND[start:end]
    assert "Model diagnostics" in final_result
    assert final_result.index("Model diagnostics") < final_result.index("render_noise_utility")
    assert pipeline.index("show_pipeline_overview") < pipeline.index("_render_final_analysis_result")
