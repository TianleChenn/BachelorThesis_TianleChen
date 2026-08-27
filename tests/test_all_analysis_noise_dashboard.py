from __future__ import annotations

import ast
from pathlib import Path


FRONTEND = Path("frontend.py").read_text(encoding="utf-8")
RENDERER = Path("ui/result_renderer.py").read_text(encoding="utf-8")


def _function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    return ast.get_source_segment(source, function) or ""


def _noise_renderer_source() -> str:
    tree = ast.parse(RENDERER)
    functions = [
        ast.get_source_segment(RENDERER, node) or ""
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and "noise" in node.name.lower()
    ]
    assert functions, "A shared noise-result renderer is required."
    return "\n".join(functions)


def test_shared_noise_renderer_consumes_the_three_part_result_contract():
    renderer = _noise_renderer_source()
    assert "mean_perturbed_result" in renderer
    assert "average_difference" in renderer
    assert "stability_figure" in renderer
    assert 'st.caption("PART 1")' in FRONTEND
    assert "original_title" in FRONTEND
    assert "Mean Perturbed" in RENDERER
    assert "Noise Stability Evaluation" in RENDERER


def test_mean_perturbed_figure_is_rendered_before_its_table():
    renderer = _function_source(RENDERER, "_render_mean_perturbed_result")
    assert renderer.index("st.pyplot(") < renderer.index("st.dataframe(")


def test_anonymous_profile_uses_the_shared_three_part_noise_renderer():
    assert '"individual_profile"' in RENDERER
    assert "Original Anonymous Athlete Profile" in RENDERER
    assert "Mean Perturbed Anonymous Athlete Profile" in RENDERER
    assert "Anonymous Profile Noise Stability Evaluation" in RENDERER
    assert "Average Anonymous Profile Difference" in RENDERER
    assert "Anonymous Profile Stability After Numerical Perturbation" in RENDERER


def test_noise_renderer_has_one_metric_and_no_removed_detail_dashboard():
    renderer = _noise_renderer_source()
    assert renderer.count("st.metric(") == 1
    for removed in (
        "Technical details",
        "Full-precision numerical results",
        "Original vs Mean Perturbed Comparison",
        "Change in Model Fit",
        "Change in Logistic Model Fit",
    ):
        assert removed not in renderer


def test_original_mean_and_stability_render_inside_final_result_after_pipeline():
    final_renderer = _function_source(FRONTEND, "_render_final_analysis_result")
    tree = ast.parse(final_renderer)
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.append((node.lineno, node.func.id))

    original_line = min(line for line, name in calls if name == "show_result_table")
    noise_calls = [
        line for line, name in calls
        if "noise" in name.lower() and name.startswith("render")
    ]
    assert len(noise_calls) == 1
    assert original_line < noise_calls[0]
    pipeline_renderer = _function_source(FRONTEND,"show_pipeline_response")
    assert pipeline_renderer.index("show_pipeline_overview(response=response)") < pipeline_renderer.index("_render_final_analysis_result(response)") < pipeline_renderer.index("show_prism_privacy_result(response=response)")


def test_pipeline_overview_and_every_downstream_module_remain_in_original_order():
    renderer = _function_source(FRONTEND, "show_pipeline_response")
    expected = [
        "show_pipeline_overview(response=response)",
        "_render_module_separator()",
        "show_prism_privacy_result(response=response)",
        'render_llm_result(response.get("llm_result"),response=response)',
    ]
    offsets = [renderer.index(item) for item in expected]
    assert offsets == sorted(offsets)
    assert "Local execution diagnostics" in renderer


def test_all_six_dashboard_analyses_use_the_shared_noise_contract():
    for analysis in (
        '"table1"',
        '"table2"',
        '"figure1"',
        '"figure2"',
        '"correlation"',
        '"variance_analysis"',
    ):
        assert analysis in FRONTEND
    assert "NOISE_ENABLED_ANALYSES" in FRONTEND
    assert "def _has_current_noise_utility" in FRONTEND
