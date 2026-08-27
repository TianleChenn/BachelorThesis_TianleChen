import ast
from pathlib import Path


def test_complete_renderer_call_order():
    tree = ast.parse(Path("frontend.py").read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "show_pipeline_response"
    )
    calls = []
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {
                "show_pipeline_overview",
                "show_prism_privacy_result",
                "render_llm_result",
                "_render_final_analysis_result",
            }:
                calls.append((node.lineno, node.func.id))
    ordered = [name for _, name in sorted(calls)]
    assert ordered == [
        "show_pipeline_overview",
        "_render_final_analysis_result",
        "show_prism_privacy_result",
        "render_llm_result",
    ]
