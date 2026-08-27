import ast
from pathlib import Path

import frontend


SOURCE = Path("frontend.py").read_text(encoding="utf-8")


def function_source(name):
    tree = ast.parse(SOURCE)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return ast.get_source_segment(SOURCE, node) or ""


def test_method_c_route_section_renders_exactly_four_ordered_cards():
    source = function_source("_render_method_c_route_examples")
    assert "Representative Examples by Method C Route" in source
    assert "enumerate(PRIVACY_ROUTE_ORDER, 1)" in source
    assert "Example {index} — Method C predicts" in source
    for text in ("User Query", "Expected Route", "Method A", "Method B", "Method C", "Risk Score", "Comparison Summary"):
        assert text in source
    assert "Contrast Level" not in source
    assert frontend._friendly_label("local_edge") == "Local Edge"


def test_local_details_fallback_is_read_only_and_model_free():
    source = function_source("_method_c_examples_with_local_fallback")
    assert "artifacts/privacy_methods_frontend60_details.jsonl" in source
    assert "build_method_c_route_examples" in source
    for forbidden in ("run_method_a", "run_method_b", "run_method_c", "prism_route", "st.button", "subprocess", "os.system"):
        assert forbidden not in source


def test_old_representative_section_is_not_rendered():
    source = function_source("_render_privacy_method_comparison_tab")
    assert "_render_method_c_route_examples(report)" in source
    assert "Representative Examples from the Frontend-Realistic Evaluation" not in source
