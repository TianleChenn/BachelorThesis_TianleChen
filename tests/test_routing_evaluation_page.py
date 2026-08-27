import ast
from pathlib import Path


def _page_source():
    source = Path("frontend.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_render_cloud_local_evaluation_tab")
    return ast.get_source_segment(source, function) or ""


def test_cloud_local_evaluation_page_is_saved_result_only():
    source = _page_source()
    assert "st.button" not in source and "st.progress" not in source
    assert "athlete_cloud_local_router.json" in source
    assert "athlete_cloud_local_router_evaluation.json" in source
    assert "python scripts/train_athlete_cloud_local_router.py" in source
    for label in ("Cloud Recall", "Local Recall", "Cloud Usage Rate", "Routing Accuracy"):
        assert label in source
    for forbidden in ("AthleteStrongWeakRouter", "call_strong_model", "call_weak_model", "call_judge"):
        assert forbidden not in source
