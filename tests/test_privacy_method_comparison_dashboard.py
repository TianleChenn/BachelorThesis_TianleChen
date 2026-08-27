import ast
from pathlib import Path
import frontend

SOURCE=Path("frontend.py").read_text(encoding="utf-8")
def function_source(name):
    tree=ast.parse(SOURCE); node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
    return ast.get_source_segment(SOURCE,node) or ""

def test_two_evaluation_tabs_and_artifact_path():
    page=function_source("page_routing_evaluation")
    assert all(label in page for label in ("Cloud/Local Evaluation","Privacy Evaluation"))
    assert "Privacy Method Comparison" not in page
    assert "_render_cloud_local_evaluation_tab" in page
    assert "_render_privacy_evaluation_tab" in page
    cloud_local=function_source("_render_cloud_local_evaluation_tab")
    assert 'artifacts/athlete_cloud_local_router_evaluation.json' in cloud_local
    assert "Cost-aware Routing" in cloud_local
    assert "LLM ROUTER" not in cloud_local
    assert "Training source:" not in cloud_local
    assert "Router artifact modified time:" not in cloud_local
    assert "Evaluation artifact modified time:" not in cloud_local
    assert frontend.PRIVACY_METHOD_COMPARISON_PATH=="artifacts/privacy_methods_frontend60_comparison.json"

def test_comparison_is_saved_result_only():
    source=function_source("_render_privacy_method_comparison_tab")
    assert "_load_json_artifact" in source
    for forbidden in ("st.button","subprocess","os.system","run_evaluation","call_privacy","call_strong_model","call_weak_model","call_judge","route_with_llm_scalar_baseline","prism_route"):
        assert forbidden not in source
    assert "python scripts/evaluate_privacy_methods_frontend60.py --resume" in source

def test_comparison_labels_and_method_names():
    source=function_source("_render_privacy_method_comparison_tab")
    for label in ("Exact Route Accuracy","Safety-aware Accuracy","Macro F1","Overprotection Rate"):
        assert label in source
    assert "Underprotection" not in source
    assert frontend.PRIVACY_METHOD_DISPLAY_NAMES=={
      "method_a_fixed_4d":"Fixed Rules + Soft Gating",
      "method_b_llm_scalar":"Simple LLM + Soft Gating",
      "method_c_llm_4d_soft_gating":"Privacy Prompt LLM + Soft Gating"}

def test_common_subset_preferred_and_missing_not_zero():
    report={"methods":{"method_a_fixed_4d":{"metrics":{"macro_f1":.1}}},"common_completed_subset":{"methods":{"method_a_fixed_4d":{"macro_f1":.9}}}}
    payload,metrics=frontend._get_comparison_method_metrics(report,"method_a_fixed_4d")
    assert metrics["macro_f1"]==.9
    assert frontend._format_percentage(None)=="Not available"
    assert frontend._format_percentage("bad")=="Not available"

def test_formal_validation_and_route_order_present():
    source=function_source("_render_privacy_method_comparison_tab")
    assert 'report.get("status") == "formal_comparison"' in source
    assert 'report.get("sample_count") == 60' in source
    for route,count in (("cloud",5),("collaboration",35),("local_edge",10),("blocked",10)):
        assert f'"{route}": {count}' in source
    assert frontend.PRIVACY_ROUTE_ORDER==("cloud","collaboration","local_edge","blocked")
    assert "The saved comparison is incomplete" in source

def test_top_summary_keeps_only_evaluation_samples_and_dataset():
    source=function_source("_render_privacy_method_comparison_tab")
    summary=source[source.index('st.success("Formal three-method comparison result loaded successfully.")'):source.index("annotation = report.get", source.index("st.success"))]
    assert "Common Completed Samples" not in summary
    assert "route_columns" not in summary
    assert 'st.columns(2)' in summary
    assert 'metric("Evaluation Samples"' in summary
    assert 'metric("Evaluation Dataset"' in summary
    assert 'metric("Blocked"' not in summary

def test_obsolete_method_b_thresholds_are_not_displayed():
    source=function_source("_render_privacy_method_comparison_tab")
    for label in ("Method B Calibration Information","cloud_to_collaboration","collaboration_to_local_edge","local_edge_to_blocked"):
        assert label not in source
    assert "calibrated on a separate calibration dataset" not in source

def test_representative_examples_do_not_remove_main_comparison():
    source=function_source("_render_privacy_method_comparison_tab")
    assert "#### Main Comparison" in source
    assert '"Method": short_name' in source
    assert 'short_names = ("Method A", "Method B", "Method C")' in source
    assert "_render_method_c_route_examples(report)" in source
    assert "#### Representative Examples from the Frontend-Realistic Evaluation" not in source
    assert "50 frontend-realistic requests plus 10 additional " in source
    assert source.index("#### Main Comparison") < source.index("_render_method_c_route_examples(report)") < source.index("Per-route Accuracy Details")
