from pathlib import Path

from frontend import _format_representative_generated_code


FRONTEND=(Path(__file__).resolve().parents[1]/"frontend.py").read_text(encoding="utf-8")


def test_evaluation_page_uses_quality_and_safety_aware_labels():
    assert "P_cloud" in FRONTEND
    assert "Independent Strong Usage" not in FRONTEND
    assert "Exact Route Accuracy" in FRONTEND
    assert "Safety-aware Accuracy" in FRONTEND
    comparison=FRONTEND[FRONTEND.index("def _render_privacy_method_comparison_tab():"):FRONTEND.index("def page_routing_evaluation():")]
    assert "Macro F1" in comparison
    assert "Privacy Evaluation" in FRONTEND
    assert "Hard Independent Sports Privacy Evaluation" not in FRONTEND
    assert "_render_privacy_evaluation_tab" in FRONTEND


def test_cloud_recall_local_recall_and_usage_share_one_metric_row():
    section = FRONTEND[
        FRONTEND.index("def _render_cloud_local_evaluation_tab") :
        FRONTEND.index("def _render_cloud_codegen_evaluation")
    ]
    assert "routing_metric_columns = st.columns(3)" in section
    assert 'routing_metric_columns[0].metric(\n        "Cloud Recall"' in section
    assert 'routing_metric_columns[1].metric(\n        "Local Recall"' in section
    assert 'routing_metric_columns[2].metric(\n        "Cloud Usage Rate"' in section


def test_representative_generated_code_is_pretty_printed_for_every_model():
    compact = (
        "result = analysis.table1(predictors=['muscular_strength', "
        "'lower_body_dynamics'], target='elite_status', controls=[[], ['sex']], "
        "filters={})"
    )

    formatted = _format_representative_generated_code(compact)

    assert formatted.startswith("result = analysis.table1(\n")
    assert "    predictors=['muscular_strength', 'lower_body_dynamics']," in formatted
    assert "    target='elite_status'," in formatted
    assert formatted.endswith("\n)")
    assert _format_representative_generated_code("No code returned") == "No code returned"
