from pathlib import Path

import pytest

import frontend


def test_evaluation_dashboard_keeps_privacy_second_and_adds_prompt_third():
    source = Path("frontend.py").read_text(encoding="utf-8")
    assert "_render_privacy_evaluation_tab" in source
    assert "Hard Independent Sports Privacy Evaluation" not in source
    page = source[source.index("def page_routing_evaluation"):source.index("def _clear_individual_analysis_on_group_change")]
    assert '["Cloud/Local Evaluation", "Privacy Evaluation", "Prompt Evaluation"]' in page
    assert "_render_cloud_local_evaluation_tab" in page
    assert "_render_privacy_evaluation_tab" in page
    assert "_render_prompt_evaluation_tab" in page
    assert page.index("_render_cloud_local_evaluation_tab") < page.index(
        "_render_privacy_evaluation_tab"
    ) < page.index("_render_prompt_evaluation_tab")


def test_privacy_prompt_chart_uses_saved_rates_and_weighted_combined_denominator():
    report = frontend._load_json_artifact(
        "evaluation/results/prompt_ablation/privacy_prompt_ablation_summary.json"
    )
    data, incomplete = frontend._privacy_prompt_chart_data(report)
    assert incomplete is False
    for prompt_key, prompt_name in frontend._PRIVACY_PROMPT_DISPLAY_NAMES.items():
        selected = data[data["Prompt"] == prompt_name].set_index("Benchmark")
        controlled = report["benchmarks"]["controlled"][prompt_key]
        independent = report["benchmarks"]["independent"][prompt_key]
        assert selected.loc["Controlled", "Exact Route Accuracy (%)"] == pytest.approx(
            controlled["exact_route_accuracy"] * 100
        )
        assert selected.loc["Independent", "Exact Route Accuracy (%)"] == pytest.approx(
            independent["exact_route_accuracy"] * 100
        )
        expected_combined = (
            round(controlled["exact_route_accuracy"] * controlled["completed_samples"])
            + round(independent["exact_route_accuracy"] * independent["completed_samples"])
        ) / (
            controlled["expected_samples"] + independent["expected_samples"]
        ) * 100
        assert selected.loc["Combined", "Exact Route Accuracy (%)"] == pytest.approx(
            expected_combined
        )


@pytest.mark.parametrize("artifact", [
    "evaluation/results/prompt_design_v2/codegen_prompt_design_v2_summary.json",
    "evaluation/results/local_prompt_design_v2/local_codegen_prompt_design_v2_summary.json",
])
def test_codegen_prompt_chart_uses_only_saved_five_stage_rates(artifact):
    report = frontend._load_json_artifact(artifact)
    data = frontend._codegen_prompt_chart_data(report)
    assert set(data["Prompt"]) == set(frontend._CODEGEN_PROMPT_DISPLAY_NAMES.values())
    assert list(data["Stage"].drop_duplicates()) == [
        stage for stage, _ in frontend._CODEGEN_PROMPT_STAGES
    ]
    for prompt_key, prompt_name in frontend._CODEGEN_PROMPT_DISPLAY_NAMES.items():
        selected = data[data["Prompt"] == prompt_name].set_index("Stage")
        for stage, field in frontend._CODEGEN_PROMPT_STAGES:
            assert selected.loc[stage, "Accuracy (%)"] == pytest.approx(
                report["prompts"][prompt_key][field] * 100
            )


def test_prompt_evaluation_renderer_is_read_only_and_has_missing_artifact_commands():
    source = Path("frontend.py").read_text(encoding="utf-8")
    renderer = source[
        source.index("def _render_prompt_evaluation_tab"):
        source.index("def page_routing_evaluation")
    ]
    for command in (
        "python scripts/evaluate_privacy_prompt_ablation.py --resume",
        "python scripts/evaluate_codegen_prompt_design_v2.py --resume",
        "python scripts/evaluate_local_codegen_prompt_design_v2.py --resume",
    ):
        assert command in renderer
    for forbidden in (
        "subprocess", "os.system", "evaluate_privacy_prompt(", "call_privacy_risk_model(",
        "raw_model_response", "raw_model_output", "checkpoint_rows",
    ):
        assert forbidden not in renderer
