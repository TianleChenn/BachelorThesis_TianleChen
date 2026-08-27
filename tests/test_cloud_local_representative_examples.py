import ast
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


def _sample(sample_id, ground_truth, prediction):
    return {
        "id": sample_id,
        "prompt": f"Request {sample_id}",
        "ground_truth": ground_truth,
        "prediction": prediction,
        "p_cloud": 0.5,
        "threshold": 0.6,
        "cloud": {"fully_correct": ground_truth == "cloud"},
        "local": {"fully_correct": ground_truth == "local"},
    }


def test_examples_are_derived_deterministically_and_invalid_rows_are_ignored():
    samples = [
        _sample("invalid-first", "invalid", "cloud"),
        _sample("cloud-first", "cloud", "cloud"),
        _sample("cloud-second", "cloud", "cloud"),
        _sample("local-first", "local", "local"),
        _sample("invalid-wrong", "invalid", "local"),
        _sample("incorrect-first", "cloud", "local"),
        _sample("incorrect-second", "local", "cloud"),
    ]

    first = frontend._select_cloud_local_representative_examples(samples)
    second = frontend._select_cloud_local_representative_examples(samples)

    assert first == second
    assert first["correct_cloud"]["id"] == "cloud-first"
    assert first["correct_cloud"]["ground_truth"] == "cloud"
    assert first["correct_cloud"]["prediction"] == "cloud"
    assert first["correct_local"]["id"] == "local-first"
    assert first["correct_local"]["ground_truth"] == "local"
    assert first["correct_local"]["prediction"] == "local"
    assert first["incorrect"]["id"] == "incorrect-first"
    assert first["incorrect"]["ground_truth"] in {"cloud", "local"}
    assert first["incorrect"]["prediction"] != first["incorrect"]["ground_truth"]
    assert all(row["ground_truth"] != "invalid" for row in first.values())


def test_frontend_falls_back_to_saved_per_sample_results_without_running_evaluation():
    renderer = _function_source("_render_cloud_local_evaluation_tab")
    assert 'independent.get("per_sample_results")' in renderer
    assert "_select_cost_router_explanation_examples" in renderer
    assert "Representative Cost-aware Routing Examples" in renderer
    assert "No matching saved independent evaluation example is available." in renderer
    assert "OOF example" not in renderer
    assert "Cloud Model Needed" in renderer
    assert "Local Model Is Sufficient" in renderer
    assert "Router Prediction Error" in renderer
    assert "summary_columns" not in renderer
    assert "Objective Code Validation" not in renderer
    for forbidden in (
        "subprocess",
        "os.system",
        "evaluate_athlete_cloud_local_router",
        "call_cloud_model",
        "call_local_model",
        "verify_and_execute_generated_code",
    ):
        assert forbidden not in renderer


def test_current_saved_artifact_provides_all_three_example_categories():
    report = frontend._load_json_artifact(
        "artifacts/athlete_cloud_local_router_evaluation.json"
    )
    selected = frontend._select_cloud_local_representative_examples(
        report.get("per_sample_results") or []
    )
    assert set(selected) == {"correct_cloud", "correct_local", "incorrect"}


def test_capability_examples_follow_exact_preferred_rules():
    samples = [
        {
            **_sample("invalid", "invalid", "local"),
            "cloud": {"fully_correct": True},
            "local": {"fully_correct": False},
        },
        {
            **_sample("cloud-needed", "cloud", "cloud"),
            "cloud": {"fully_correct": True},
            "local": {"fully_correct": False},
        },
        {
            **_sample("local-both-pass", "local", "local"),
            "cloud": {"fully_correct": True},
            "local": {"fully_correct": True},
        },
        {
            **_sample("under-route", "cloud", "local"),
            "cloud": {"fully_correct": True},
            "local": {"fully_correct": False},
        },
    ]
    first = frontend._select_cost_router_explanation_examples(samples)
    second = frontend._select_cost_router_explanation_examples(samples)
    assert first == second
    assert first["cloud_needed"]["id"] == "cloud-needed"
    assert first["local_sufficient"]["id"] == "local-both-pass"
    assert first["routing_error"]["id"] == "under-route"
    assert all(row["ground_truth"] != "invalid" for row in first.values())


def test_local_sufficient_falls_back_when_both_pass_is_unavailable():
    fallback = {
        **_sample("local-fallback", "local", "local"),
        "cloud": {"fully_correct": False},
        "local": {"fully_correct": True},
    }
    selected = frontend._select_cost_router_explanation_examples([fallback])
    assert selected["local_sufficient"] is fallback


def test_router_score_uses_saved_probability_and_threshold():
    example = {"p_cloud": 0.786504401818397, "threshold": 0.603939393939394}
    assert frontend._router_score_caption(example) == (
        "Router score: P_cloud = 0.7865 >= threshold = 0.6039"
    )
