import json
from pathlib import Path

from scripts import evaluate_athlete_strong_weak_router as athlete_evaluator
from scripts.athlete_router_evaluation_common import load_frontend_benchmark


def test_shared_benchmark_maps_exactly_40_eligible_requests():
    metadata, samples = load_frontend_benchmark(
        Path("evaluation/frontend_realistic_benchmark_60.json")
    )
    assert metadata["shared_benchmark_samples"] == 60
    assert metadata["eligible_llm_samples"] == len(samples) == 40
    assert metadata["route_distribution"] == {"cloud": 5, "collaboration": 35}
    assert {row["privacy_route"] for row in samples} == {"cloud", "collaboration"}
    assert all(row["ground_truth_route"] not in {"local_edge", "blocked"} for row in samples)
    assert len({row["prompt"] for row in samples}) == 40


def test_current_independent_artifact_defaults_use_the_shared_benchmark():
    assert athlete_evaluator.DATASET.name == "frontend_realistic_benchmark_60.json"
    assert athlete_evaluator.DETAILS.name == "routellm_frontend40_results.json"
    assert athlete_evaluator.OUTPUT.name == "athlete_strong_weak_router_evaluation.json"


def test_trained_router_metadata_remains_fixed():
    metadata = json.loads(Path("artifacts/athlete_strong_weak_router.json").read_text(encoding="utf-8"))
    assert metadata["training_samples"] == 94
    assert metadata["strong_samples"] == 26
    assert metadata["weak_samples"] == 68
    assert metadata["training_dataset"] == "artifacts/athlete_router_preferences_v2_valid.json"
    assert metadata["router_type"] == "project_specific_preference_router"
    assert metadata["uses_official_mf_score"] is False
    assert .01 <= metadata["threshold"] <= .99
