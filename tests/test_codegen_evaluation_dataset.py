from pathlib import Path

from scripts.athlete_router_evaluation_common import load_frontend_benchmark


def test_restricted_codegen_evaluation_uses_shared_frontend_subset():
    metadata, rows = load_frontend_benchmark(
        Path("evaluation/frontend_realistic_benchmark_60.json")
    )
    assert metadata["dataset_name"] == "Frontend-Realistic LLM Evaluation"
    assert metadata["shared_benchmark_samples"] == 60
    assert metadata["eligible_llm_samples"] == len(rows) == 40
    assert metadata["route_distribution"] == {"cloud": 5, "collaboration": 35}
    assert len({row["id"] for row in rows}) == 40
    assert len({row["prompt"] for row in rows}) == 40
    assert all(row["llm_router_eligible"] is True for row in rows)
    assert all(row["privacy_route"] in {"cloud", "collaboration"} for row in rows)
