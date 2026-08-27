import json
from collections import Counter
from pathlib import Path

from scripts.collect_athlete_router_preferences_v2 import _summaries
from scripts.generate_athlete_router_training_prompts_v2 import build_samples, generate


def test_v2_candidates_are_unique_balanced_and_disjoint(tmp_path):
    samples = build_samples()
    assert len(samples) == len({row["prompt"].casefold() for row in samples}) == 100
    assert Counter(row["difficulty"] for row in samples) == {
        "simple": 40, "medium": 30, "hard": 30,
    }
    payload = generate(tmp_path / "training.json")
    assert payload["dataset_name"] == "athlete_router_training_prompts_v2_100"
    assert len(payload["samples"]) == 100


def test_preference_summary_keeps_every_real_result():
    samples = [
        {"id": "a"}, {"id": "b"}, {"id": "c"},
    ]
    rows = [
        {"sample_id": "a", "difficulty": "simple", "analysis_type": "correlation_analysis", "preference": "strong"},
        {"sample_id": "b", "difficulty": "medium", "analysis_type": "table1_analysis", "preference": "weak"},
        {"sample_id": "c", "difficulty": "hard", "analysis_type": "table2_analysis", "preference": "invalid_or_tie"},
    ]
    summary, valid = _summaries(samples, rows)
    assert summary["valid_preferences"] == 2
    assert summary["strong_wins"] == summary["weak_wins"] == 1
    assert summary["invalid_or_ties"] == 1
    assert len(valid) == 2


def test_real_saved_v2_artifacts_are_consistent_when_present():
    raw_path = Path("artifacts/athlete_router_preferences_v2_raw.json")
    valid_path = Path("artifacts/athlete_router_preferences_v2_valid.json")
    if not raw_path.exists() or not valid_path.exists():
        return
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    valid = json.loads(valid_path.read_text(encoding="utf-8"))
    assert len(raw["results"]) == 100
    assert len(valid["results"]) == raw["summary"]["valid_preferences"]
    assert all(row["preference"] in {"strong", "weak"} for row in valid["results"])
    assert raw["summary"]["valid_preferences"] + raw["summary"]["invalid_or_ties"] == 100
