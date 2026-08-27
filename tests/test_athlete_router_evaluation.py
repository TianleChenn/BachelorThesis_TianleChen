import json
from unittest.mock import patch

import pytest

from scripts.evaluate_athlete_strong_weak_router import evaluate


class OfflineRouter:
    def __init__(self, metadata_path):
        self.metadata_path = metadata_path

    def predict(self, prompt, **kwargs):
        probability = .8 if prompt.startswith("strong") else .2
        return {"p_strong": probability, "threshold": .5,
                "selected_tier": "strong" if probability >= .5 else "weak"}


def test_evaluation_uses_40_saved_independent_references(tmp_path):
    samples = []
    details = []
    for index in range(40):
        truth = "strong" if index < 20 else "weak"
        prompt = f"{truth} request {index}"
        samples.append({"id": f"i{index}", "prompt": prompt,
                        "privacy_route": "cloud" if index < 5 else "collaboration"})
        details.append({"prompt_id": f"i{index}", "prompt": prompt, "status": "valid",
                        "strong_score": 9 if truth == "strong" else 7,
                        "weak_score": 7 if truth == "strong" else 9,
                        "judge_label": f"{truth}_win"})
    metadata = {"dataset_name": "Frontend-Realistic LLM Evaluation",
                "dataset_path": "evaluation/frontend_realistic_benchmark_60.json",
                "shared_benchmark_samples": 60, "eligible_llm_samples": 40}
    details_path = tmp_path / "details.json"
    details_path.write_text(json.dumps({"results": details}), encoding="utf-8")
    metadata_path = tmp_path / "router.json"
    metadata_path.write_text(json.dumps({"training_samples": 65}), encoding="utf-8")
    with patch("scripts.evaluate_athlete_strong_weak_router.load_frontend_benchmark",
               return_value=(metadata, samples)):
        report = evaluate(tmp_path / "dataset.json", details_path,
                          tmp_path / "evaluation.json", OfflineRouter(metadata_path))
    assert report["valid_samples"] == 40
    assert report["training_samples"] == 65 and report["evaluation_samples"] == 40
    assert report["accuracy"] == report["balanced_accuracy"] == 1.0
    assert report["strong_recall"] == report["weak_recall"] == 1.0
    assert report["strong_usage_rate"] == pytest.approx(.5)
    assert report["uses_official_mf_score"] is False
    assert report["representative_examples"]["correct_strong"] is not None
    assert report["representative_examples"]["weak_example"] is not None
