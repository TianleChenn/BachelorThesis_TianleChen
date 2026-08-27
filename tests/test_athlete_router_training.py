import json

import numpy as np
import pytest
from unittest.mock import patch

from scripts.train_athlete_strong_weak_router import INPUT, load_training_rows, train_router


def test_default_training_source_is_v2_valid_preferences():
    assert INPUT.name == "athlete_router_preferences_v2_valid.json"


def test_legacy_calibration_schema_is_rejected(tmp_path):
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"results": []}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="v2_valid schema"):
        load_training_rows(legacy)


def test_training_uses_five_fold_out_of_fold_probabilities(tmp_path):
    rows = []
    for index in range(65):
        label = 1 if index < 44 else 0
        rows.append({
            "sample_id": f"p{index}", "prompt": f"Athlete aggregate question {index}",
            "requested_analysis": "table1", "difficulty": ("hard" if index % 3 == 0 else "easy"),
            "filters": {}, "requires_code": True, "privacy_route": "cloud",
            "official_mf_score": index / 100, "strong_score": 9 if label else 8,
            "weak_score": 8 if label else 9, "strong_valid": True, "weak_valid": True,
            "preference": "strong" if label else "weak", "status": "valid",
        })
    source = tmp_path / "labels.json"
    source.write_text(json.dumps({"dataset_name": "athlete_router_preferences_v2_valid",
                                  "results": rows}), encoding="utf-8")
    model, metadata = tmp_path / "router.joblib", tmp_path / "router.json"
    oof = np.column_stack([np.linspace(.9, .1, 65), np.linspace(.1, .9, 65)])
    with patch("scripts.train_athlete_strong_weak_router.cross_val_predict", return_value=oof) as predict:
        report = train_router(source, model, metadata)
    assert predict.call_args.kwargs["method"] == "predict_proba"
    assert predict.call_args.kwargs["cv"].n_splits == 5
    assert report["threshold_source"] == "stratified 5-fold out-of-fold probabilities"
    assert report["model_sha256"] and model.exists()
    assert report["router_type"] == "project_specific_preference_router"
    assert report["uses_official_mf_score"] is False
    assert report["strong_samples"] == 44 and report["weak_samples"] == 21
