import json
from collections import Counter
from pathlib import Path
import pytest

from privacy.llm_soft_gating_data import (
    ACTIVE_LLM_4D_TRAINING_DATASET,
    HARD_PROMPT_DATASET,
    PROJECT_ROOT,
    load_llm_gating_dataset,
)
from scripts.audit_hard_llm_privacy_training_features import DEFAULT as AUDIT_DATASET
from scripts.evaluate_llm_privacy_soft_gater_4d import DEFAULT_DATASET as EVALUATION_DATASET
from scripts.train_llm_privacy_soft_gater_4d_hard import DEFAULT_DATASET as TRAINING_DATASET


def test_active_4d_scripts_use_hard_llm_generated_dataset():
    expected=(PROJECT_ROOT/"evaluation/privacy_gating_train_4d_hard_llm_generated_90.json").resolve()
    assert ACTIVE_LLM_4D_TRAINING_DATASET.resolve()==expected
    assert AUDIT_DATASET.resolve()==expected
    assert TRAINING_DATASET.resolve()==expected
    assert EVALUATION_DATASET.resolve()==expected
    assert expected.is_file()
    assert HARD_PROMPT_DATASET.is_file()


def test_legacy_schema_compatibility_uses_isolated_fixture(tmp_path):
    legacy=tmp_path/"legacy.json"
    legacy.write_text(json.dumps({"samples":[{
        "question":"Run Table 2 for all athletes.",
        "privacy_risk_score":.2,"subject_scope":.1,
        "data_sensitivity":.3,"disclosure_level":.4,
        "ground_truth_route":"cloud",
    }]}),encoding="utf-8")
    with pytest.warns(UserWarning, match="instead of the recommended 90"):
        rows,info=load_llm_gating_dataset(
            legacy,return_info=True,validate_distribution=False
        )
    assert rows[0]["question"]=="Run Table 2 for all athletes."
    assert rows[0]["ground_truth_route"]=="cloud"
    assert [rows[0][name] for name in (
        "privacy_risk_score","subject_scope","data_sensitivity","disclosure_level"
    )]==[.2,.1,.3,.4]
    assert info["field_validation"]=="passed"


def test_current_dataset_has_ninety_balanced_approved_samples():
    rows,info=load_llm_gating_dataset(
        ACTIVE_LLM_4D_TRAINING_DATASET,require_training_size=True,return_info=True
    )
    assert len(rows)==90
    assert Counter(row["ground_truth_route"] for row in rows)=={
        "cloud":30,"collaboration":30,"local_edge":30,
    }
    assert all(len(row["features"])==4 for row in rows)
    assert all(0<=float(value)<=1 for row in rows for value in row["features"])
    assert not any(row.get("fallback_used") for row in rows)
    assert all(row.get("review_status")=="approved" for row in rows)
    assert info["class_distribution"]=={"cloud":30,"collaboration":30,"local_edge":30}


def test_deleted_continuous_dataset_is_not_an_active_dependency():
    active_sources=[
        Path("privacy/llm_soft_gating_data.py"),
        Path("scripts/audit_hard_llm_privacy_training_features.py"),
        Path("scripts/train_llm_privacy_soft_gater_4d_hard.py"),
        Path("scripts/evaluate_llm_privacy_soft_gater_4d.py"),
    ]
    assert all("privacy_gating_train_4d_continuous.json" not in path.read_text(encoding="utf-8") for path in active_sources)
