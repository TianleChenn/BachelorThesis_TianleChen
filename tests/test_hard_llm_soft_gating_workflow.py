import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from frontend import _format_probability_percent
from privacy import prism_router
from privacy.llm_privacy_assessor import PrivacyAssessmentResult
from privacy.llm_soft_gating_model import FEATURE_NAMES, LLMPrivacySoftGater
from scripts.generate_hard_llm_privacy_training_features import generate
from scripts.train_llm_privacy_soft_gater_4d_hard import DEFAULT_OUTPUT, choose_group_split
from scripts.create_hard_4d_prompt_dataset import build as build_prompt_dataset, validate as validate_prompt_dataset


def result(*, success=True, fallback=False):
    return PrivacyAssessmentResult(.43,.38,.69,.51,"correlation_analysis",False,
        ["MEDICAL"],"Assessed by the production rubric.",.88,"gpt-4.1","gpt-4.1",
        "mock",success,fallback,None if success else "offline")


def prompt_payload():
    routes=("cloud","collaboration","local_edge")
    return {"schema_version":"llm-4d-hard-prompts-v1","samples":[
        {"id":f"hard4d_{i+1:03d}","prompt":f"Hard athlete question {i+1}",
         "ground_truth_route":routes[i//30],"prompt_family":f"family_{i%9}"}
        for i in range(90)]}


def test_created_hard_prompt_dataset_is_balanced_and_feature_free():
    payload=build_prompt_dataset(); counts,duplicate_ids,duplicate_prompts,invalid,features=validate_prompt_dataset(payload)
    assert len(payload["samples"])==90
    assert counts=={"cloud":30,"collaboration":30,"local_edge":30}
    assert (duplicate_ids,duplicate_prompts,invalid,features)==(0,0,0,0)
    assert all(row["difficulty"]=="hard" and row["review_status"]=="pending" for row in payload["samples"])


def test_hard_generator_preserves_all_manual_routes_and_uses_assessor(tmp_path):
    source=tmp_path/"source.json"; output=tmp_path/"output.json"; errors=tmp_path/"errors.json"; source.write_text(json.dumps(prompt_payload()),encoding="utf-8")
    calls=[]; generated,failures=generate(source,output,errors,assessor=lambda prompt:calls.append(prompt) or result())
    assert len(calls)==len(generated["samples"])==90 and failures==[]
    assert [row["route"] for row in generated["samples"]]==[row["ground_truth_route"] for row in prompt_payload()["samples"]]
    assert all(len(row["features"])==4 and all(0<=value<=1 for value in row["features"]) for row in generated["samples"])


def test_hard_generator_excludes_fallback_and_resumes(tmp_path):
    source=tmp_path/"source.json"; output=tmp_path/"output.json"; errors=tmp_path/"errors.json"; source.write_text(json.dumps(prompt_payload()),encoding="utf-8")
    calls=[]
    generated,failures=generate(source,output,errors,assessor=lambda prompt:calls.append(prompt) or (result(fallback=True) if prompt.endswith("1") else result()))
    first_calls=len(calls); generate(source,output,errors,assessor=lambda prompt:calls.append(prompt) or result())
    assert len(generated["samples"])<90 and failures
    assert len(calls)==first_calls+len(failures)


def test_group_split_prevents_family_leakage_and_covers_routes():
    rows=[]
    for family in range(9):
        for route in ("cloud","collaboration","local_edge"):
            rows.append({"prompt_family":f"f{family}","route":route})
    train,validation,_=choose_group_split(rows,.2,2026)
    assert {rows[i]["prompt_family"] for i in train}.isdisjoint({rows[i]["prompt_family"] for i in validation})
    assert {rows[i]["route"] for i in validation}=={"cloud","collaboration","local_edge"}


def test_hard_checkpoint_path_cannot_overwrite_existing_models():
    assert DEFAULT_OUTPUT.name=="prism_soft_gater_4d_llm_hard.pt"


def test_probability_format_never_displays_negative_zero():
    assert _format_probability_percent(-0.0)=="0.0%"
    assert _format_probability_percent(-1e-14)=="0.0%"
    assert sum((.12,.81,.07))==pytest.approx(1.0)


def test_router_model_path_can_switch_by_configuration(tmp_path):
    checkpoint=tmp_path/"hard.pt"; torch.save({"model_state_dict":LLMPrivacySoftGater().state_dict(),"input_dim":4,"feature_names":FEATURE_NAMES},checkpoint)
    with patch.dict("os.environ",{"LLM_PRIVACY_GATER_MODEL_PATH":str(checkpoint)}):
        prism_router._ACTIVE_PRISM_GATER_CACHE=None; prism_router._ACTIVE_PRISM_GATER_CACHE_PATH=None
        assert prism_router.get_active_prism_gater_path()==checkpoint.resolve()
        assert prism_router.load_active_prism_gater().linear.in_features==4
