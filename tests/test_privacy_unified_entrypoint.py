import json
from pathlib import Path

from llm.model_clients import ModelCallResult
from privacy import llm_privacy_assessor as assessor_module
from privacy import prism_router
from sports.service import _build_privacy_test


def assessment_json():
    return json.dumps({"privacy_risk_score":.18,"subject_scope":.06,
        "data_sensitivity":.14,"disclosure_level":.17,"analysis_type":"general_explanation",
        "blocked_request":False,"sensitive_categories":[],"explanation":"General explanation only.",
        "confidence":.96})


def test_decision_service_and_frontend_receive_same_assessment_json(monkeypatch):
    parsed=assessor_module.parse_privacy_assessment_response(assessment_json())
    calls=[]
    monkeypatch.setattr(prism_router,"assess_privacy_with_llm",lambda prompt,**kwargs: calls.append(prompt) or parsed)
    monkeypatch.setattr(prism_router,"trained_soft_gating_features",lambda values:
        {"cloud":.9,"collaboration":.05,"local_edge":.05})
    decision=prism_router.prism_route("Explain Table 2")
    privacy_test=_build_privacy_test("Explain Table 2",decision)
    assert calls == ["Explain Table 2"]
    assert privacy_test["llm_generated_json"] is decision.llm_privacy_assessment
    assert privacy_test["llm_generated_json"] == parsed.to_dict()
    assert privacy_test["assessment_success"] is True


def test_failed_results_are_never_cached(monkeypatch,tmp_path):
    monkeypatch.setattr(assessor_module,"CACHE_PATH",tmp_path/"cache.json")
    monkeypatch.setattr(assessor_module,"call_privacy_risk_model",lambda *args,**kwargs:
        ModelCallResult(None,"gpt-4.1",None,"mock",False,True,False,"offline"))
    result=assessor_module.assess_privacy_with_llm("request",use_cache=True)
    assert result.fallback_used is True
    assert not assessor_module.CACHE_PATH.exists()


def test_frontend_and_service_do_not_call_privacy_assessor_directly():
    frontend=Path("frontend.py").read_text(encoding="utf-8")
    service=Path("sports/service.py").read_text(encoding="utf-8")
    assert "assess_privacy_with_llm" not in frontend
    assert "assess_privacy_with_llm" not in service
    assert 'st.session_state["latest_analysis_result"] = response' in frontend


def test_both_diagnostics_use_public_entrypoint():
    direct=Path("scripts/test_privacy_risk_model.py").read_text(encoding="utf-8")
    router=Path("privacy/prism_router.py").read_text(encoding="utf-8")
    assert "from privacy.llm_privacy_assessor import assess_privacy_with_llm" in direct
    assert "assess_privacy_with_llm(prompt, use_cache=True)" in router
