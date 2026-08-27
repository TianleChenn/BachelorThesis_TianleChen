import json
from pathlib import Path

from llm.model_clients import ModelCallResult
from llm.model_config import get_strong_model_name
from privacy import llm_privacy_assessor as assessor
from privacy import prism_router


def valid_json():
    return json.dumps({"privacy_risk_score":.31,"subject_scope":.22,
        "data_sensitivity":.46,"disclosure_level":.41,"analysis_type":"descriptive_statistics",
        "blocked_request":False,"sensitive_categories":[],"explanation":"Moderate contextual risk.",
        "confidence":.9})


def test_privacy_assessor_uses_shared_strong_model_not_deprecated_alias(monkeypatch):
    monkeypatch.setenv("LLM_STRONG_MODEL","gpt-4.1")
    monkeypatch.setenv("PRIVACY_RISK_MODEL","wrong-model")
    monkeypatch.setenv("PRIVACY_RISK_CACHE_ENABLED","false")
    captured={}
    def fake(messages, **kwargs):
        captured["model"]=get_strong_model_name()
        return ModelCallResult(valid_json(),"gpt-4.1","gpt-4.1","mock",True,False,False,None)
    monkeypatch.setattr(assessor,"call_privacy_risk_model",fake)
    result=assessor.assess_privacy_with_llm("aggregate statistics")
    assert result.requested_model == captured["model"] == "gpt-4.1"
    assert result.success is True


def test_successful_shared_model_assessment_feeds_exact_four_values(monkeypatch):
    result=assessor.parse_privacy_assessment_response(valid_json())
    monkeypatch.setattr(prism_router,"assess_privacy_with_llm",lambda prompt,**kwargs:result)
    captured=[]
    monkeypatch.setattr(prism_router,"trained_soft_gating_features",lambda values: captured.append(values) or
        {"cloud":.8,"collaboration":.1,"local_edge":.1})
    prism_router.prism_route("aggregate statistics")
    assert captured == [[.31,.22,.46,.41]]


def test_api_failure_skips_gating_and_preserves_none_scores(monkeypatch):
    monkeypatch.setenv("PRIVACY_RISK_CACHE_ENABLED","false")
    monkeypatch.setattr(assessor,"call_privacy_risk_model",lambda *args,**kwargs:
        ModelCallResult(None,"gpt-4.1",None,"mock",False,True,False,"API unavailable"))
    result=assessor.assess_privacy_with_llm("request")
    monkeypatch.setattr(prism_router,"assess_privacy_with_llm",lambda prompt:result)
    decision=prism_router.prism_route("request")
    assert result.success is False and result.fallback_used is True
    assert decision.route == "local_edge" and decision.gating_skipped is True
    assert decision.gating_features is None and decision.probabilities is None
    assert all(result.to_dict()[name] is None for name in
        ["privacy_risk_score","subject_scope","data_sensitivity","disclosure_level"])


def test_frontend_and_source_use_shared_model_configuration():
    frontend=Path("frontend.py").read_text(encoding="utf-8")
    production="\n".join(Path(path).read_text(encoding="utf-8") for path in
        ["llm/model_clients.py","privacy/llm_privacy_assessor.py"])
    assert "LLM_STRONG_MODEL" in frontend
    assert "Final Privacy Decision" not in frontend
    assert "Assessment Technical Details" not in frontend
    assert "PRIVACY_RISK_MODEL" not in production
