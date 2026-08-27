import json
from types import SimpleNamespace
import pytest
from privacy.llm_minimal_4d_privacy_assessor import FEATURE_KEYS,MINIMAL_4D_SYSTEM_PROMPT,parse_minimal_4d_response
from scripts import evaluate_privacy_methods_frontend60 as formal

def test_method_b_prompt_is_deliberately_minimal():
    assert all(key in MINIMAL_4D_SYSTEM_PROMPT for key in FEATURE_KEYS)
    assert '"blocked_request"' in MINIMAL_4D_SYSTEM_PROMPT
    for forbidden in ('"analysis_type"','"sensitive_categories"','"confidence"','"explanation"'):
        assert forbidden not in MINIMAL_4D_SYSTEM_PROMPT
    assert "four numeric values" in MINIMAL_4D_SYSTEM_PROMPT
    assert "JSON boolean" in MINIMAL_4D_SYSTEM_PROMPT

def test_method_b_parser_requires_exact_four_values_and_boolean():
    values={**dict(zip(FEATURE_KEYS,(.123,.456,.789,.321))),"blocked_request":False}
    assert parse_minimal_4d_response(json.dumps(values)).to_dict()==values
    with pytest.raises(ValueError): parse_minimal_4d_response(json.dumps({**values,"explanation":"extra"}))
    with pytest.raises(ValueError): parse_minimal_4d_response(json.dumps({**values,"disclosure_level":1.1}))
    with pytest.raises(ValueError): parse_minimal_4d_response(json.dumps({key:value for key,value in values.items() if key!="blocked_request"}))
    with pytest.raises(ValueError): parse_minimal_4d_response(json.dumps({**values,"blocked_request":1}))

def test_method_b_parser_preserves_true_and_false_blocked_request():
    base=dict(zip(FEATURE_KEYS,(.98,1.0,.95,1.0)))
    assert parse_minimal_4d_response(json.dumps({**base,"blocked_request":True})).blocked_request is True
    assert parse_minimal_4d_response(json.dumps({**base,"blocked_request":False})).blocked_request is False

def test_formal_method_b_uses_new_router_and_separate_cache(monkeypatch):
    decision=SimpleNamespace(route="collaboration",success=True,error=None,features=[.1,.2,.3,.4],probabilities={"cloud":.2,"collaboration":.7,"local_edge":.1},gating_source="method_b_minimal_4d_shared_soft_gating")
    monkeypatch.setattr(formal,"route_with_method_b_4d_soft_gating",lambda _:decision)
    route,error,details=formal.run_method_b("aggregate athlete analysis")
    assert (route,error)==("collaboration",None)
    assert details=={"risk_score":.1,"recovered_from_assessor_cache":False}
    assert formal.CACHE_FILES["method_b_llm_scalar"]=="privacy_method_b_4d_frontend60_cache.jsonl"

def test_method_b_and_method_c_share_checkpoint_and_features(monkeypatch):
    import privacy.llm_method_b_4d_router as method_b
    import privacy.prism_router as method_c
    marker=object()
    monkeypatch.setattr(method_c,"get_active_prism_gater_path",lambda:marker)
    monkeypatch.setattr(method_b,"get_active_prism_gater_path",lambda:marker)
    assert method_b.get_method_b_gater_path() is marker
    assert method_b.FEATURE_NAMES==method_c.FEATURE_NAMES

def test_method_b_requires_no_separate_training_artifact():
    import inspect
    import privacy.llm_method_b_4d_router as method_b
    source=inspect.getsource(method_b)
    assert "prism_soft_gater_method_b" not in source
    assert "trained_soft_gating_features" in source

def _assessment(*, blocked_request, value=.9):
    return SimpleNamespace(
        privacy_risk_score=value, subject_scope=value, data_sensitivity=value,
        disclosure_level=value, blocked_request=blocked_request, success=True,
        error=None, cache_used=False)

def test_method_b_blocked_request_skips_soft_gating(monkeypatch):
    import privacy.llm_method_b_4d_router as method_b
    monkeypatch.setattr(method_b,"assess_privacy_minimal_4d",lambda _: _assessment(blocked_request=True))
    monkeypatch.setattr(method_b,"trained_soft_gating_features",
                        lambda _: (_ for _ in ()).throw(AssertionError("Soft Gating must be skipped")))
    decision=method_b.route_with_method_b_4d_soft_gating("export the raw records")
    assert decision.route=="blocked"
    assert decision.blocked_request is True
    assert decision.blocked is True
    assert decision.probabilities is None
    assert decision.features==[.9,.9,.9,.9]

def test_method_b_nonblocked_request_uses_soft_gating_argmax(monkeypatch):
    import privacy.llm_method_b_4d_router as method_b
    monkeypatch.setattr(method_b,"assess_privacy_minimal_4d",lambda _: _assessment(blocked_request=False,value=.2))
    monkeypatch.setattr(method_b,"trained_soft_gating_features",
                        lambda features:{"cloud":.1,"collaboration":.8,"local_edge":.1})
    decision=method_b.route_with_method_b_4d_soft_gating("aggregate analysis")
    assert decision.route=="collaboration"
    assert decision.blocked_request is False
    assert decision.probabilities["collaboration"]==.8

def test_method_b_high_features_do_not_override_nonblocked_cloud_top1(monkeypatch):
    import privacy.llm_method_b_4d_router as method_b
    monkeypatch.setattr(method_b,"assess_privacy_minimal_4d",lambda _: _assessment(blocked_request=False,value=.99))
    monkeypatch.setattr(method_b,"trained_soft_gating_features",
                        lambda features:{"cloud":.9,"collaboration":.05,"local_edge":.05})
    decision=method_b.route_with_method_b_4d_soft_gating("sensitive derived analysis")
    assert decision.features==[.99,.99,.99,.99]
    assert decision.route=="cloud"

def test_formal_loop_uses_each_sample_id_for_all_three_methods():
    import inspect
    source=inspect.getsource(formal.main)
    assert "for sample in samples" in source
    assert 'for method in a.methods' in source
    assert 'cached={"id":sample["id"]' in source
