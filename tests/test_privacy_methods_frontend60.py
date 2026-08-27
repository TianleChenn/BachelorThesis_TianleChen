import inspect,json,pytest
import scripts.evaluate_privacy_methods_frontend60 as module

def test_formal_evaluator_calls_existing_methods():
    assert "route_with_method_a_fixed_4d" in inspect.getsource(module.run_method_a)
    assert "route_with_method_b_4d_soft_gating" in inspect.getsource(module.run_method_b)
    assert "prism_route" in inspect.getsource(module.run_method_c)
def test_no_recalibration_or_training():
    source=inspect.getsource(module)
    for forbidden in ("select_thresholds(","optimizer","backward(","train_llm"):
        assert forbidden not in source
def test_predictions_happen_before_label_read():
    source=inspect.getsource(module.main)
    assert source.index("predictions[method]") < source.index('expected=sample["ground_truth_route"]')
def test_method_failure_is_none(monkeypatch):
    class Decision:
        success=False; route="local_edge"; error="failed"; features=None; probabilities=None; gating_source="failed"
    monkeypatch.setattr(module,"route_with_method_b_4d_soft_gating",lambda *a,**k:Decision())
    assert module.run_method_b("question")== (None,"failed",{"risk_score":None,"recovered_from_assessor_cache":False})

def test_each_method_saves_its_actual_risk_score(monkeypatch):
    class A: route="cloud"; risk_score=.18; features=[.18,.15,.15,.2]; gating_source="fixed"
    class B: success=True; route="collaboration"; error=None; features=[.43,.2,.3,.4]
    class C: route="cloud"; risk_score=.16; privacy_assessment_success=True
    monkeypatch.setattr(module,"route_with_method_a_fixed_4d",lambda _:A())
    monkeypatch.setattr(module,"route_with_method_b_4d_soft_gating",lambda _:B())
    monkeypatch.setattr(module,"prism_route",lambda _:C())
    assert module.run_method_a("q")[2]["risk_score"]==.18
    assert module.run_method_b("q")[2]["risk_score"]==.43
    assert module.run_method_c("q")[2]["risk_score"]==.16
def test_common_completed_subset_metrics():
    rows=[{"ground_truth_route":"cloud","predicted_route":"cloud","latency_seconds":.1}]
    assert module.summarize(rows,0)["exact_route_accuracy"]==1
def valid_dataset():
    routes=["cloud"]*5+["collaboration"]*35+["local_edge"]*10+["blocked"]*10
    data={"evaluation_status":"formal","independent_evaluation":True,"used_for_training":False,
      "used_for_threshold_calibration":False,"locked":True,
      "samples":[{"id":f"sample_{index:03d}","question":f"question {index}","ground_truth_route":route} for index,route in enumerate(routes,1)]}
    data["dataset_sha256"]=module.dataset_digest(data)
    return data

def test_validates_frontend60_dataset_without_external_audit(monkeypatch):
    data=valid_dataset(); monkeypatch.setattr(module,"load_json",lambda path:data)
    assert module.validate_inputs("dataset") is data

@pytest.mark.parametrize("field,value",[
    ("evaluation_status","draft"),("independent_evaluation",False),
    ("used_for_training",True),("used_for_threshold_calibration",True),("locked",False),
])
def test_rejects_invalid_formal_metadata(monkeypatch,field,value):
    data=valid_dataset(); data[field]=value; data["dataset_sha256"]=module.dataset_digest(data)
    monkeypatch.setattr(module,"load_json",lambda path:data)
    with pytest.raises(ValueError): module.validate_inputs("dataset")

def test_rejects_digest_or_route_distribution_mismatch(monkeypatch):
    data=valid_dataset(); data["dataset_sha256"]="bad"
    monkeypatch.setattr(module,"load_json",lambda path:data)
    with pytest.raises(ValueError): module.validate_inputs("dataset")
    data=valid_dataset(); data["samples"][0]["ground_truth_route"]="collaboration"; data["dataset_sha256"]=module.dataset_digest(data)
    with pytest.raises(ValueError): module.validate_inputs("dataset")

def detail(sample_id,question,expected,method_a,method_b,method_c):
    return {"id":sample_id,"question":question,"ground_truth_route":expected,"predictions":{
      "method_a_fixed_4d":{"predicted_route":method_a,"risk_score":.1},
      "method_b_llm_scalar":{"predicted_route":method_b,"risk_score":.2},
      "method_c_llm_4d_soft_gating":{"predicted_route":method_c,"risk_score":.3}}}

def test_method_c_examples_normalize_routes_and_prefer_contrast():
    rows=[
      detail("cloud_low","short","Cloud","Cloud","Cloud","Cloud"),
      detail("cloud_medium","longer contrasting query","cloud","Local Edge","cloud","cloud"),
      detail("collaboration_high","filtered cohort","Collaboration","Cloud","Local Edge","Collaboration"),
      detail("local_low","profile","local-edge","local","Local Edge","Local Edge"),
      detail("blocked_fallback","raw request","local_edge","cloud","collaboration","Blocked"),
    ]
    examples=module.build_method_c_route_examples(rows)
    assert list(examples)==["cloud","collaboration","local_edge","blocked"]
    assert examples["cloud"]["sample_id"]=="cloud_medium"
    assert examples["collaboration"]["contrast_level"]=="high"
    assert examples["local_edge"]["method_c_prediction"]=="local_edge"
    assert examples["blocked"]["method_c_correct"] is False

def test_method_c_examples_only_missing_when_prediction_is_absent():
    examples=module.build_method_c_route_examples([])
    assert all(value=={"missing":True,"message":"No Method C example found for this route."} for value in examples.values())
