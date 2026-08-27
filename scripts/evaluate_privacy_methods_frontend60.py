"""Formal comparison of fixed-rule, minimal-LLM, and privacy-prompt 4D routing."""
from __future__ import annotations
import argparse,hashlib,json,statistics,sys,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from privacy.evaluation_metrics import calculate_privacy_metrics
from scripts.evaluation_io import append_jsonl,load_json,project_path,read_jsonl,write_json

METHODS=("method_a_fixed_4d","method_b_llm_scalar","method_c_llm_4d_soft_gating")
FEATURE_NAMES=("privacy_risk_score","subject_scope","data_sensitivity","disclosure_level")
route_with_method_a_fixed_4d=None
route_with_method_b_4d_soft_gating=None
prism_route=None
CACHE_FILES={"method_a_fixed_4d":"privacy_method_a_4d_frontend60_cache.jsonl","method_b_llm_scalar":"privacy_method_b_4d_frontend60_cache.jsonl","method_c_llm_4d_soft_gating":"privacy_method_c_frontend60_cache.jsonl"}
EXPECTED_ROUTE_DISTRIBUTION={"cloud":5,"collaboration":35,"local_edge":10,"blocked":10}
REQUIRED_SAMPLE_FIELDS=("id","question","ground_truth_route")
METHOD_C_EXAMPLE_ROUTES=("cloud","collaboration","local_edge","blocked")
ROUTE_PROTECTION_LEVEL={route:index for index,route in enumerate(METHOD_C_EXAMPLE_ROUTES)}

def dataset_digest(data):
    clean={key:value for key,value in data.items() if key!="dataset_sha256"}
    encoded=json.dumps(clean,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def validate_inputs(dataset_path):
    data=load_json(dataset_path)
    if data.get("dataset_sha256")!=dataset_digest(data): raise ValueError("dataset SHA256 does not match its contents")
    if data.get("evaluation_status")!="formal": raise ValueError("formal evaluation_status required")
    if data.get("independent_evaluation") is not True: raise ValueError("independent evaluation dataset required")
    if data.get("used_for_training") is not False: raise ValueError("evaluation dataset must not be used for training")
    if data.get("used_for_threshold_calibration") is not False: raise ValueError("evaluation dataset must not be used for threshold calibration")
    if data.get("locked") is not True: raise ValueError("locked evaluation dataset required")
    samples=data.get("samples")
    if not isinstance(samples,list) or len(samples)!=60: raise ValueError("frontend-realistic benchmark must contain exactly 60 samples")
    for index,sample in enumerate(samples):
        if not isinstance(sample,dict) or any(field not in sample for field in REQUIRED_SAMPLE_FIELDS): raise ValueError(f"sample {index} is missing a required field")
        if sample["ground_truth_route"] not in EXPECTED_ROUTE_DISTRIBUTION: raise ValueError(f"sample {index} has an invalid ground_truth_route")
    distribution=dict(Counter(sample["ground_truth_route"] for sample in samples))
    if distribution!=EXPECTED_ROUTE_DISTRIBUTION: raise ValueError(f"unexpected route distribution: {distribution}")
    return data

def reset_jsonl(value):
    path=project_path(value); path.parent.mkdir(parents=True,exist_ok=True); path.write_text("",encoding="utf-8")
def run_method_a(question):
    global route_with_method_a_fixed_4d
    if route_with_method_a_fixed_4d is None:
        from privacy.method_a_fixed_4d_router import route_with_method_a_fixed_4d as implementation
        route_with_method_a_fixed_4d=implementation
    decision=route_with_method_a_fixed_4d(question)
    return decision.route,None,{"risk_score":decision.risk_score,"features":decision.features,
      "feature_names":list(FEATURE_NAMES),"gating_source":decision.gating_source,
      "recovered_from_assessor_cache":False}
def run_method_b(question):
    global route_with_method_b_4d_soft_gating
    if route_with_method_b_4d_soft_gating is None:
        from privacy.llm_method_b_4d_router import route_with_method_b_4d_soft_gating as implementation
        route_with_method_b_4d_soft_gating=implementation
    decision=route_with_method_b_4d_soft_gating(question)
    technical={"risk_score":decision.features[0] if decision.features else None,
      "recovered_from_assessor_cache":bool(getattr(decision,"assessment_cache_used",False))}
    return (decision.route,None,technical) if decision.success else (None,decision.error,technical)
def run_method_c(question):
    global prism_route
    if prism_route is None:
        from privacy.prism_router import prism_route as implementation
        prism_route=implementation
    decision=prism_route(question)
    technical={"risk_score":decision.risk_score,
      "recovered_from_assessor_cache":bool(getattr(decision,"privacy_assessment_cache_used",False))}
    if getattr(decision,"privacy_assessment_success",True) is False: return None,getattr(decision,"privacy_assessment_error","privacy assessment failed"),technical
    return decision.route,None,technical

def _normalize_route(value):
    normalized=str(value or "").strip().casefold().replace("-","_").replace(" ","_")
    aliases={"local":"local_edge","localedge":"local_edge"}
    normalized=aliases.get(normalized,normalized)
    return normalized if normalized in METHOD_C_EXAMPLE_ROUTES else None

def _method_prediction(record,method):
    raw=((record.get("predictions") or {}).get(method) or {}).get("predicted_route")
    return _normalize_route(raw)

def _contrast_level(record):
    method_c=_method_prediction(record,"method_c_llm_4d_soft_gating")
    differences=sum(_method_prediction(record,method)!=method_c for method in ("method_a_fixed_4d","method_b_llm_scalar"))
    return ("low","medium","high")[differences]

def _selection_priority(record,target_route):
    method_c=_method_prediction(record,"method_c_llm_4d_soft_gating")
    if method_c!=target_route: return None
    expected=_normalize_route(record.get("ground_truth_route"))
    method_a=_method_prediction(record,"method_a_fixed_4d")
    method_b=_method_prediction(record,"method_b_llm_scalar")
    method_c_correct=method_c==expected
    differences=sum(predicted!=method_c for predicted in (method_a,method_b))
    if method_c_correct and differences==2: return 1
    if method_c_correct and differences==1: return 2
    if method_c_correct: return 3
    return 4

def _selection_reason(priority,contrast):
    if priority==1: return f"Method C is correct and the methods provide {contrast} contrast."
    if priority==2: return "Method C is correct while another method misses the expected route."
    if priority==3: return "Method C is correct and all three methods agree."
    return "Selected as the deterministic fallback for this Method C route."

def _comparison_summary(record):
    expected=_normalize_route(record.get("ground_truth_route"))
    method_c=_method_prediction(record,"method_c_llm_4d_soft_gating")
    comparisons=[]
    for method,label in (("method_a_fixed_4d","Method A"),("method_b_llm_scalar","Method B")):
        predicted=_method_prediction(record,method)
        if predicted==method_c: continue
        if predicted not in ROUTE_PROTECTION_LEVEL or method_c not in ROUTE_PROTECTION_LEVEL:
            direction="different"
        elif ROUTE_PROTECTION_LEVEL[predicted]<ROUTE_PROTECTION_LEVEL[method_c]:
            direction="more permissive"
        else:
            direction="more protective"
        comparisons.append((label,direction))
    route_label=str(method_c or "unknown").replace("_"," ").title()
    expected_label=str(expected or "unknown").replace("_"," ").title()
    if method_c==expected:
        if not comparisons: return f"All three methods agree on the {route_label} route for this request."
        if len(comparisons)==2 and comparisons[0][1]==comparisons[1][1]:
            verb="differ" if comparisons[0][1]=="different" else f"are {comparisons[0][1]}"
            return f"Method C selects the correct {route_label} route, while Methods A and B {verb}."
        if len(comparisons)==2:
            first="differs" if comparisons[0][1]=="different" else f"is {comparisons[0][1]}"
            second="differs" if comparisons[1][1]=="different" else f"is {comparisons[1][1]}"
            return f"Method C selects the correct {route_label} route; {comparisons[0][0]} {first}, while {comparisons[1][0]} {second}."
        verb="differs" if comparisons[0][1]=="different" else f"is {comparisons[0][1]}"
        return f"Method C selects the correct {route_label} route, while {comparisons[0][0]} {verb}."
    if not comparisons: return f"All three methods select {route_label}, but the expected route is {expected_label}."
    labels=" and ".join(label for label,_ in comparisons)
    verb="provides" if len(comparisons)==1 else "provide"
    return f"Method C selects {route_label} instead of the expected {expected_label}; {labels} {verb} a contrasting prediction."

def _method_c_example_payload(record,priority):
    expected=_normalize_route(record.get("ground_truth_route"))
    predictions=record.get("predictions") or {}
    method_specs=(("method_a_fixed_4d","method_a"),("method_b_llm_scalar","method_b"),("method_c_llm_4d_soft_gating","method_c"))
    payload={"sample_id":record.get("id"),"user_query":record.get("question"),"ground_truth_route":expected}
    for method,prefix in method_specs:
        value=predictions.get(method) or {}; predicted=_normalize_route(value.get("predicted_route"))
        payload[f"{prefix}_prediction"]=predicted
        payload[f"{prefix}_correct"]=predicted==expected
        payload[f"{prefix}_score"]=value.get("risk_score")
    contrast=_contrast_level(record)
    payload["contrast_level"]=contrast
    payload["selection_reason"]=_selection_reason(priority,contrast)
    payload["comparison_summary"]=_comparison_summary(record)
    return payload

def build_method_c_route_examples(records):
    selected={}; used=set()
    for target_route in METHOD_C_EXAMPLE_ROUTES:
        candidates=[]
        for sample_index,record in enumerate(records):
            priority=_selection_priority(record,target_route)
            sample_id=record.get("id")
            if priority is None or sample_id in used: continue
            method_c=_method_prediction(record,"method_c_llm_4d_soft_gating")
            differences=sum(_method_prediction(record,method)!=method_c for method in ("method_a_fixed_4d","method_b_llm_scalar"))
            candidates.append(((priority,-differences,len(str(record.get("question") or "")),sample_index),record,priority))
        if not candidates:
            selected[target_route]={"missing":True,"message":"No Method C example found for this route."}
            continue
        _,record,priority=min(candidates,key=lambda item:item[0])
        used.add(record.get("id"))
        selected[target_route]=_method_c_example_payload(record,priority)
    return selected

def rebuild_method_c_route_examples(details_path,output_path):
    records=read_jsonl(details_path)
    examples=build_method_c_route_examples(records)
    report=load_json(output_path)
    report["method_c_route_examples"]=examples
    write_json(output_path,report)
    print(f"Representative examples rebuilt from {len(records)} existing records:")
    for route in METHOD_C_EXAMPLE_ROUTES:
        label=route.replace("_"," ").title()
        example=examples[route]
        print(f"\n{label}:")
        if example.get("missing"):
            print(f"  No Method C prediction of {route} exists in the {len(records)} completed records.")
            continue
        print(f"  sample = {example['sample_id']}")
        print(f"  expected = {example['ground_truth_route']}")
        print(f"  A = {example['method_a_prediction']}")
        print(f"  B = {example['method_b_prediction']}")
        print(f"  C = {example['method_c_prediction']}")
    return examples

def summarize(rows,failures):
    metric_rows=[{"ground_truth_route":x["ground_truth_route"],"predicted_route":x["predicted_route"]} for x in rows]
    metrics=calculate_privacy_metrics(metric_rows) if rows else None; safety=metrics["privacy_safety_metrics"] if metrics else {}; per=metrics["per_route_metrics"] if metrics else {}; latency=[x["latency_seconds"] for x in rows]
    return {"completed_count":len(rows),"failure_count":failures,"exact_route_accuracy":metrics["exact_route_match_rate"] if metrics else None,
      "safety_aware_accuracy":metrics["safety_aware_privacy_accuracy"] if metrics else None,"macro_precision":metrics["macro_precision"] if metrics else None,"macro_recall":metrics["macro_recall"] if metrics else None,
      "macro_f1":metrics["macro_f1"] if metrics else None,"weighted_f1":metrics["weighted_f1"] if metrics else None,
      **{k:safety.get(k) for k in ("underprotection_rate","overprotection_rate","sensitive_to_cloud_rate","blocked_bypass_rate","collaboration_misrouting_rate")},
      **{f"{r}_accuracy":per.get(r,{}).get("accuracy") for r in ("cloud","collaboration","local_edge","blocked")},
      "precision_per_route":{r:x["precision"] for r,x in per.items()},"recall_per_route":{r:x["recall"] for r,x in per.items()},"f1_per_route":{r:x["f1"] for r,x in per.items()},
      "confusion_matrix":metrics["confusion_matrix"] if metrics else {},"predicted_route_distribution":metrics["route_distribution"] if metrics else {},
      "average_latency":statistics.mean(latency) if latency else None,"median_latency":statistics.median(latency) if latency else None,"p95_latency":sorted(latency)[min(len(latency)-1,int(.95*len(latency)))] if latency else None}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",default="evaluation/frontend_realistic_benchmark_60.json")
    p.add_argument("--output",default="artifacts/privacy_methods_frontend60_comparison.json"); p.add_argument("--details",default="artifacts/privacy_methods_frontend60_details.jsonl")
    p.add_argument("--failures",default="artifacts/privacy_methods_frontend60_failures.jsonl"); p.add_argument("--cache-dir",default="artifacts"); p.add_argument("--limit",type=int); p.add_argument("--resume",action="store_true"); p.add_argument("--offline-cache",action="store_true"); p.add_argument("--methods",nargs="*",choices=METHODS,default=list(METHODS)); p.add_argument("--dry-run",action="store_true"); p.add_argument("--rebuild-examples",action="store_true"); a=p.parse_args()
    if a.rebuild_examples:
        rebuild_method_c_route_examples(a.details,a.output)
        return
    data=validate_inputs(a.dataset); samples=data["samples"][:a.limit] if a.limit else data["samples"]
    if a.dry_run: print(f"Dry run: {len(samples)} formal samples, methods={a.methods}; no model calls."); return
    caches={m:{x["id"]:x for x in read_jsonl(project_path(a.cache_dir)/CACHE_FILES[m])} if a.resume else {} for m in a.methods}
    if not a.resume:
        for method in a.methods: reset_jsonl(project_path(a.cache_dir)/CACHE_FILES[method])
    reset_jsonl(a.details); reset_jsonl(a.failures)
    details=[]; failures=Counter(); recovery=Counter()
    for sample in samples:
        question=sample["question"]; predictions={}
        for method in a.methods:
            cached=caches[method].get(sample["id"])
            if cached and cached.get("risk_score") is None:
                legacy=(cached.get("technical_details") or {}).get("privacy_risk_score")
                if legacy is not None: cached={**cached,"risk_score":legacy}
                else: cached=None
            technical={}
            risk_score=None
            if cached:
                route,error,latency=cached.get("predicted_route"),cached.get("error"),cached.get("latency_seconds",0)
                risk_score=cached.get("risk_score"); recovery["recovered_from_existing_cache"]+=1
            elif a.offline_cache: route,error,latency=None,"offline cache miss",0
            else:
                start=time.perf_counter()
                try:
                    if method=="method_a_fixed_4d": route,error,technical=run_method_a(question)
                    elif method=="method_b_llm_scalar": route,error,technical=run_method_b(question)
                    else: route,error,technical=run_method_c(question)
                except Exception as exc: route,error=None,f"{type(exc).__name__}: {exc}"
                latency=time.perf_counter()-start; risk_score=technical.get("risk_score")
                if technical.get("recovered_from_assessor_cache"): recovery["recovered_from_existing_cache"]+=1
                else: recovery["newly_evaluated"]+=1
                cached={"id":sample["id"],"predicted_route":route,"risk_score":risk_score,"error":error,"latency_seconds":latency}
                if method=="method_a_fixed_4d": cached["technical_details"]={key:technical.get(key) for key in ("risk_score","features","feature_names","gating_source")}
                append_jsonl(project_path(a.cache_dir)/CACHE_FILES[method],cached)
            if route is None or risk_score is None: recovery["failed"]+=1
            predictions[method]={"predicted_route":route,"risk_score":risk_score,"error":error,"latency_seconds":latency}
            if method=="method_a_fixed_4d": predictions[method]["technical_details"]=(cached.get("technical_details") or {})
        # Labels and provenance are accessed only after every selected method has predicted.
        expected=sample["ground_truth_route"]
        row={"id":sample["id"],"question":question,"ground_truth_route":expected,"predictions":predictions}; details.append(row); append_jsonl(a.details,row)
        for method,value in predictions.items():
            if value["predicted_route"] is None: failures[method]+=1; append_jsonl(a.failures,{"id":sample["id"],"method":method,"error":value["error"]})
    method_results={}
    for method in a.methods:
        completed=[{"ground_truth_route":x["ground_truth_route"],"predicted_route":x["predictions"][method]["predicted_route"],"latency_seconds":x["predictions"][method]["latency_seconds"]} for x in details if x["predictions"][method]["predicted_route"] is not None]
        method_results[method]=summarize(completed,failures[method])
    common=[x for x in details if all(x["predictions"][m]["predicted_route"] is not None for m in a.methods)]
    common_results={m:summarize([{"ground_truth_route":x["ground_truth_route"],"predicted_route":x["predictions"][m]["predicted_route"],"latency_seconds":x["predictions"][m]["latency_seconds"]} for x in common],0) for m in a.methods}
    method_c_route_examples=build_method_c_route_examples(details)
    result={"status":"formal_comparison","dataset_name":data["dataset_name"],"dataset_sha256":data["dataset_sha256"],"dataset_annotation":{"annotation_type":data.get("annotation_type"),
      "generator_model":data.get("generator_model"),"verifier_model":data.get("verifier_model")},"sample_count":len(samples),
      "route_distribution":dict(Counter(x["ground_truth_route"] for x in samples)),"methods":method_results,"common_completed_subset":{"sample_count":len(common),"methods":common_results},
      "method_c_route_examples":method_c_route_examples,"created_at":datetime.now(timezone.utc).isoformat()}
    write_json(a.output,result)
    print(f"Recovered from existing cache: {recovery['recovered_from_existing_cache']}")
    print(f"Newly evaluated: {recovery['newly_evaluated']}")
    print(f"Failed: {recovery['failed']}")
    print("Method                         Exact   Safety   Macro F1   Under   Over")
    for name,m in method_results.items(): print(f"{name:30} {m['exact_route_accuracy'] or 0:.3f}   {m['safety_aware_accuracy'] or 0:.3f}   {m['macro_f1'] or 0:.3f}   {m['underprotection_rate'] or 0:.3f}   {m['overprotection_rate'] or 0:.3f}")
if __name__=="__main__": main()
