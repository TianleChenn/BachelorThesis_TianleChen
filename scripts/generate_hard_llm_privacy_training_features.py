"""Resume-safe generation of hard-set features with the production assessor."""

from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from privacy.llm_privacy_assessor import ASSESSMENT_RULES_VERSION, assess_privacy_with_llm
from privacy.llm_soft_gating_model import FEATURE_NAMES, ROUTE_TO_INDEX
from privacy.llm_soft_gating_data import ACTIVE_LLM_4D_TRAINING_DATASET, HARD_PROMPT_DATASET

DEFAULT_INPUT=HARD_PROMPT_DATASET
DEFAULT_OUTPUT=ACTIVE_LLM_4D_TRAINING_DATASET
DEFAULT_ERRORS=PROJECT_ROOT/"artifacts/privacy_gating_hard_feature_generation_errors.json"

def resolve(value):
    path=Path(value); return (path if path.is_absolute() else PROJECT_ROOT/path).resolve()
def save(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix(".tmp.json"); temporary.write_text(json.dumps(payload,indent=2),encoding="utf-8"); temporary.replace(path)

def generate(input_path,output_path,error_path,*,assessor=assess_privacy_with_llm):
    if not input_path.is_file(): raise FileNotFoundError(f"Hard 4D prompt dataset not found: {input_path}")
    source=json.loads(input_path.read_text(encoding="utf-8-sig"))
    if source.get("schema_version")!="llm-4d-hard-prompts-v1" or len(source.get("samples",[]))!=90: raise ValueError("Hard prompt dataset must use llm-4d-hard-prompts-v1 and contain 90 samples")
    existing={}; model=None
    if output_path.exists():
        old=json.loads(output_path.read_text(encoding="utf-8-sig")); existing={row["id"]:row for row in old.get("samples",[])}; model=old.get("model")
    errors=[]
    if error_path.exists(): errors=json.loads(error_path.read_text(encoding="utf-8-sig")).get("errors",[])
    error_ids={row["id"] for row in errors}; total=len(source["samples"])
    for position,item in enumerate(source["samples"],1):
        identifier=str(item.get("id","")).strip(); route=item.get("ground_truth_route"); prompt=str(item.get("prompt","")).strip()
        if not identifier or not prompt or route not in ROUTE_TO_INDEX or not item.get("prompt_family"): raise ValueError(f"Invalid manually labelled row {position}")
        if identifier in existing: print(f"{position}/{total} resumed"); continue
        result=assessor(prompt)
        if result.success and not result.fallback_used:
            model=result.actual_model or result.requested_model
            existing[identifier]={"id":identifier,"prompt":prompt,"features":[float(getattr(result,name)) for name in FEATURE_NAMES],"route":route,"prompt_family":item["prompt_family"],"assessment_explanation":result.explanation,"assessment_confidence":result.confidence,"assessment_model":model,"review_status":"pending"}
            payload={"schema_version":"llm-generated-4d-hard-training-v1","rules_version":ASSESSMENT_RULES_VERSION,"model":model,"route_source":"manual_project_specific","feature_source":"active_llm_privacy_assessor","samples":list(existing.values())}; save(output_path,payload)
            errors=[row for row in errors if row["id"]!=identifier]; error_ids.discard(identifier)
        else:
            if identifier not in error_ids: errors.append({"id":identifier,"prompt":prompt,"error":result.error or "assessment failed","fallback_used":bool(result.fallback_used)})
            save(error_path,{"errors":errors})
        print(f"{position}/{total}")
    payload={"schema_version":"llm-generated-4d-hard-training-v1","rules_version":ASSESSMENT_RULES_VERSION,"model":model,"route_source":"manual_project_specific","feature_source":"active_llm_privacy_assessor","samples":list(existing.values())}; save(output_path,payload); save(error_path,{"errors":errors})
    counts=Counter(row["route"] for row in existing.values()); print(f"Success: {len(existing)}"); print(f"Failed: {len(errors)}"); print(f"Routes: {dict(counts)}"); return payload,errors

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--input",default=str(DEFAULT_INPUT)); parser.add_argument("--output",default=str(DEFAULT_OUTPUT)); parser.add_argument("--errors",default=str(DEFAULT_ERRORS)); args=parser.parse_args(); generate(resolve(args.input),resolve(args.output),resolve(args.errors)); return 0
if __name__=="__main__": raise SystemExit(main())
