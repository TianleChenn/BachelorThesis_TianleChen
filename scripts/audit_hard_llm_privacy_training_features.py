"""Audit hard LLM-generated features without changing review decisions or values."""
from __future__ import annotations
import argparse,json,math,sys
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from privacy.llm_soft_gating_model import FEATURE_NAMES,ROUTE_TO_INDEX
from privacy.llm_soft_gating_data import ACTIVE_LLM_4D_TRAINING_DATASET
DEFAULT=ACTIVE_LLM_4D_TRAINING_DATASET
def audit(path):
    payload=json.loads(path.read_text(encoding="utf-8-sig")); rows=payload.get("samples",[]); invalid=[]
    for i,row in enumerate(rows,1):
        values=row.get("features");
        if not isinstance(values,list) or len(values)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in (values or [])): invalid.append(i)
        if row.get("fallback_used"): invalid.append(i)
    prompts=[" ".join(row.get("prompt","").casefold().split()) for row in rows]; vectors=[tuple(row.get("features",[])) for row in rows]; counts=Counter(row.get("route") for row in rows)
    ranges={}; accuracy={}; warnings=[]
    if rows and not invalid:
        x=np.asarray([row["features"] for row in rows]); y=np.asarray([ROUTE_TO_INDEX[row["route"]] for row in rows])
        ranges={route:{name:[float(x[y==idx,j].min()),float(x[y==idx,j].max())] for j,name in enumerate(FEATURE_NAMES)} for route,idx in ROUTE_TO_INDEX.items()}
        for j,name in enumerate(FEATURE_NAMES):
            model=DecisionTreeClassifier(max_depth=2,random_state=2026).fit(x[:,[j]],y); accuracy[name]=float(model.score(x[:,[j]],y))
            if accuracy[name]>.90: warnings.append(f"{name}: The dataset may be overly separable by a single feature.")
    report={"sample_count":len(rows),"route_distribution":dict(counts),"invalid_samples":sorted(set(invalid)),"fallback_samples":sum(bool(row.get("fallback_used")) for row in rows),"duplicate_prompts":sum(c-1 for c in Counter(prompts).values() if c>1),"duplicate_vectors":sum(c-1 for c in Counter(vectors).values() if c>1),"route_feature_ranges":ranges,"single_feature_accuracy":accuracy,"warnings":warnings,"review_status":dict(Counter(row.get("review_status") for row in rows)),"balanced_90":len(rows)==90 and set(counts.values())=={30}}
    print(json.dumps(report,indent=2)); return report
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset",default=str(DEFAULT)); args=parser.parse_args(); audit(Path(args.dataset)); return 0
if __name__=="__main__": raise SystemExit(main())
