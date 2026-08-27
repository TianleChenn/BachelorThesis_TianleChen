"""Audit overlap and warn when one feature almost perfectly predicts the route."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from privacy.llm_soft_gating_model import FEATURE_NAMES, ROUTE_TO_INDEX
from privacy.llm_soft_gating_workflow import load_reviewed_features

def audit(rows):
    x=np.asarray([row["features"] for row in rows]); y=np.asarray([ROUTE_TO_INDEX[row["route"]] for row in rows])
    ranges={route:{name:[float(x[y==index,j].min()),float(x[y==index,j].max())] for j,name in enumerate(FEATURE_NAMES)} for route,index in ROUTE_TO_INDEX.items()}
    feature_accuracy={}
    for j,name in enumerate(FEATURE_NAMES):
        classifier=DecisionTreeClassifier(max_depth=2,random_state=2026).fit(x[:,[j]],y)
        feature_accuracy[name]=float(classifier.score(x[:,[j]],y))
    overlap={name:all(max(ranges[route][name][0] for route in ROUTE_TO_INDEX)<=min(ranges[route][name][1] for route in ROUTE_TO_INDEX) for _ in [0]) for name in FEATURE_NAMES}
    warnings=[f"{name} alone reaches {score:.1%} training accuracy" for name,score in feature_accuracy.items() if score>=.95]
    return {"route_feature_ranges":ranges,"all_route_overlap":overlap,"single_feature_training_accuracy":feature_accuracy,"over_separable_warning":warnings}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset",default=str(PROJECT_ROOT/"evaluation/privacy_gating_train_4d_llm_generated.json")); args=parser.parse_args()
    print(json.dumps(audit(load_reviewed_features(args.dataset,approved_only=True)),indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
