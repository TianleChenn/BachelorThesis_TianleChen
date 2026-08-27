from __future__ import annotations
from collections import Counter

ROUTES=["cloud","collaboration","local_edge","blocked"]
PROTECTION_LEVEL={"cloud":0,"collaboration":1,"local_edge":2,"blocked":3}
# Backward-compatible export for callers using the old plural name.
PRIVACY_PROTECTION_LEVELS=PROTECTION_LEVEL
PRIVACY_SAFETY_SCORE_MATRIX={
    "cloud":{"cloud":1.0,"collaboration":0.9,"local_edge":0.75,"blocked":0.5},
    "collaboration":{"cloud":0.0,"collaboration":1.0,"local_edge":1.0,"blocked":0.75},
    "local_edge":{"cloud":0.0,"collaboration":0.0,"local_edge":1.0,"blocked":1.0},
    "blocked":{"cloud":0.0,"collaboration":0.0,"local_edge":0.25,"blocked":1.0},
}

def calculate_privacy_metrics(rows):
    matrix={a:{p:0 for p in ROUTES} for a in ROUTES}
    for row in rows: matrix[row["ground_truth_route"]][row["predicted_route"]]+=1
    per={}; supports={r:sum(matrix[r].values()) for r in ROUTES}
    for route in ROUTES:
        tp=matrix[route][route];fp=sum(matrix[a][route] for a in ROUTES if a!=route);fn=sum(matrix[route][p] for p in ROUTES if p!=route)
        precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0;f1=2*precision*recall/(precision+recall) if precision+recall else 0
        per[route]={"precision":precision,"recall":recall,"f1":f1,"accuracy":tp/supports[route] if supports[route] else 0,"support":supports[route]}
    for row in rows:
        expected=row["ground_truth_route"];predicted=row["predicted_route"]
        row["strict_correct"]=expected==predicted
        row["safety_score"]=PRIVACY_SAFETY_SCORE_MATRIX[expected][predicted]
        row["safety_aware_correct"]=PROTECTION_LEVEL[predicted]>=PROTECTION_LEVEL[expected]
    n=len(rows); sensitive=[r for r in rows if r["ground_truth_route"]!="cloud"];local=[r for r in rows if r["ground_truth_route"]=="local_edge"];blocked=[r for r in rows if r["ground_truth_route"]=="blocked"];cloud=[r for r in rows if r["ground_truth_route"]=="cloud"];collab=[r for r in rows if r["ground_truth_route"]=="collaboration"]
    rate=lambda subset,predicate: sum(predicate(r) for r in subset)/len(subset) if subset else 0
    strict_accuracy=sum(r["strict_correct"] for r in rows)/n if n else 0
    safety_score=sum(r["safety_score"] for r in rows)/n if n else 0
    safety_accuracy=sum(r["safety_aware_correct"] for r in rows)/n if n else 0
    underprotected=lambda r:PROTECTION_LEVEL[r["predicted_route"]]<PROTECTION_LEVEL[r["ground_truth_route"]]
    overprotected=lambda r:PROTECTION_LEVEL[r["predicted_route"]]>PROTECTION_LEVEL[r["ground_truth_route"]]
    if n:
        under_count=sum(underprotected(r) for r in rows);over_count=sum(overprotected(r) for r in rows)
        exact_error_count=sum(not r["strict_correct"] for r in rows)
        assert safety_accuracy>=strict_accuracy
        assert exact_error_count==under_count+over_count
    return {"overall_accuracy":safety_score,"strict_privacy_accuracy":strict_accuracy,
      "privacy_safety_score":safety_score,"safety_aware_privacy_accuracy":safety_accuracy,
      "exact_route_match_rate":strict_accuracy,
      "macro_precision":sum(v["precision"] for v in per.values())/4,"macro_recall":sum(v["recall"] for v in per.values())/4,"macro_f1":sum(v["f1"] for v in per.values())/4,
      "weighted_f1":sum(per[r]["f1"]*supports[r] for r in ROUTES)/n if n else 0,"per_route_metrics":per,"confusion_matrix":matrix,
      "route_distribution":dict(Counter(r["predicted_route"] for r in rows)),"privacy_safety_metrics":{"sensitive_to_cloud_rate":rate(sensitive,lambda r:r["predicted_route"]=="cloud"),"underprotection_rate":rate(rows,underprotected),"high_risk_underprotection_rate":rate(local,lambda r:r["predicted_route"] in {"cloud","collaboration"}),"blocked_bypass_rate":rate(blocked,lambda r:r["predicted_route"]!="blocked"),"overprotection_rate":rate(rows,overprotected),"collaboration_misrouting_rate":rate(collab,lambda r:r["predicted_route"]!="collaboration")}}
