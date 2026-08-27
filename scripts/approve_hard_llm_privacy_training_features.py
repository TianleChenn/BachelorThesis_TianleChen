"""List pending hard samples and apply only explicit human approval choices."""
from __future__ import annotations
import argparse,json
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]; DEFAULT=PROJECT_ROOT/"evaluation/privacy_gating_train_4d_hard_llm_generated_90.json"
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset",default=str(DEFAULT)); parser.add_argument("--approve-id",action="append",default=[]); parser.add_argument("--approve-all",action="store_true"); parser.add_argument("--confirm",default=""); args=parser.parse_args(); path=Path(args.dataset); payload=json.loads(path.read_text(encoding="utf-8-sig")); pending=[row for row in payload.get("samples",[]) if row.get("review_status")=="pending"]
    for row in pending: print(f"{row['id']} | {row['route']} | {row['prompt_family']}\n  {row['prompt']}\n  {row['features']}\n  {row.get('assessment_explanation','')}")
    selected=set(args.approve_id)
    if args.approve_all:
        if args.confirm!="APPROVE_ALL_REVIEWED": raise ValueError("--approve-all requires --confirm APPROVE_ALL_REVIEWED")
        selected={row["id"] for row in pending}
    if selected:
        unknown=selected-{row["id"] for row in pending}
        if unknown: raise ValueError(f"IDs are not pending: {sorted(unknown)}")
        for row in payload["samples"]:
            if row["id"] in selected: row["review_status"]="approved"
        path.write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(f"Approved: {len(selected)}")
    else: print("No statuses changed. Review first, then pass --approve-id or confirmed --approve-all.")
    return 0
if __name__=="__main__": raise SystemExit(main())
