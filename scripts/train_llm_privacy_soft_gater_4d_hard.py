"""Train a separate hard-set linear gater with prompt-family group isolation."""
from __future__ import annotations
import argparse,copy,json,math,random,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from privacy.llm_privacy_assessor import ASSESSMENT_RULES_VERSION
from privacy.llm_soft_gating_model import FEATURE_NAMES,INDEX_TO_ROUTE,ROUTE_TO_INDEX,LLMPrivacySoftGater
from privacy.llm_soft_gating_workflow import load_reviewed_features
from privacy.llm_soft_gating_data import ACTIVE_LLM_4D_TRAINING_DATASET
DEFAULT_DATASET=ACTIVE_LLM_4D_TRAINING_DATASET; DEFAULT_OUTPUT=PROJECT_ROOT/"artifacts/prism_soft_gater_4d_llm_hard.pt"; DEFAULT_METADATA=PROJECT_ROOT/"artifacts/prism_soft_gater_4d_llm_hard_metadata.json"
def choose_group_split(rows,ratio=.2,seed=2026):
    labels=np.asarray([ROUTE_TO_INDEX[r["route"]] for r in rows]); groups=np.asarray([r["prompt_family"] for r in rows]); best=None
    for candidate in range(seed,seed+100):
        train,val=next(GroupShuffleSplit(n_splits=1,test_size=ratio,random_state=candidate).split(np.arange(len(rows)),labels,groups))
        if set(labels[train])!=set(range(3)) or set(labels[val])!=set(range(3)): continue
        score=(min(Counter(labels[val]).values()),-abs(len(val)/len(rows)-ratio))
        if best is None or score>best[0]: best=(score,train,val,candidate)
    if best is None: raise ValueError("No prompt-family split covers all three routes")
    _,train,val,chosen=best
    if set(groups[train])&set(groups[val]): raise RuntimeError("Prompt-family leakage")
    return train,val,chosen
def calculate(model,x,y):
    with torch.no_grad(): logits=model(x); probs=torch.softmax(logits,1); pred=logits.argmax(1)
    truth=y.numpy(); guessed=pred.numpy(); precision,recall,f1,_=precision_recall_fscore_support(truth,guessed,labels=range(3),average="macro",zero_division=0); per=precision_recall_fscore_support(truth,guessed,labels=range(3),average=None,zero_division=0)[1]; matrix=confusion_matrix(truth,guessed,labels=range(3)); entropy=-(probs*torch.log(probs.clamp_min(1e-12))).sum(1).mean()
    return {"accuracy":float((pred==y).float().mean()),"macro_precision":float(precision),"macro_recall":float(recall),"macro_f1":float(f1),"per_class_recall":{INDEX_TO_ROUTE[i]:float(v) for i,v in enumerate(per)},"confusion_matrix":matrix.tolist(),"logits_range":[float(logits.min()),float(logits.max())],"probability_entropy":float(entropy)}
def train(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed); path=Path(args.dataset); payload=json.loads(path.read_text(encoding="utf-8-sig"))
    if args.allow_pending:
        rows=[row for row in payload.get("samples",[]) if row.get("review_status") in {"approved","pending"}]
        if not rows: raise ValueError("No approved or pending samples")
    else: rows=load_reviewed_features(path,approved_only=True)
    train_idx,val_idx,split_seed=choose_group_split(rows,args.validation_ratio,args.seed); x=torch.tensor([r["features"] for r in rows],dtype=torch.float32);y=torch.tensor([ROUTE_TO_INDEX[r["route"]] for r in rows]);model=LLMPrivacySoftGater();opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay);loss_fn=torch.nn.CrossEntropyLoss();best=None;best_f1=-1.;best_epoch=0;wait=0
    for epoch in range(1,args.epochs+1):
        model.train();opt.zero_grad();loss=loss_fn(model(x[train_idx]),y[train_idx]);loss.backward();opt.step();current=calculate(model.eval(),x[val_idx],y[val_idx])["macro_f1"]
        if current>best_f1+1e-9: best_f1=current;best=copy.deepcopy(model.state_dict());best_epoch=epoch;wait=0
        else: wait+=1
        # A linear three-way boundary needs a short warm-up before patience is
        # allowed to stop training; otherwise the middle route can remain
        # completely unlearned while epoch 1 is incorrectly retained.
        if epoch>=args.minimum_epochs and wait>=args.patience: break
    model.load_state_dict(best);model.eval();training=calculate(model,x[train_idx],y[train_idx]);validation=calculate(model,x[val_idx],y[val_idx]);output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);torch.save({"model_state_dict":model.state_dict(),"input_dim":4,"feature_names":FEATURE_NAMES,"routes":list(ROUTE_TO_INDEX),"rules_version":ASSESSMENT_RULES_VERSION,"training_stage":"hard_llm_generated_reviewed"},output)
    metadata={"model_name":"LLM-based 4D Soft Gating","training_dataset":"privacy_gating_train_4d_hard_llm_generated_90.json","features_source":"GPT-4.1 Privacy Assessor","group_split":True,"independent_evaluation":False,"approved_only":not args.allow_pending,"sample_count":len(rows),"route_distribution":dict(Counter(r["route"] for r in rows)),"split_seed":split_seed,"training_families":sorted({rows[i]["prompt_family"] for i in train_idx}),"validation_families":sorted({rows[i]["prompt_family"] for i in val_idx}),"best_epoch":best_epoch,"training_metrics":training,"validation_metrics":validation,"training_timestamp":datetime.now(timezone.utc).isoformat()};Path(args.metadata_output).write_text(json.dumps(metadata,indent=2),encoding="utf-8");print(json.dumps(metadata,indent=2));return metadata
def parser():
    p=argparse.ArgumentParser();p.add_argument("--dataset",default=str(DEFAULT_DATASET));p.add_argument("--output",default=str(DEFAULT_OUTPUT));p.add_argument("--metadata-output",default=str(DEFAULT_METADATA));p.add_argument("--epochs",type=int,default=300);p.add_argument("--learning-rate",type=float,default=.01);p.add_argument("--weight-decay",type=float,default=.01);p.add_argument("--seed",type=int,default=2026);p.add_argument("--patience",type=int,default=30);p.add_argument("--minimum-epochs",type=int,default=100);p.add_argument("--validation-ratio",type=float,default=.2);p.add_argument("--allow-pending",action="store_true");return p
if __name__=="__main__": train(parser().parse_args())
