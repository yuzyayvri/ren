"""Single bounded 148 + frozen Path Foundation context comparison."""
from __future__ import annotations

import hashlib
import json
import pickle
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from evaluate_hover_checkpoint import coverage_indices, fold2_arrays
from nucleus_evaluation import match_instances
from run_phase2_nonlinear_classifier import predict, score_cache
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from verify_embeddings import verify_fold
from verify_phase2_appearance_context_cache import load_dataset

ROOT=Path("artifacts/phase2_path_foundation_context_classifier_v1"); CACHE=Path("artifacts/phase2_appearance_context_classifier_v1/cache")
CTRL=Path("artifacts/phase2_appearance_context_classifier_v1"); OLD=Path("artifacts/phase2_nonlinear_classifier_v1")
EMB=Path("embeddings/path-foundation"); SEED=20260909; WIDTH=532
EXPECTED={"fold1":(2656,384,"d01c53c31b6e32aad8d8121d84e13dcaff3566fefc0d439004f7103c3efb9f95","2dd1891dbe97c23f616eb6b48f55eede573034e26a014c176612daf82ab8afb7"),"fold2":(2523,384,"a2e24582bf3622d17d816a0895dd02086f56732e1473c8e883a7944fe3f9ef73","94e2f9295512c8124bcdf6d83ea05c4ab2be66ae5ffd15843b722a58ea1943b3")}
CLASS_NAMES={1:"neoplastic",2:"inflammatory",3:"connective",4:"dead",5:"epithelial"}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_inputs():
    out={}
    for fold in ("fold1","fold2"):
        errors=verify_fold(fold,"path-foundation")
        if errors: raise ValueError(f"{fold} embedding verification failed: {errors}")
        n,d,eh,ph=EXPECTED[fold]; ep=EMB/fold
        if sha(ep/"embeddings.npy")!=eh or sha(ep/"provenance.npz")!=ph: raise ValueError(f"{fold} frozen hash mismatch")
        x=np.load(ep/"embeddings.npy",mmap_mode="r"); assert x.shape==(n,d) and x.dtype==np.float32 and np.isfinite(x).all(); shape=list(x.shape)
        out[fold]={"embedding":str(ep/"embeddings.npy"),"provenance":str(ep/"provenance.npz"),"embedding_sha256":eh,"provenance_sha256":ph,"shape":shape,"verify_errors":[]}
    return out
def net(): return torch.nn.Sequential(torch.nn.Linear(WIDTH,64),torch.nn.ReLU(),torch.nn.Dropout(.20),torch.nn.Linear(64,5))
def metrics(y,p):
    pr,rc,f,s=precision_recall_fscore_support(y,p,labels=np.arange(1,6),zero_division=0)
    return {"accuracy":float(accuracy_score(y,p)),"macro_f1":float(f.mean()),"classes":{str(i):{"precision":float(a),"recall":float(b),"f1":float(c),"support":int(d)} for i,a,b,c,d in zip(range(1,6),pr,rc,f,s)}}
def main():
    t0=time.time(); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); ROOT.mkdir(parents=True,exist_ok=True)
    frozen=verify_inputs(); f1,f2,pr=[load_dataset(CACHE,n) for n in ("fold1_truth","fold2_truth","fold2_predicted")]
    arrays={}
    for fold,d in (("fold1",f1),("fold2",f2),("fold2_predicted",pr)):
        emb=np.load(EMB/("fold1" if fold=="fold1" else "fold2")/"embeddings.npy",mmap_mode="r")
        ix=d["patch_index"].astype(np.int64)
        if ix.min()<0 or ix.max()>=len(emb): raise ValueError("patch index out of range")
        joined=np.asarray(emb[ix],dtype=np.float32)
        if not np.isfinite(joined).all(): raise ValueError("non-finite joined embedding")
        arrays[fold]=np.c_[d["base_features"],d["added_features"],joined].astype(np.float32)
    x,y=arrays["fold1"],f1["labels"].astype(np.int64); sc=StandardScaler().fit(x)
    oldsc=np.load(CTRL/"standardizer.npz"); assert np.allclose(sc.mean_[:148],oldsc["mean"],rtol=0,atol=1e-6) and np.allclose(sc.scale_[:148],oldsc["scale"],rtol=0,atol=1e-6)
    np.savez_compressed(ROOT/"standardizer.npz",mean=sc.mean_,scale=sc.scale_)
    old=torch.load(CTRL/"initial_state.pt",weights_only=True); m=net(); st=m.state_dict(); st["0.weight"][:,:148]=old["0.weight"]; st["0.weight"][:,148:]=0; st["0.bias"]=old["0.bias"]; st["3.weight"]=old["3.weight"]; st["3.bias"]=old["3.bias"]; m.load_state_dict(st); torch.save(m.state_dict(),ROOT/"initial_state.pt")
    control=torch.nn.Sequential(torch.nn.Linear(148,64),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(64,5)); control.load_state_dict(old); control.eval(); m.eval()
    with torch.inference_mode(): assert torch.equal(control(torch.from_numpy(((x[:256,:148]-oldsc["mean"])/oldsc["scale"]).astype(np.float32))),m(torch.from_numpy(((x[:256]-sc.mean_)/sc.scale_).astype(np.float32))))
    with (OLD/"fold2_predicted_cache.pkl").open("rb") as h: oldcache=pickle.load(h)
    pc={int(p):{"features":arrays["fold2_predicted"][pr["patch_index"]==p],"ids":pr["instance_id"][pr["patch_index"]==p]} for p in np.unique(pr["patch_index"])}
    for p,r in oldcache.items():
        if int(p) not in pc:
            emb=np.load(EMB/"fold2"/"embeddings.npy",mmap_mode="r")[int(p)]
            feat=np.c_[r["features"],np.repeat(emb[None,:],len(r["features"]),axis=0)].astype(np.float32) if len(r["features"]) else np.empty((0,532),np.float32)
            pc[int(p)]={"features":feat,"ids":r["ids"]}
        pc[int(p)].update({k:r[k] for k in ("prediction","truth","truth_types","ignored")})
    monitor=set(coverage_indices(fold2_arrays()[1])); w=torch.tensor(json.loads((OLD/"report.json").read_text())["class_weights"],dtype=torch.float32); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4); lossfn=torch.nn.CrossEntropyLoss(weight=w); rng=np.random.default_rng(SEED); hist=[]
    for e in range(1,31):
        m.train(); order=rng.permutation(len(y)); losses=[]
        for q in range(0,len(y),512):
            z=order[q:q+512]; opt.zero_grad(); loss=lossfn(m(torch.from_numpy(((x[z]-sc.mean_)/sc.scale_).astype(np.float32))),torch.from_numpy(y[z]-1)); loss.backward(); opt.step(); losses.append(float(loss))
        m.eval(); em=score_cache({i:pc[i] for i in sorted(monitor)},pc,sc.mean_,sc.scale_,m); cp=ROOT/f"epoch_{e:02d}.pt"; torch.save(m.state_dict(),cp); pp=predict(m,x,sc.mean_,sc.scale_); hist.append({"epoch":e,"fold1_loss":float(np.mean(losses)),"fold1_accuracy":float(accuracy_score(y,pp)),"fold1_macro_f1":float(f1_score(y,pp,average="macro")),"monitor_end_to_end":em,"monitor_end_to_end_macro_f1":em["macro_f1"],"checkpoint_sha256":sha(cp)})
    eligible=[r for r in hist if r["monitor_end_to_end"]["classes"]["4"]["recall"]>=.20 and r["monitor_end_to_end"]["classes"]["4"]["f1"]>=.15]; pool=eligible or hist; best=max(pool,key=lambda r:(r["monitor_end_to_end_macro_f1"],r["monitor_end_to_end"]["matched_instance_typing_accuracy"],-r["fold1_loss"],-r["epoch"])); m.load_state_dict(torch.load(ROOT/f"epoch_{best['epoch']:02d}.pt",weights_only=True)); full=score_cache(pc,pc,sc.mean_,sc.scale_,m); truthp=predict(m,arrays["fold2"],sc.mean_,sc.scale_); truth_metrics=metrics(f2["labels"],truthp)
    confusion=np.zeros((5,5),int); paired=[0,0,0]; iou=defaultdict(lambda:[0,0]); unmatched=Counter(); deadmist=0
    for p,r in oldcache.items():
        matches, fps,_=match_instances(r["prediction"],r["truth"],r["ignored"]); ppred=dict(zip(r["ids"].tolist(),predict(m,pc[p]["features"],sc.mean_,sc.scale_).tolist()))
        for pid,tid,iv in matches:
            actual=int(r["truth_types"][tid]); pred=int(ppred[pid])
            confusion[actual-1,pred-1]+=1; paired[0]+=pred==actual; paired[2]+=1; deadmist+=pred==4 and actual!=4
            b=f"({min(iv//.1*.1+.1,.9):.1f},{min(iv//.1*.1+.2,1):.1f}]"; iou[b][0]+=pred==actual; iou[b][1]+=1
        for pid in fps: unmatched[CLASS_NAMES[ppred[pid]]]+=1
    gates={"detection_f1":full["detection"]["f1"]>=.70,"binary_pq":full["binary_pq"]>=.50,"end_to_end_macro_f1":full["macro_f1"]>=.55,"dead_recall":full["classes"]["4"]["recall"]>=.20,"dead_f1":full["classes"]["4"]["f1"]>=.15}
    report={"experiment":"phase2-path-foundation-context-classifier-v1","feature_width":532,"feature_order":"148 existing nucleus features followed by 384 Path Foundation patch context","seed":SEED,"architecture":"532->64 ReLU dropout 0.20 ->5","selected_epoch":best["epoch"],"selection_pool":"dead recall >= 0.20 and dead F1 >= 0.15" if eligible else "all epochs; no epoch met both Dead constraints","eligible_epochs":[r["epoch"] for r in eligible],"selection":best,"history":hist,"full_fold2":full,"valid_full_fold2_truth_mask":truth_metrics,"monitor_indices":sorted(monitor),"geometry_invariant":{"tp":full["detection"]["tp"],"fp":full["detection"]["fp"],"fn":full["detection"]["fn"],"expected":{"tp":40894,"fp":9608,"fn":18475,"detection_f1":0.7444002512036844,"binary_pq":0.596356927353324},"passes":full["detection"]["tp"]==40894 and full["detection"]["fp"]==9608 and full["detection"]["fn"]==18475},"gates":gates,"retention":{"control_monitor_macro_f1":0.5133344600416373,"candidate_monitor_macro_f1":best["monitor_end_to_end_macro_f1"],"retained":"candidate" if best["monitor_end_to_end_macro_f1"]>0.5133344600416373 and best["monitor_end_to_end"]["classes"]["4"]["recall"]>=.2 and best["monitor_end_to_end"]["classes"]["4"]["f1"]>=.15 else "148-feature epoch-29 control"},"fold3_accessed":False,"frozen_inputs":frozen,"elapsed_seconds":time.time()-t0}
    (ROOT/"input_manifest.json").write_text(json.dumps({"source_paths":{"cache":str(CACHE),"fold1_embeddings":str(EMB/"fold1"),"fold2_embeddings":str(EMB/"fold2")},"hashes":{"control_epoch29":sha(CTRL/"epoch_29.pt"),"control_initial":sha(CTRL/"initial_state.pt"),"cache_manifest":sha(CTRL/"cache/manifest.json"),"proposal_cache":sha(OLD/"fold2_predicted_cache.pkl"),"mask_audit":sha(Path("artifacts/pannuke_mask_audit.json"))},"rows":{"fold1_truth":len(f1["labels"]),"fold2_truth":len(f2["labels"]),"fold2_predicted":len(pr["labels"])},"feature_order":report["feature_order"],"join_checks":"all patch indices in range; finite; one vector shared per patch","fold3_accessed":False,"provenance":frozen},indent=2)+"\n")
    (ROOT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    (ROOT/"diagnostics.json").write_text(json.dumps({"matched_type_confusion_rows_truth_columns_prediction":confusion.tolist(),"paired_mask_results_same_matches":{"proposal_mask":{"correct":paired[0],"total":paired[2],"accuracy":paired[0]/paired[2]},"truth_mask":truth_metrics},"unmatched_retained_prediction_class_counts":dict(unmatched),"dead_false_positives":{"mistyped_matched":deadmist,"unmatched_proposals":unmatched["dead"],"total":deadmist+unmatched["dead"]},"fold3_accessed":False},indent=2)+"\n")
    print(json.dumps({"selected_epoch":best["epoch"],"candidate_monitor_macro_f1":best["monitor_end_to_end_macro_f1"],"full_macro_f1":full["macro_f1"],"retained":report["retention"]["retained"]},indent=2))
if __name__=="__main__": main()
