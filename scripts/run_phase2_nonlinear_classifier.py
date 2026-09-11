#!/usr/bin/env python3
"""Bounded MLP comparison on frozen fold1/fold2 segmentation proposals."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from evaluate_hover_checkpoint import (
    coverage_indices,
    fold2_arrays,
    instance_types,
    load_model,
)
from evaluate_instance_suppression import (
    classifier_arrays,
    confidence_features,
    model_features,
    retain,
)
from hover_postprocess import extract_instances
from nucleus_evaluation import (
    add_scores,
    empty_score,
    match_instances,
    score_instances,
    summarize_score,
)
from pannuke_target_policy import IGNORE, make_target
from pilot_hover_fast import fold1_arrays
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from train_instance_classifier import FeatureCapture, pooled_features

SEED=20260909; WIDTH=107; CLASSES=np.arange(1,6)
OUT=Path("artifacts/phase2_nonlinear_classifier_v1")
def digest(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def seed_all(): random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
def mlp(): return torch.nn.Sequential(torch.nn.Linear(WIDTH,64),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(64,5))
def load_mlp_checkpoint(path):
    net = mlp()
    net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    net.eval()
    assert not net.training
    return net
def metrics(y,p):
    pr,rc,f1,s=precision_recall_fscore_support(y,p,labels=CLASSES,zero_division=0)
    return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1.mean()),'classes':{str(c):{'precision':float(a),'recall':float(b),'f1':float(d),'support':int(e)} for c,a,b,d,e in zip(CLASSES,pr,rc,f1,s)}}
def predict(net,x,mean,scale):
        if net.training: net.eval()
        assert not net.training
        with torch.inference_mode(): return net(torch.from_numpy(((x-mean)/scale).astype(np.float32))).argmax(1).numpy()+1

def true_instance_rows(instance, semantic, feature_map, image):
    """Pool features from ground-truth instances; prediction IDs never enter."""
    ids, features = pooled_features(instance, feature_map, image)
    types = instance_types(instance, semantic)
    keep = np.array([int(ident) in types for ident in ids], dtype=bool)
    return ids[keep], features[keep], np.array([types[int(i)] for i in ids[keep]], dtype=np.int64)

def true_mask_metrics(labels, predictions):
    return metrics(np.asarray(labels, dtype=np.int64), np.asarray(predictions, dtype=np.int64))
def build(args, out):
    model=load_model(args.segmentation); model.eval(); [p.requires_grad_(False) for p in model.parameters()]; cap=FeatureCapture(model)
    filt=classifier_arrays(args.validity); all_true=[]; all_y=[]; pred_cache={}
    images1,masks1=fold1_arrays(); images2,masks2=fold2_arrays(); monitor=set(coverage_indices(masks2)); matched_rows=[]
    with torch.inference_mode():
      for fold,images,masks in [(1,images1,masks1),(2,images2,masks2)]:
       for i in range(len(images)):
        sem,truth,_=make_target(masks[i],i); fmap,npl,hv=model_features(model,cap,images[i]); norm=images[i].astype(np.float32)/255
        ids,xf=pooled_features(truth,fmap,norm); ty=instance_types(truth,sem)
        if fold==1:
          all_true.append(xf); all_y.append(np.array([ty[int(z)] for z in ids],np.int64))
        pred=extract_instances(npl,hv,threshold=args.np_threshold); pids,pf=pooled_features(pred,fmap,norm)
        if fold == 1 and len(pids):
          matches, _, _ = match_instances(pred, truth, sem == IGNORE)
          by_id = {int(pid): (int(tid), float(iou)) for pid, tid, iou in matches}
          for pid, feat in zip(pids, pf, strict=True):
            if int(pid) in by_id:
              tid, iou = by_id[int(pid)]
              matched_rows.append((i, int(pid), tid, iou, int(ty[tid]), feat.copy()))
        conf=confidence_features(pred,npl); keep=np.zeros(len(pids),bool)
        if len(pids):
          std=(np.column_stack((pf,conf))-filt['mean'])/filt['scale']; log=std@filt['coefficients'].T+filt['intercept']; keep=(1/(1+np.exp(-log[:,0])))>=args.filter_threshold
        if fold==2:
          pred_cache[i]={'prediction':retain(pred,pids,keep),'ids':pids[keep],'features':pf[keep],'truth':truth,'truth_types':ty,'ignored':sem==IGNORE}
        if (i+1)%256==0: print(f'cached_fold{fold}={i+1}/{len(images)}',flush=True)
    cap.close(); model=None
    np.savez_compressed(out/'true_fold1_cache.npz',features=np.concatenate(all_true).astype(np.float32),labels=np.concatenate(all_y))
    np.savez_compressed(out/'fold1_matched_proposal_cache.npz', patch_index=np.array([r[0] for r in matched_rows], np.int64), proposal_id=np.array([r[1] for r in matched_rows], np.int64), matched_truth_instance_id=np.array([r[2] for r in matched_rows], np.int64), match_iou=np.array([r[3] for r in matched_rows], np.float32), true_class=np.array([r[4] for r in matched_rows], np.int64), features=np.stack([r[5] for r in matched_rows]).astype(np.float32))
    with open(out/'fold2_predicted_cache.pkl','wb') as f: pickle.dump(pred_cache,f,protocol=5)
    return pred_cache, np.concatenate(all_true), np.concatenate(all_y), monitor
def score_cache(cache, pred_by_patch, mean, scale, net):
    total=empty_score(); matched=correct=0; counts=np.zeros(5,int)
    cats={k:0 for k in ('background_only','split_fragment','merge','boundary_localization','other_unresolved')}
    for r in cache.values():
      p=predict(net,r['features'],mean,scale); counts+=np.bincount(p-1,minlength=5); pt=dict(zip(r['ids'].tolist(),p.tolist()))
      add_scores(total,score_instances(r['prediction'],r['truth'],pt,r['truth_types'],r['ignored']))
      m,_,_=match_instances(r['prediction'],r['truth'],r['ignored']); matched+=len(m); correct+=sum(pt[a]==r['truth_types'][b] for a,b,_ in m)
      for a,b in zip(r['ids'],p):
        if int(a) in pt: pass
      # category calculation is imported lazily to keep this script focused.
      from evaluate_instance_suppression import false_positive_categories
      for k,v in false_positive_categories(r['prediction'],r['truth'],r['ignored']).items(): cats[k]+=v
    z=summarize_score(total); z.update({'matched_instance_typing_accuracy':correct/matched if matched else 0,'predicted_class_counts':counts.tolist(),'false_positive_categories':cats}); return z
def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--segmentation',type=Path,default=Path('artifacts/phase2_development_run_v10_segmentation_only/candidate_detection.pt')); ap.add_argument('--validity',type=Path,default=Path('artifacts/phase2_instance_suppression_v1/validity_filter.npz')); ap.add_argument('--output',type=Path,default=OUT); ap.add_argument('--np-threshold',type=float,default=.60); ap.add_argument('--filter-threshold',type=float,default=.35); ap.add_argument('--proposal-cache',type=Path); ap.add_argument('--control-dir',type=Path); a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True); seed_all()
  pred, x,y,monitor=build(a,a.output)
  if a.proposal_cache:
    with np.load(a.proposal_cache) as d: x,y=d['features'],d['true_class']
  scaler=StandardScaler().fit(x); np.savez_compressed(a.output/'standardizer.npz',mean=scaler.mean_,scale=scaler.scale_)
  net=mlp()
  if a.control_dir: net.load_state_dict(torch.load(a.control_dir/'initial_state.pt',weights_only=True),strict=True)
  initial={k:v.detach().clone() for k,v in net.state_dict().items()}; torch.save(initial,a.output/'initial_state.pt'); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4); w=torch.tensor(json.loads((a.control_dir/'report.json').read_text())['class_weights'] if a.control_dir else len(y)/(5*np.bincount(y,minlength=6)[1:]),dtype=torch.float32); lossfn=torch.nn.CrossEntropyLoss(weight=w); rng=np.random.default_rng(SEED); hist=[]; best=None
  # Use frozen predicted feature cache for both monitor paths; geometry is unchanged.
  for epoch in range(1,31):
    net.train(); order=rng.permutation(len(y)); losses=[]
    for start in range(0,len(y),512):
      ix=order[start:start+512]; opt.zero_grad(); logits=net(torch.from_numpy(((x[ix]-scaler.mean_)/scaler.scale_).astype(np.float32))); loss=lossfn(logits,torch.from_numpy(y[ix]-1)); loss.backward(); opt.step(); losses.append(float(loss))
    net.eval(); monitor_cache={i:pred[i] for i in sorted(monitor)}
    em=score_cache(monitor_cache,pred,scaler.mean_,scaler.scale_,net)
    tr={'accuracy':None,'macro_f1':None,'classes':{},'source':'repair_phase2_nonlinear_classifier.py'}
    rec={'epoch':epoch,'fold1_loss':float(np.mean(losses)),'fold1_accuracy':float(accuracy_score(y,predict(net,x,scaler.mean_,scaler.scale_))),'fold1_macro_f1':float(f1_score(y,predict(net,x,scaler.mean_,scaler.scale_),average='macro')),'monitor_true_mask_macro_f1':tr['macro_f1'],'monitor_end_to_end_macro_f1':em['macro_f1'],'monitor_end_to_end':em,'monitor_true_mask':tr,'checkpoint_sha256':None}; cp=a.output/f'epoch_{epoch:02d}.pt'; torch.save(net.state_dict(),cp); rec['checkpoint_sha256']=digest(cp); hist.append(rec)
    eligible=[r for r in hist if r['monitor_end_to_end']['classes']['4']['recall']>=.20 and r['monitor_end_to_end']['classes']['4']['f1']>=.15]; pool=eligible or hist; chosen=max(pool,key=lambda r:(r['monitor_end_to_end_macro_f1'],r['monitor_end_to_end']['matched_instance_typing_accuracy'],-r['fold1_loss'],-r['epoch'])); best=chosen
  torch.load(a.output/f'epoch_{best["epoch"]:02d}.pt',weights_only=True); net.load_state_dict(torch.load(a.output/f'epoch_{best["epoch"]:02d}.pt',weights_only=True)); full=score_cache(pred,pred,scaler.mean_,scaler.scale_,net)
  report={'experiment':'phase2-nonlinear-classifier-v1','seed':SEED,'architecture':'107->64 ReLU dropout .20 ->5','optimizer':'AdamW lr=1e-3 weight_decay=1e-4 batch=512 epochs=30','class_weight_formula':'n/(5*n_class)','class_weights':w.tolist(),'selected_epoch':best['epoch'],'selection':best,'history':hist,'full_fold2_mlp':full,'frozen_inputs':{'segmentation':str(a.segmentation),'segmentation_sha256':digest(a.segmentation),'validity_filter':str(a.validity),'validity_filter_sha256':digest(a.validity),'validity_threshold':a.filter_threshold,'np_threshold':a.np_threshold},'monitor_indices':sorted(monitor),'fold3_accessed':False,'fold2_is_development_data':True,'feature_width':WIDTH}
  (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({'selected_epoch':best['epoch'],'full_fold2_macro_f1':full['macro_f1']},indent=2))
if __name__=='__main__': main()
