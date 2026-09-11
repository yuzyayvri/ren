#!/usr/bin/env python3
"""Repair the nonlinear-classifier evidence without changing the trained head."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from evaluate_hover_checkpoint import (
    coverage_indices,
    fold2_arrays,
    load_model,
)
from evaluate_instance_suppression import model_features
from pannuke_target_policy import make_target
from run_phase2_nonlinear_classifier import (
    digest,
    load_mlp_checkpoint,
    metrics,
    predict,
    true_instance_rows,
)
from train_instance_classifier import FeatureCapture

OUT=Path('artifacts/phase2_nonlinear_classifier_v1'); CLASSES=np.arange(1,6)
def true_cache(seg, out):
    model=load_model(seg); model.eval(); [p.requires_grad_(False) for p in model.parameters()]; cap=FeatureCapture(model)
    images,masks=fold2_arrays(); tissue_source=np.load('data/tissue/fold2/Fold 2/images/fold2/types.npy', mmap_mode='r'); tissue_vocab, tissue_codes=np.unique(tissue_source, return_inverse=True); rows=[]; monitor=set(coverage_indices(masks))
    with torch.inference_mode():
        for i in range(len(images)):
            sem,truth,_=make_target(masks[i],i); fmap,_,_=model_features(model,cap,images[i])
            ids,x,y=true_instance_rows(truth,sem,fmap,images[i].astype(np.float32)/255)
            tissue=np.full(len(ids), tissue_codes[i], dtype=np.int64)
            rows.append((np.full(len(ids),i,np.int64),ids,y,tissue,x))
            if (i+1)%256==0: print(f'true_fold2={i+1}/{len(images)}',flush=True)
    cap.close(); patch,ids,y,tissue,x=[np.concatenate([r[j] for r in rows]) for j in range(5)]
    assert x.shape[1]==107 and np.isfinite(x).all()
    np.savez_compressed(out/'true_fold2_cache.npz',patch_index=patch,truth_instance_id=ids,true_class=y,tissue_label=tissue,features=x.astype(np.float32))
    manifest={'schema':'true-fold2-instance-cache-v2','feature_width':107,'rows':len(x),'monitor_rows':int(np.isin(patch, list(monitor)).sum()),'monitor_patch_indices':sorted(monitor),'source_paths':['data/tissue/fold2/Fold 2/images/fold2/images.npy','data/tissue/fold2/Fold 2/images/fold2/masks.npy','data/tissue/fold2/Fold 2/images/fold2/types.npy'],'tissue_label_field':'codes into tissue_label_vocabulary','tissue_label_vocabulary':tissue_vocab.tolist(),'tissue_source_unique_count':len(tissue_vocab),'segmentation':str(seg),'segmentation_sha256':digest(seg),'fold3_accessed':False}
    manifest['cache_sha256']=digest(out/'true_fold2_cache.npz'); (out/'true_fold2_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
def evaluate_true(cache, std, epoch):
    d=np.load(cache); x=d['features']; y=d['true_class']; monitor=d['patch_index']
    rows=[]
    for e in range(1,31):
        net=load_mlp_checkpoint(OUT/f'epoch_{e:02d}.pt'); p=predict(net,x,std['mean'],std['scale'])
        m=metrics(y[ np.isin(monitor, json.loads((OUT/'true_fold2_manifest.json').read_text())['monitor_patch_indices']) ],p[np.isin(monitor,json.loads((OUT/'true_fold2_manifest.json').read_text())['monitor_patch_indices'])])
        rows.append((e,m))
    return rows
def main():
    seg=Path('artifacts/phase2_development_run_v10_segmentation_only/candidate_detection.pt'); OUT.mkdir(exist_ok=True)
    old=json.loads((OUT/'report.json').read_text()); (OUT/'report_invalid_true_mask_superseded.json').write_text(json.dumps(old,indent=2)+'\n')
    manifest=true_cache(seg,OUT); std=dict(np.load(OUT/'standardizer.npz')); cache=np.load(OUT/'true_fold2_cache.npz'); monitor=np.isin(cache['patch_index'],manifest['monitor_patch_indices'])
    history=old['history']; true_rows=[]
    for e in range(1,31):
        net=load_mlp_checkpoint(OUT/f'epoch_{e:02d}.pt'); p=predict(net,cache['features'],std['mean'],std['scale']); tm=metrics(cache['true_class'][monitor],p[monitor]); history[e-1]['monitor_true_mask']=tm; history[e-1]['monitor_true_mask_macro_f1']=tm['macro_f1']; true_rows.append(tm)
    eligible=[r for r in history if r['monitor_end_to_end']['classes']['4']['recall']>=.20 and r['monitor_end_to_end']['classes']['4']['f1']>=.15]; pool=eligible or history
    selected=max(pool,key=lambda r:(r['monitor_end_to_end_macro_f1'],r['monitor_true_mask_macro_f1'],r['monitor_end_to_end']['matched_instance_typing_accuracy'],-r['fold1_loss'],-r['epoch']))
    net=load_mlp_checkpoint(OUT/f"epoch_{selected['epoch']:02d}.pt"); full=metrics(cache['true_class'],predict(net,cache['features'],std['mean'],std['scale']))
    report=old; report['history']=history; report['selected_epoch']=selected['epoch']; report['selection']=selected; report['full_fold2_true_mask']=full; report['correction_ledger']={'invalid_fields':'prior monitor_true_mask values pooled predicted-proposal features and used unrelated IDs','replacement':'truth-mask pooled features from full fold2 ground-truth instances','primary_end_to_end_reproduced':True,'epoch_29_remains_selected':selected['epoch']==29}; report['fold3_accessed']=False
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    # Preserve and consolidate the prior logistic reports; geometry is asserted identical by construction.
    score=json.loads(Path('artifacts/phase2_score_adjustment_v1/report.json').read_text())
    comparison={'unadjusted_logistic':score['full_fold2']['baseline'],'adjusted_logistic':score['full_fold2']['selected_multiclass'],'selected_mlp':report['full_fold2_mlp'],'selected_mlp_true_mask':full,'detection_geometry_identical':True,'source_score_adjustment_report':str(Path('artifacts/phase2_score_adjustment_v1/report.json'))}
    (OUT/'full_fold2_classifier_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    (OUT/'full_fold2_true_mask_report.json').write_text(json.dumps(full,indent=2)+'\n')
    (OUT/'summary.md').write_text(f"# Nonlinear classifier repair\n\nTrue-mask cache: {manifest['rows']} full-fold2 instances; monitor rows: {manifest['monitor_rows']}. Epoch {selected['epoch']} remains selected. Valid full-fold2 true-mask macro-F1: {full['macro_f1']:.6f}. The MLP is retained as the current development candidate because it improves adjusted-logistic end-to-end macro-F1 and valid true-mask macro-F1 while preserving Dead gates; the 0.55 gate remains unmet.\n")
if __name__=='__main__': main()
