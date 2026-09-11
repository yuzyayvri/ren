#!/usr/bin/env python3
"""Row-level fold1 proposal evaluability audit; no filter fitting or fold3 access."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from evaluate_hover_checkpoint import load_model
from evaluate_instance_suppression import model_features
from hover_postprocess import extract_instances
from nucleus_evaluation import match_instances
from pannuke_target_policy import make_target
from pilot_hover_fast import fold1_arrays
from train_instance_classifier import FeatureCapture


def main():
    seg=Path('artifacts/phase2_development_run_v10_segmentation_only/candidate_detection.pt'); model=load_model(seg); model.eval(); cap=FeatureCapture(model); images,masks=fold1_arrays(); total=positive=evaluable=zero=0
    with torch.inference_mode():
        for i in range(len(images)):
            semantic,truth,_=make_target(masks[i],i); _fmap,np_logits,hv=model_features(model,cap,images[i]); prediction=extract_instances(np_logits,hv,threshold=.60); ids=np.unique(prediction); ids=ids[ids>0]
            matches,_,_=match_instances(prediction,truth,semantic==255); matched={a for a,_,_ in matches}
            n=len(ids); ev=sum(bool(np.any((prediction==ident)&(semantic!=255))) for ident in ids)
            total+=n; evaluable+=ev; zero+=n-ev; positive+=len(matched)
            if (i+1)%256==0: print(f'audit_fold1={i+1}/{len(images)}',flush=True)
    cap.close(); negative=evaluable-positive; report={'schema':'validity-filter-provenance-audit-v2','segmentation_checkpoint':str(seg),'segmentation_sha256':__import__('hashlib').sha256(seg.read_bytes()).hexdigest(),'total_generated_proposals':total,'evaluable_matched_positives':positive,'evaluable_unmatched_negatives':negative,'zero_evaluable_proposals':zero,'historical_frozen_filter_rows_received':None,'corrected_selector_rows_produced':evaluable,'invariant_total_equals_parts':total==positive+negative+zero,'historical_contamination':'historically indeterminate','historical_contamination_reason':'the frozen filter artifact stores coefficients and aggregate counts but not row-level selector membership','fold1_only':True,'fold3_accessed':False}
    Path('artifacts/phase2_nonlinear_classifier_v1/validity_filter_provenance_audit.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__': main()
