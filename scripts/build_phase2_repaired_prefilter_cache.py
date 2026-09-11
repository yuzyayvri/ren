"""Resumable unfiltered fold2 proposal cache; inference only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from evaluate_hover_checkpoint import fold2_arrays, instance_types, load_model
from evaluate_instance_suppression import (
 FeatureCapture,
 confidence_features,
 model_features,
)
from hover_postprocess import extract_instances
from pannuke_target_policy import IGNORE, make_target
from train_instance_classifier import pooled_features


def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def main(a):
 out=Path(a.output); out.mkdir(parents=True,exist_ok=True); images,masks=fold2_arrays(); model=load_model(a.segmentation_checkpoint); model.eval()
 for p in model.parameters(): p.requires_grad_(False)
 cap=FeatureCapture(model); n=len(images)
 for start in range(0,n,a.shard_size):
  end=min(start+a.shard_size,n); path=out/f"shard_{start//a.shard_size:04d}.pkl"
  if path.exists(): continue
  records={}
  with torch.inference_mode():
   for idx in range(start,end):
    semantic,truth,tissue=make_target(masks[idx],idx); fmap,nplog,hv=model_features(model,cap,images[idx]); pred=extract_instances(nplog,hv,threshold=a.np_threshold)
    ids,base=pooled_features(pred,fmap,images[idx].astype(np.float32)/255.0); conf=confidence_features(pred,nplog)
    records[idx]={"prediction":pred,"ids":ids,"features":base.astype(np.float32),"confidence_features":conf.astype(np.float32),"validity_features":np.column_stack((base,conf)).astype(np.float32),"truth":truth,"truth_types":instance_types(truth,semantic),"ignored":(semantic==IGNORE),"tissue_label":str(tissue)}
    if (idx+1)%64==0: print(f"fold2_cache={idx+1}/{n}",flush=True)
  tmp=path.with_suffix('.tmp'); tmp.write_bytes(pickle.dumps(records,protocol=5)); os.replace(tmp,path)
  path.with_suffix('.json').write_text(json.dumps({"shard":start//a.shard_size,"first_patch":start,"last_patch":end-1,"patches":list(range(start,end)),"rows":sum(len(r['ids']) for r in records.values()),"sha256":sha(path)},indent=2)+"\n")
 cap.close()
 metas=[json.loads(p.with_suffix('.json').read_text()) for p in sorted(out.glob('shard_*.json'))]
 manifest={"schema":"phase2-repaired-prefilter-cache-v1","fold":2,"shard_size":a.shard_size,"expected_patch_coverage":[0,n-1],"segmentation_checkpoint":str(a.segmentation_checkpoint),"segmentation_sha256":sha(a.segmentation_checkpoint),"np_threshold":a.np_threshold,"shards":metas,"complete":len(metas)==(n+a.shard_size-1)//a.shard_size and [x for m in metas for x in m['patches']]==list(range(n))}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+"\n"); print(json.dumps({"complete":manifest['complete'],"patches":sum(len(m['patches']) for m in metas),"proposals":sum(m['rows'] for m in metas)},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--segmentation-checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--shard-size',type=int,default=64); p.add_argument('--np-threshold',type=float,default=.60); main(p.parse_args())
