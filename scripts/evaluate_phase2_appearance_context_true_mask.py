"""Deterministically evaluate the retained augmented MLP on fold2 truth rows."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from verify_phase2_appearance_context_cache import load_dataset

ROOT=Path('artifacts/phase2_appearance_context_classifier_v1')
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    cache=load_dataset(ROOT/'cache','fold2_truth')
    x=np.c_[cache['base_features'],cache['added_features']].astype(np.float32)
    y=cache['labels'].astype(np.int64)
    with np.load(ROOT/'standardizer.npz') as s: mean,scale=s['mean'],s['scale']
    net=torch.nn.Sequential(torch.nn.Linear(148,64),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(64,5))
    net.load_state_dict(torch.load(ROOT/'epoch_29.pt',map_location='cpu',weights_only=True)); net.eval()
    with torch.inference_mode(): p=net(torch.from_numpy(((x-mean)/scale).astype(np.float32))).argmax(1).numpy()+1
    pr,rc,f1,s=precision_recall_fscore_support(y,p,labels=np.arange(1,6),zero_division=0)
    result={'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1.mean()),'classes':{str(c):{'precision':float(a),'recall':float(b),'f1':float(d),'support':int(e)} for c,a,b,d,e in zip(range(1,6),pr,rc,f1,s)},'rows':len(y),'checkpoint_sha256':digest(ROOT/'epoch_29.pt'),'standardizer_sha256':digest(ROOT/'standardizer.npz'),'cache_manifest_sha256':digest(ROOT/'cache/manifest.json'),'fold3_accessed':False}
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__': main()
