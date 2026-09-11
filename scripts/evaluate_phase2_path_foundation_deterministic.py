"""Deterministic epoch-30 evaluation, explicitly limited to fold2 artifacts."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from verify_phase2_appearance_context_cache import load_dataset

EXP=Path('artifacts/phase2_path_foundation_context_classifier_v1'); CACHE=Path('artifacts/phase2_appearance_context_classifier_v1/cache'); EMB=np.load('embeddings/path-foundation/fold2/embeddings.npy',mmap_mode='r')
def main():
 s=np.load(EXP/'standardizer.npz'); m=torch.nn.Sequential(torch.nn.Linear(532,64),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(64,5)); m.load_state_dict(torch.load(EXP/'epoch_30.pt',weights_only=True)); m.eval(); out={}
 for name in ('fold2_predicted','fold2_truth'):
  d=load_dataset(CACHE,name); x=np.c_[d['base_features'],d['added_features'],EMB[d['patch_index'].astype(int)]].astype(np.float32)
  with torch.inference_mode(): p=m(torch.from_numpy(((x-s['mean'])/s['scale']).astype(np.float32))).argmax(1).numpy().astype(np.int64)+1
  out[name]={'patch_index':d['patch_index'].astype(np.int64).tolist(),'instance_id':d['instance_id'].astype(np.int64).tolist(),'predictions':p.tolist()}
 payload=json.dumps(out,sort_keys=True,separators=(',',':'))+'\n'; path=EXP/'deterministic_eval.json'; path.write_text(payload); print(json.dumps({'path':str(path),'sha256':hashlib.sha256(payload.encode()).hexdigest(),'rows':{k:len(v['predictions']) for k,v in out.items()}},indent=2))
if __name__=='__main__': main()
