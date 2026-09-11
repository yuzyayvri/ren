"""Publish validated appearance/context rows as fixed patch-aligned shards."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np

ROOT=Path('artifacts/phase2_appearance_context_classifier_v1'); BAD=ROOT/'cache_mixed_invalid'; OUT=ROOT/'cache'; SUPER=ROOT/'cache_row_sliced_superseded'
SHARD=64; SCHEMA=['patch_index','instance_id','labels','tissue_label','base_features','added_features']
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def load(name):
 fs=sorted((BAD/name).glob('shard_*.npz')); assert fs
 ds=[dict(np.load(p)) for p in fs]; keys=set(ds[0]); assert all(set(d)==keys for d in ds)
 return {k:np.concatenate([d[k] for d in ds]) for k in keys}
def normalize(d):
 n=len(d['patch_index']); assert all(len(v)==n for k,v in d.items() if k != 'width' and getattr(v,'ndim',0)>0)
 keys=np.asarray(list(zip(d['patch_index'].tolist(),d['instance_id'].tolist())),dtype=[('patch','i8'),('instance','i8')])
 _,first=np.unique(keys,return_index=True); o=first[np.lexsort((d['instance_id'][first],d['patch_index'][first]))]
 return {k:v[o] for k,v in d.items() if k != 'width'}
def publish(name,d,expected,sources,total_patches):
 d=normalize(d); n=len(d['patch_index']); assert n==expected, (name,n,expected)
 assert d['base_features'].ndim==2 and d['added_features'].ndim==2 and d['base_features'].shape[-1]==107 and d['added_features'].shape[-1]==41, (name,d['base_features'].shape,d['added_features'].shape)
 assert np.isfinite(d['base_features']).all() and np.isfinite(d['added_features']).all()
 patches=np.arange(total_patches,dtype=np.int64); assert d['patch_index'].min()>=0 and d['patch_index'].max()<total_patches
 dest=OUT/name; dest.mkdir(parents=True,exist_ok=True); metas=[]
 for s,start in enumerate(range(0,total_patches,SHARD)):
  ps=patches[start:start+SHARD]; q=np.isin(d['patch_index'],ps); payload={k:d[k][q] for k in SCHEMA}; payload['width']=np.array([148],np.int64)
  p=dest/f'shard_{s:04d}.npz'; tmp=p.with_suffix('.tmp.npz'); np.savez_compressed(tmp,**payload); os.replace(tmp,p)
  m={'shard':s,'patches':ps.tolist(),'first_patch':int(ps[0]),'last_patch':int(ps[-1]),'rows':int(q.sum()),'schema':SCHEMA,'base_width':107,'added_width':41,'total_width':148,'sha256':sha(p)}; p.with_suffix('.json').write_text(json.dumps(m,indent=2)+'\n'); metas.append(m)
 manifest={'dataset':name,'expected_patch_coverage':[int(patches[0]),int(patches[-1])],'actual_patch_coverage':patches.tolist(),'total_rows':n,'schema':SCHEMA,'widths':{'base':107,'added':41,'total':148},'shards':metas,'source_cache_hashes':sources,'fold3_accessed':False}; (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); return manifest
def main():
 if OUT.exists() and not SUPER.exists(): shutil.move(str(OUT),str(SUPER))
 OUT.mkdir(parents=True,exist_ok=True); ms=[]
 for name,n,pcount in [('fold1_truth',62709,2656),('fold2_truth',59369,2523)]:
  src=sorted((BAD/name).glob('shard_*.npz')); ms.append(publish(name,load(name),n,[sha(p) for p in src],pcount))
 name='fold2_predicted'; src=sorted((BAD/name).glob('shard_*.npz')); d=normalize(load(name));
 with (ROOT.parent/'phase2_nonlinear_classifier_v1'/'fold2_predicted_cache.pkl').open('rb') as f: canonical=pickle.load(f)
 wanted={(int(p),int(i)) for p,r in canonical.items() for i in r['ids']}; have={(int(p),int(i)) for p,i in zip(d['patch_index'],d['instance_id'])}; assert wanted <= have, 'predicted proposal IDs missing from canonical cache'; keep=np.array([(int(p),int(i)) in wanted for p,i in zip(d['patch_index'],d['instance_id'])]); d={k:v[keep] for k,v in d.items()}
 ms.append(publish(name,d,50584,[sha(p) for p in src],pcount))
 (OUT/'manifest.json').write_text(json.dumps({'dataset':'phase2_appearance_context_cache','datasets':ms,'fold3_accessed':False},indent=2)+'\n')
if __name__=='__main__': main()
