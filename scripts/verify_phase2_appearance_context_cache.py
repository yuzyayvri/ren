"""Fail-closed verifier for the published appearance/context cache.

Truth identity is ``(patch, class_label, instance_id)`` because PanNuke IDs
repeat across class channels. Proposal identity is ``(patch, proposal_id)``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA=['patch_index','instance_id','labels','tissue_label','base_features','added_features']
WIDTHS={'base':107,'added':41,'total':148}
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def identity(name,d):
 patch=d['patch_index'].astype(np.int64); ids=d['instance_id'].astype(np.int64)
 if name.endswith('_truth'):
  return list(zip(patch.tolist(),d['labels'].astype(np.int64).tolist(),ids.tolist()))
 if name.endswith('_predicted'):
  return list(zip(patch.tolist(),ids.tolist()))
 raise ValueError(f'unknown identity scope: {name}')
def load_dataset(root,name):
 root=Path(root); mpath=root/name/'manifest.json'
 if not mpath.exists(): raise ValueError(f'missing manifest: {mpath}')
 m=json.loads(mpath.read_text()); shards=m.get('shards',[])
 if not shards or m.get('schema')!=SCHEMA or m.get('widths')!=WIDTHS or len({int(s['shard']) for s in shards})!=len(shards): raise ValueError(f'invalid manifest: {name}')
 out=[]; covered=[]; identities=[]; total=0
 for meta in sorted(shards,key=lambda x:int(x['shard'])):
  p=root/name/f"shard_{meta['shard']:04d}.npz"; j=p.with_suffix('.json')
  if not p.exists() or not j.exists() or sha(p)!=meta.get('sha256'): raise ValueError(f'invalid shard: {p}')
  if json.loads(j.read_text())!=meta: raise ValueError(f'metadata mismatch: {p}')
  with np.load(p) as z: d=dict(z)
  if sorted([k for k in d if k!='width'])!=sorted(SCHEMA) or int(d['width'][0])!=148: raise ValueError(f'schema: {p}')
  n=len(d['patch_index']);
  if any(len(d[k])!=n for k in SCHEMA) or d['base_features'].shape!=(n,107) or d['added_features'].shape!=(n,41) or not np.isfinite(d['base_features']).all() or not np.isfinite(d['added_features']).all(): raise ValueError(f'rows: {p}')
  patches=d['patch_index'].astype(np.int64).tolist(); expected=list(range(meta['first_patch'],meta['last_patch']+1))
  unique_patches=sorted(set(patches))
  if meta['patches']!=expected or not set(unique_patches).issubset(expected) or patches!=sorted(patches): raise ValueError(f'patch coverage: {p}')
  if int(meta.get('rows',-1))!=n: raise ValueError(f'row count: {p}')
  identities.extend(identity(name,d)); total+=n
  covered.extend(meta['patches']); out.append(d)
 expected=list(range(m['expected_patch_coverage'][0],m['expected_patch_coverage'][1]+1))
 if covered!=expected or len(set(covered))!=len(covered): raise ValueError(f'incomplete/overlapping coverage: {name}')
 if len(set(identities))!=len(identities): raise ValueError(f'duplicate identities: {name}')
 if int(m.get('total_rows',-1))!=total or m.get('actual_patch_coverage')!=expected: raise ValueError(f'manifest/count disagreement: {name}')
 labels=np.concatenate([d['labels'] for d in out]); tissue=np.concatenate([d['tissue_label'] for d in out])
 class_support={str(int(x)):int((labels==x).sum()) for x in np.unique(labels)}
 tissue_support={str(x):int((tissue==x).sum()) for x in np.unique(tissue)}
 if m.get('class_support') not in (None,class_support) or m.get('tissue_support') not in (None,tissue_support): raise ValueError(f'support disagreement: {name}')
 return {k:np.concatenate([d[k] for d in out]) for k in SCHEMA}
if __name__=='__main__':
 root=Path('artifacts/phase2_appearance_context_classifier_v1/cache'); result={n:len(load_dataset(root,n)['patch_index']) for n in ('fold1_truth','fold2_truth','fold2_predicted')}; print(json.dumps(result,indent=2))
