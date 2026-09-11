"""Atomic, checksummed variable-length prediction cache."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def publish(root, ids, boxes, classes, confs, offsets, binding):
 root=Path(root);root.mkdir(parents=True,exist_ok=True); payload=root/'payload.npz'; meta=root/'metadata.json'; commit=root/'commit.json'
 if commit.exists(): raise RuntimeError('committed cache already exists')
 ids=np.asarray(ids); boxes=np.asarray(boxes,np.float32); classes=np.asarray(classes,np.int8); confs=np.asarray(confs,np.float32); offsets=np.asarray(offsets,np.int64)
 if offsets.size!=ids.size+1 or offsets[0]!=0 or offsets[-1]!=len(boxes): raise ValueError('coverage offsets invalid')
 if len({str(x) for x in ids})!=len(ids) or np.any(np.diff(offsets)<0): raise ValueError('duplicate/gap coverage')
 if boxes.ndim!=2 or boxes.shape[1]!=4 or len(classes)!=len(boxes) or len(confs)!=len(boxes): raise ValueError('payload shapes invalid')
 if not np.isfinite(boxes).all() or not np.isfinite(confs).all() or np.any(boxes[:,2:]<=boxes[:,:2]) or np.any((classes<0)|(classes>2)) or np.any(confs<0): raise ValueError('payload values invalid')
 if np.any(np.diff(offsets)>300): raise ValueError('maxDet exceeded')
 tmp=payload.with_suffix('.tmp.npz');np.savez_compressed(tmp,ids=ids,boxes=boxes,classes=classes,confs=confs,offsets=offsets);tmp.replace(payload)
 m={'schema':'phase3-prediction-cache-v1','payload_sha256':digest(payload),'ids':[str(x) for x in ids.tolist()],'binding':binding,'shapes':{k:list(v.shape) for k,v in [('ids',ids),('boxes',boxes),('classes',classes),('confs',confs),('offsets',offsets)]},'dtypes':{k:str(v.dtype) for k,v in [('ids',ids),('boxes',boxes),('classes',classes),('confs',confs),('offsets',offsets)]}}
 mt=meta.with_suffix('.tmp');mt.write_text(json.dumps(m,sort_keys=True,indent=2)+'\n');mt.replace(meta); ct=commit.with_suffix('.tmp');ct.write_text(json.dumps({'metadata_sha256':digest(meta),'payload_sha256':digest(payload)},sort_keys=True)+'\n');ct.replace(commit)
def read(root):
 root=Path(root);payload=root/'payload.npz';meta=root/'metadata.json';commit=root/'commit.json'
 if not all(x.exists() for x in (payload,meta,commit)): raise RuntimeError('incomplete cache publication')
 c=json.loads(commit.read_text());
 if c['payload_sha256']!=digest(payload) or c['metadata_sha256']!=digest(meta): raise RuntimeError('cache tampering')
 m=json.loads(meta.read_text());
 if m['payload_sha256']!=digest(payload): raise RuntimeError('metadata binding mismatch')
 with np.load(payload,allow_pickle=False) as z: d={k:z[k] for k in z.files}
 if [str(x) for x in d['ids'].tolist()]!=m['ids']: raise RuntimeError('identity mismatch')
 return d,m
