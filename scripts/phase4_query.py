#!/usr/bin/env python3
"""Offline MedCPT query, symbolic, vector, and hybrid retrieval."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]; OUT=Path(os.environ.get('PHASE4_ARTIFACT_ROOT',ROOT/'artifacts/phase4_protocol_v1')); _MODEL=None; _TOK=None
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 m=json.loads((OUT/'index_manifest.json').read_text())
 if sha(OUT/'article_embeddings.npy')!=m.get('embedding_sha256',m.get('embeddings_sha256')) or sha(OUT/'documents.json')!=m['documents_sha256']: raise RuntimeError('immutable index hash mismatch')
def symbolic(q,k=10):
 db=sqlite3.connect(OUT/'ontology.sqlite'); rows=db.execute('SELECT id,name FROM terms WHERE obsolete=0').fetchall(); aliases=db.execute('SELECT alias,term_id FROM aliases').fetchall(); red=dict(db.execute('SELECT old_id,new_id FROM redirects')); db.close(); ql=q.casefold(); exact=[red.get(i,i) for a,i in aliases if a==ql]; scored=[]
 for i,n in rows:
  toks=set(ql.split()); score=(1000 if i in exact else 0)+(500 if n.lower()==ql else 0)+len(toks&set(n.lower().split()))
  if score: scored.append((score,i))
 return [i for _,i in sorted(scored,key=lambda x:(-x[0],x[1]))[:k]]
def vector(q,k=10):
 verify(); os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
 global _MODEL,_TOK
 if _MODEL is None: _TOK=AutoTokenizer.from_pretrained(ROOT/'models/embed/medcpt/query',local_files_only=True); _MODEL=AutoModel.from_pretrained(ROOT/'models/embed/medcpt/query',local_files_only=True).eval()
 tok=_TOK; model=_MODEL; x=tok([q],padding=True,truncation=True,max_length=64,return_tensors='pt')
 with torch.no_grad(): v=model(**x).last_hidden_state[:,0,:].float().numpy()
 v/=max(float(np.linalg.norm(v)),1e-12); a=np.load(OUT/'article_embeddings.npy',mmap_mode='r'); s=a@v[0]; ids=[x[0] for x in sqlite3.connect(OUT/'ontology.sqlite').execute('SELECT id FROM terms WHERE obsolete=0 ORDER BY id')]; return [ids[i] for i in np.argsort(-s,kind='stable')[:k]]
def hybrid(q,k=10):
 a=symbolic(q,k*2); b=vector(q,k*2); score={i:1/(60+j) for j,i in enumerate(a)}; score.update({i:score.get(i,0)+1/(60+j) for j,i in enumerate(b)}); return sorted(score,key=lambda i:(-score[i],i))[:k]
def traverse(go_id,direction='ancestors',depth=10):
 db=sqlite3.connect(OUT/'ontology.sqlite'); seen={go_id}; frontier=[go_id]; out=[]
 for _ in range(depth):
  nxt=[]
  for x in frontier:
   col='parent' if direction=='ancestors' else 'child'; other='child' if direction=='ancestors' else 'parent'; rows=db.execute(f'SELECT {col} FROM edges WHERE {other}=?',(x,)).fetchall()
   for (y,) in rows:
    if y not in seen: seen.add(y); out.append(y); nxt.append(y)
  frontier=nxt
 db.close(); return sorted(out)
