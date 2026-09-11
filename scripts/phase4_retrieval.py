#!/usr/bin/env python3
"""Fail-closed, offline GO ingestion stage; downstream stages require artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'artifacts/phase4_protocol_v1'; OBO=ROOT/'data/ontology/go-basic.obo'
EXPECTED_OBO_SHA256='c72fc198a86983d55e43aac585d1ffdbeb6e3601475b3f18b6045acdc0a0734c'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def _quoted(v):
 import re
 m=re.match(r'"((?:\\.|[^"\\])*)"',v)
 return bytes(m.group(1),'utf8').decode('unicode_escape') if m else v.strip()
def ingest(path=OBO):
 if not path.is_file(): raise RuntimeError(f'missing approved GO OBO: {path}')
 if path.resolve()==OBO.resolve() and sha(path)!=EXPECTED_OBO_SHA256: raise RuntimeError('approved GO OBO hash mismatch')
 blocks=path.read_text(encoding='utf-8').split('\n[Term]\n')[1:]; terms=[]
 for b in blocks:
  f={}
  for line in b.splitlines():
   if ': ' in line:
    k,v=line.split(': ',1); f.setdefault(k,[]).append(v.split(' ! ',1)[0])
  if f.get('id',[''])[0].startswith('GO:'): terms.append(f)
 if len(terms)<1000: raise RuntimeError('GO OBO appears incomplete')
 OUT.mkdir(parents=True,exist_ok=True); db=sqlite3.connect(OUT/'ontology.sqlite'); db.executescript('CREATE TABLE IF NOT EXISTS terms(id TEXT PRIMARY KEY,name TEXT,namespace TEXT,definition TEXT,obsolete INTEGER); CREATE TABLE IF NOT EXISTS aliases(alias TEXT,term_id TEXT); CREATE TABLE IF NOT EXISTS redirects(old_id TEXT,new_id TEXT); CREATE TABLE IF NOT EXISTS edges(parent TEXT,child TEXT); DELETE FROM terms; DELETE FROM aliases; DELETE FROM redirects; DELETE FROM edges;')
 for f in terms:
  i=f['id'][0]; db.execute('INSERT INTO terms VALUES(?,?,?,?,?)',(i,f.get('name',[''])[0],f.get('namespace',[''])[0],_quoted(f.get('def',[''])[0]),f.get('is_obsolete',['false'])[0]=='true'))
  for a in [f.get('name',[''])[0],*[_quoted(x) for x in f.get('synonym',[])],*f.get('alt_id',[])]: db.execute('INSERT INTO aliases VALUES(?,?)',(a.casefold(),i))
  for p in f.get('is_a',[]): db.execute('INSERT INTO edges VALUES(?,?)',(p.split(' ! ')[0],i))
  for rel in f.get('relationship',[]):
   if rel.startswith('part_of '): db.execute('INSERT INTO edges VALUES(?,?)',(rel.split()[1],i))
  for r in f.get('replaced_by',[]): db.execute('INSERT INTO redirects VALUES(?,?)',(i,r.split()[0]))
  for old in f.get('alt_id',[]): db.execute('INSERT OR REPLACE INTO redirects VALUES(?,?)',(old,i))
 db.commit(); db.close(); (OUT/'source.json').write_text(json.dumps({'path':str(path),'sha256':sha(path),'total_term_stanzas':len(blocks),'indexed_nonobsolete_terms':len(terms)},indent=2)+'\n')
def embed():
 import os
 os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
 import numpy as np
 import torch
 from transformers import AutoModel, AutoTokenizer
 db=sqlite3.connect(OUT/'ontology.sqlite'); rows=db.execute('SELECT id,name,definition FROM terms WHERE obsolete=0 ORDER BY id').fetchall(); db.close()
 tok=AutoTokenizer.from_pretrained(ROOT/'models/embed/medcpt/article',local_files_only=True); model=AutoModel.from_pretrained(ROOT/'models/embed/medcpt/article',local_files_only=True).eval()
 shard_dir=OUT/'embedding_shards'; shard_dir.mkdir(exist_ok=True)
 for j in range(0,len(rows),256):
  shard=shard_dir/f'{j:06d}.npz'
  if shard.exists():
   z=np.load(shard); 
   if z['indices'][0]!=j or z['indices'][-1]>=min(j+256,len(rows)) or z['vectors'].shape[1]!=768: raise RuntimeError(f'invalid shard {shard}')
   continue
  x=tok([f'{i} {n} {d}' for i,n,d in rows[j:j+256]],padding=True,truncation=True,max_length=512,return_tensors='pt')
  with torch.no_grad(): v=model(**x).last_hidden_state[:,0,:].float().numpy()
  v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12); tmp=shard.with_suffix('.tmp.npz'); np.savez(tmp,indices=np.arange(j,j+len(v)),vectors=v.astype('float32')); tmp.replace(shard)
 shards=sorted(shard_dir.glob('*.npz')); a=np.concatenate([np.load(s)['vectors'] for s in shards]);
 if len(a)!=len(rows): raise RuntimeError('incomplete shard coverage')
 np.save(OUT/'article_embeddings.npy',a.astype('float32')); (OUT/'documents.json').write_text(json.dumps([{'id':i,'text':f'{i} {n} {d}'} for i,n,d in rows])+'\n'); (OUT/'embedding_manifest.json').write_text(json.dumps({'encoder':'ncbi/MedCPT-Article-Encoder','revision':'d05a736da4bb84ee4057b7f7999485be6ed85465','shape':list(a.shape),'dtype':'float32','source_sha256':json.loads((OUT/'source.json').read_text())['sha256']},indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['ingest','embed','index','query','evaluate']); ap.add_argument('--source',type=Path,default=OBO); a=ap.parse_args()
 if a.stage=='ingest': ingest(a.source)
 elif a.stage=='embed': embed()
 elif a.stage=='index':
  if not (OUT/'article_embeddings.npy').exists(): raise SystemExit('missing canonical embeddings')
  (OUT/'index_manifest.json').write_text(json.dumps({'backend':'numpy-flat-cosine','embedding_sha256':sha(OUT/'article_embeddings.npy'),'documents_sha256':sha(OUT/'documents.json')},indent=2)+'\n')
 elif a.stage=='query':
  if not a.text: raise SystemExit('query text required')
  db=sqlite3.connect(OUT/'ontology.sqlite'); x=db.execute('SELECT term_id FROM aliases WHERE alias=?',(a.text.lower(),)).fetchone(); db.close(); print(json.dumps([x[0]] if x else []))
 elif a.stage=='evaluate':
  b=OUT/'benchmark.json'; b.write_text(json.dumps({'schema':1,'status':'candidate_pending_validation','cases':[]},indent=2)+'\n')
  raise SystemExit('benchmark is candidate-only; validation and evaluation are not yet published')
if __name__=='__main__': main()
