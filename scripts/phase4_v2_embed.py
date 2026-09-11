#!/usr/bin/env python3
"""Offline resumable MedCPT article embeddings for v2 GO documents."""
import hashlib,json,os,sqlite3
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; V=ROOT/'artifacts/phase4_protocol_v2'; os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 from transformers import AutoModel,AutoTokenizer
 import torch
 d=sqlite3.connect(V/'ontology.sqlite'); rows=d.execute('select id,name,definition from terms where obsolete=0 order by id').fetchall(); docs=[]
 for i,n,definition in rows:
  syn=[x[0] for x in d.execute('select value from synonyms where term_id=? order by rowid',(i,))]
  docs.append({'id':i,'title':f'{i} {n}','abstract':definition+(' Synonyms: '+'; '.join(syn) if syn else '')})
 d.close(); tmp=V/'documents.json.tmp'; tmp.write_text(json.dumps(docs,ensure_ascii=False,separators=(',',':'))); tmp.replace(V/'documents.json')
 tok=AutoTokenizer.from_pretrained(ROOT/'models/embed/medcpt/article',local_files_only=True); model=AutoModel.from_pretrained(ROOT/'models/embed/medcpt/article',local_files_only=True).eval(); sd=V/'embedding_shards'; sd.mkdir(exist_ok=True)
 for j in range(0,len(docs),256):
  p=sd/f'{j:06d}.npz'
  if p.exists(): continue
  x=tok([[z['title'],z['abstract']] for z in docs[j:j+256]],padding=True,truncation=True,max_length=512,return_tensors='pt')
  with torch.no_grad(): a=model(**x).last_hidden_state[:,0,:].numpy().astype('float32')
  a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12); np.savez(p,indices=np.arange(j,j+len(a)),vectors=a)
 arr=np.concatenate([np.load(p)['vectors'] for p in sorted(sd.glob('*.npz'))]); np.save(V/'article_embeddings.npy',arr); (V/'index_manifest.json').write_text(json.dumps({'schema':2,'backend':'numpy_flat_cosine','rows':len(arr),'dim':int(arr.shape[1]),'dtype':'float32','documents_sha256':sha(V/'documents.json'),'embeddings_sha256':sha(V/'article_embeddings.npy')},indent=2)+'\n')
if __name__=='__main__': main()
