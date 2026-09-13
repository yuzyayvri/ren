#!/usr/bin/env python3
"""Fail-closed local verification for selected MedCPT and MedGemma artifacts.

The legacy ``future`` filename and command names are retained for workflow
compatibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import httpx
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def digest(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def manifest_ok(root):
 m=json.loads((root/'.acquisition-manifest.json').read_text())
 for x in m['files']:
  p=root/x['filename']
  if not p.is_file() or p.stat().st_size != x['bytes'] or digest(p)!=x['sha256']: raise RuntimeError(f'manifest mismatch: {p}')
 return m
def medcpt(report):
 os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
 import torch
 from transformers import AutoModel, AutoTokenizer
 qroot=ROOT/'models/embed/medcpt/query'; aroot=ROOT/'models/embed/medcpt/article'
 qm,am=manifest_ok(qroot),manifest_ok(aroot)
 qt=AutoTokenizer.from_pretrained(qroot,local_files_only=True); at=AutoTokenizer.from_pretrained(aroot,local_files_only=True)
 q=AutoModel.from_pretrained(qroot,local_files_only=True,use_safetensors=True).eval(); a=AutoModel.from_pretrained(aroot,local_files_only=True,use_safetensors=True).eval()
 queries=['What is the role of BRCA1 in DNA repair?','What is the treatment for iron deficiency anemia?']
 articles=[('BRCA1 DNA repair','BRCA1 participates in homologous recombination repair of DNA double strand breaks.'),('Iron deficiency anemia','Iron deficiency causes microcytic hypochromic anemia and is treated by iron replacement.'),('Asthma','Asthma is a chronic inflammatory airway disease with variable obstruction.')]
 def emb(tok,model,texts):
  x=tok(texts,padding=True,truncation=True,return_tensors='pt')
  with torch.no_grad(): y=model(**x).last_hidden_state[:,0,:]
  return y.float()
 qe=emb(qt,q,queries); ae=emb(at,a,[f'{t} [SEP] {b}' for t,b in articles])
 scores=(qe@ae.T).numpy().tolist(); ranking=np.argsort(-np.asarray(scores),axis=1).tolist()
 if ranking[0][0] != 0 or ranking[1][0] != 1: raise RuntimeError(f'paired smoke ranking failed: {ranking}')
 if qe.shape[1]!=768 or ae.shape[1]!=768 or not np.isfinite(qe.numpy()).all() or not np.isfinite(ae.numpy()).all(): raise RuntimeError('bad embeddings')
 qe2=emb(qt,q,queries); ae2=emb(at,a,[f'{t} [SEP] {b}' for t,b in articles])
 if not torch.equal(qe,qe2) or not torch.equal(ae,ae2): raise RuntimeError('nondeterministic inference')
 report.update({'query_revision':qm['revision'],'article_revision':am['revision'],'query_hashes':qm['files'],'article_hashes':am['files'],'shapes':[list(qe.shape),list(ae.shape)],'scores':scores,'ranking':ranking,'deterministic':True,'safetensors':True,'network_disabled':True})
def medgemma(report):
 root=ROOT/'models/llm'; m=manifest_ok(root); p=root/'medgemma-1.5-4b-it-Q5_K_M.gguf'
 server=os.environ.get('LLAMA_SERVER','/nix/store/rx4ssxdipyj5ls4slf274l46k4idvqc1-llama-cpp-0.3.0/bin/llama-server')
 if not Path(server).exists(): raise RuntimeError('llama-server not found; set LLAMA_SERVER')
 port=18081; base=[server,'-m',str(p),'--port',str(port),'-c','8192','-ngl','0']
 t=time.monotonic(); proc=subprocess.Popen(base,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env={**os.environ,'HF_HUB_OFFLINE':'1'})
 try:
  ready=False; deadline=time.monotonic()+180
  while time.monotonic()<deadline:
   if proc.poll() is not None: break
   try:
    with httpx.Client(timeout=1) as client: resp=client.get(f'http://127.0.0.1:{port}/health')
    if resp.status_code==200: ready=True; break
   except httpx.HTTPError: pass
   time.sleep(.5)
  if not ready: raise RuntimeError('llama-server did not become ready')
  with httpx.Client(timeout=120) as client:
   resp=client.post(f'http://127.0.0.1:{port}/completion',json={'prompt':'State one non-diagnostic biomedical synthesis sentence: summarize that hemoglobin carries oxygen.','n_predict':32,'temperature':0,'seed':7})
  resp.raise_for_status(); body=resp.json(); text=body.get('content','')
  if not text.strip(): raise RuntimeError('empty generation')
  elapsed=time.monotonic()-t
  report.update({'repository':m['repository'],'revision':m['revision'],'sha256':m['files'][0]['sha256'],'model_bytes':m['files'][0]['bytes'],'requested_context':8192,'generation_nonempty':True,'elapsed_seconds':elapsed,'generation':text,'runtime_stats':{k:body.get(k) for k in ('tokens_evaluated','tokens_predicted','timings')},'backend':'llama.cpp local llama-server','returncode':0})
 finally:
  proc.terminate()
  try: proc.wait(timeout=15)
  except subprocess.TimeoutExpired: proc.kill(); proc.wait()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output',default='artifacts/future_models_verification.json'); a=ap.parse_args(); r={'schema':1}
 medcpt(r); medgemma(r); Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(r,indent=2)+'\n'); Path(a.output+'.verified').write_text('verified\n'); print(json.dumps(r,indent=2))
if __name__=='__main__': main()
