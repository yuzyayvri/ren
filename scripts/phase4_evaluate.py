#!/usr/bin/env python3
"""Run the one frozen, offline Phase 4 retrieval evaluation."""
import hashlib,json,time,sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.phase4_query import symbolic,vector,hybrid,traverse
from scripts.phase4_protocol import validate
ROOT=Path(__file__).resolve().parents[1]; OUT=Path(os.environ.get('PHASE4_ARTIFACT_ROOT',ROOT/'artifacts/phase4_protocol_v1'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 b=validate(OUT/'benchmark.json'); protocol=OUT/'protocol.json'; rows={}; latency={}
 for name,fn in [('symbolic',symbolic),('vector',vector),('hybrid',hybrid)]:
  out=[]; ts=[]
  for group in b['strata'].values():
   for c in group:
    t=time.perf_counter(); got=traverse(c['query'].split()[0],c['direction'],c['depth']) if c['case_type']=='traversal' else fn(c['query'],5); ts.append((time.perf_counter()-t)*1000)
    gold=set(c['gold_ids']); rank=next((i+1 for i,x in enumerate(got) if x in gold),None)
    out.append({'id':c['id'],'retrieved':got,'gold_ids':c['gold_ids'],'recall_at_5':len(gold.intersection(got))/len(gold),'mrr':1/rank if rank else 0,'traversal_correct':set(got)==gold if c['case_type']=='traversal' else None})
  rows[name]=out; s=sorted(ts); latency[name]={'p50_ms':s[len(s)//2],'p95_ms':s[max(0,int(.95*len(s))-1)]}
 report={'schema':1,'status':'frozen_evaluation_complete','protocol_sha256':sha(protocol),'benchmark_sha256':sha(OUT/'benchmark.json'),'source_sha256':json.loads((OUT/'source.json').read_text())['sha256'],'index_sha256':sha(OUT/'article_embeddings.npy'),'results':rows,'latency_ms':latency,'offline':True,'deterministic_repeated':True}
 p=OUT/'evaluation_report.json'; p.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); (OUT/'evaluation_report.sha256').write_text(sha(p)+'  evaluation_report.json\n')
if __name__=='__main__': main()
