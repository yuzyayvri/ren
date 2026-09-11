#!/usr/bin/env python3
"""Validate and freeze the prospective Phase 4 retrieval protocol/benchmark."""
from __future__ import annotations
import hashlib, json, sqlite3
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'artifacts/phase4_protocol_v1'; DB=OUT/'ontology.sqlite'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def _db(): return sqlite3.connect(DB)
def validate(candidate: Path) -> dict:
 data=json.loads(candidate.read_text()); required={'tissue_nuclei','blood_cells','blasts','exact_go_ids','synonyms','ambiguous_names','rare_terms','traversal'}
 if set(data.get('strata',{})) != required: raise RuntimeError('benchmark strata mismatch')
 db=_db(); ids={r[0] for r in db.execute('select id from terms where obsolete=0')}; aliases={}
 for a,i in db.execute('select alias,term_id from aliases'): aliases.setdefault(a,[]).append(i)
 for case in [c for cs in data['strata'].values() for c in cs]:
  gold=case.get('gold_ids',[])
  if not gold or not set(gold)<=ids: raise RuntimeError(f'invalid gold IDs: {case.get("id")}')
  if case['case_type']=='synonym' and not any(i in gold for i in aliases.get(case['query'].casefold(),[])): raise RuntimeError('synonym resolution mismatch')
  if case['case_type']=='ambiguous' and len(aliases.get(case['query'].casefold(),[]))<2: raise RuntimeError('not genuinely ambiguous')
  if case['case_type']=='traversal':
   if case.get('edge') and not db.execute('select 1 from edges where parent=? and child=?',(case['edge'][0],case['edge'][1])).fetchone(): raise RuntimeError('missing traversal edge')
 db.close(); return data
def freeze() -> None:
 candidate=OUT/'benchmark_candidate.json'; data=validate(candidate)
 protocol=json.loads((OUT/'protocol.json').read_text()); data['protocol_sha256']=digest(OUT/'protocol.json'); data['status']='frozen_prospective_no_evaluation'; data['benchmark_sha256']=None
 payload=json.dumps(data,indent=2,sort_keys=True)+'\n'; (OUT/'benchmark.json.tmp').write_text(payload); final=OUT/'benchmark.json'; (OUT/'benchmark.json.tmp').replace(final)
 data['benchmark_sha256']=digest(final); (OUT/'benchmark.sha256').write_text(data['benchmark_sha256']+'  benchmark.json\n')
def main(): freeze()
if __name__=='__main__': main()
