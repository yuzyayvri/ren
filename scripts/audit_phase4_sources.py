#!/usr/bin/env python3
"""Audit local Phase 4 sources without network access."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def main():
    obo=sorted(ROOT.glob('**/*.obo'))
    report={'schema':1,'network_disabled':True,'go_obo_files':[],'complete':bool(obo),'status':'complete' if obo else 'blocked_missing_local_obo','fixture':'artifacts/phase4_protocol_v1','medcpt':{}}
    for p in obo:
        h=hashlib.sha256(p.read_bytes()).hexdigest(); report['go_obo_files'].append({'path':str(p.relative_to(ROOT)),'sha256':h,'bytes':p.stat().st_size})
    for role in ('query','article'):
        p=ROOT/f'models/embed/medcpt/{role}/.acquisition-manifest.json'; report['medcpt'][role]=json.loads(p.read_text()) if p.exists() else None
    out=ROOT/'artifacts/phase4_protocol_v1/source_audit.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
