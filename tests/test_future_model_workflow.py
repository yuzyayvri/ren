import hashlib
import json
from pathlib import Path

import pytest

from scripts.verify_future_models import manifest_ok


def write_manifest(root, data):
    (root / '.acquisition-manifest.json').write_text(json.dumps(data))

def test_manifest_validates_and_rejects_truncation(tmp_path):
    p=tmp_path/'x.gguf'; p.write_bytes(b'abc')
    h=hashlib.sha256(b'abc').hexdigest(); write_manifest(tmp_path, {'revision':'r','files':[{'filename':'x.gguf','bytes':3,'sha256':h}]})
    assert manifest_ok(tmp_path)['revision']=='r'; p.write_bytes(b'a')
    with pytest.raises(RuntimeError): manifest_ok(tmp_path)

def test_missing_revision_and_hash_fail_closed(tmp_path):
    write_manifest(tmp_path, {'files':[{'filename':'missing','bytes':1,'sha256':'0'*64}]})
    with pytest.raises(RuntimeError): manifest_ok(tmp_path)

def test_verified_rerun_is_idempotent(tmp_path):
    p=tmp_path/'f'; p.write_bytes(b'payload'); h=hashlib.sha256(b'payload').hexdigest()
    data={'repository':'x','revision':'immutable','files':[{'filename':'f','bytes':7,'sha256':h}]}; write_manifest(tmp_path,data)
    assert manifest_ok(tmp_path)==data

def test_offline_local_path_enforcement(tmp_path, monkeypatch):
    monkeypatch.setenv('HF_HUB_OFFLINE','1'); monkeypatch.setenv('TRANSFORMERS_OFFLINE','1')
    assert Path('models/embed/medcpt/query').is_relative_to(Path('.'))

def test_parser_handles_llama_output(tmp_path):
    report=tmp_path/'r.json'; report.write_text(json.dumps({'requested_context':8192,'returncode':0,'generation_nonempty':True}))
    obj=json.loads(report.read_text()); assert obj['returncode']==0 and obj['generation_nonempty']
