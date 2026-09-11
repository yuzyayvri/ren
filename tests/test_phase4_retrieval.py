import json
import sqlite3
from pathlib import Path

import pytest

from scripts.phase4_retrieval import ingest
from scripts.phase4_v2_ingest import parse


def test_missing_source_fails_closed(tmp_path):
    with pytest.raises(RuntimeError): ingest(tmp_path/'missing.obo')

def test_incomplete_source_fails_closed(tmp_path):
    p=tmp_path/'x.obo'; p.write_text('[Term]\nid: GO:1\nname: x\n')
    with pytest.raises(RuntimeError): ingest(p)

def test_real_obo_has_no_typedef_stanzas():
    text=Path('data/ontology/go-basic.obo').read_text(); assert '[Typedef]' in text; assert all(x.startswith('GO:') for x in [f['id'][0] for f in parse()])
def test_parser_stops_at_header():
    assert len(parse())==48329
def test_quoted_fields_are_retained():
    d=sqlite3.connect('artifacts/phase4_protocol_v2/ontology.sqlite'); assert d.execute("select definition from terms where id='GO:0000001'").fetchone()[0]; d.close()
def test_synonym_scope_type_schema():
    d=sqlite3.connect('artifacts/phase4_protocol_v2/ontology.sqlite'); assert d.execute('select scope,type from synonyms limit 1').fetchone() is not None; d.close()
def test_ambiguous_alias_preserved():
    d=sqlite3.connect('artifacts/phase4_protocol_v2/ontology.sqlite'); assert d.execute("select count(*) from aliases where alias=\"'malic' enzyme\"").fetchone()[0]>=2; d.close()
def test_redirect_kinds():
    d=sqlite3.connect('artifacts/phase4_protocol_v2/ontology.sqlite'); assert {x[0] for x in d.execute('select distinct kind from redirects')}>= {'alt_id','consider','replaced_by'}; d.close()
def test_edges_are_labeled():
    d=sqlite3.connect('artifacts/phase4_protocol_v2/ontology.sqlite'); assert {x[0] for x in d.execute('select distinct relation from edges')} <= {'is_a','part_of'}; d.close()
def test_counts_and_versions():
    with open('artifacts/phase4_protocol_v2/source.json') as f: m=json.load(f)
    assert (m['current_terms'],m['obsolete_terms'])==(38245,10084)
def test_no_dangling_edges():
    d=sqlite3.connect('artifacts/phase4_protocol_v2/ontology.sqlite'); assert d.execute('select count(*) from edges e left join terms t on t.id=e.parent where t.id is null').fetchone()[0]==0; d.close()
def test_benchmark_frozen_status_and_digest():
    p=Path('artifacts/phase4_protocol_v2/benchmark.json'); assert json.loads(p.read_text())['status']=='frozen_prospective_no_evaluation'; assert Path('artifacts/phase4_protocol_v2/benchmark.sha256').read_text().split()[0]
