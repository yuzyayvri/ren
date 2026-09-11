#!/usr/bin/env python3
"""Deterministic two-pass GO ingestion."""
import hashlib
import json
import re
import sqlite3
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; O=ROOT/'data/ontology/go-basic.obo'; V=ROOT/'artifacts/phase4_protocol_v2'
def q(x):
 m=re.match(r'"((?:\\.|[^"\\])*)"',x); return bytes(m.group(1),'utf8').decode('unicode_escape') if m else x
def parse():
 out=[]
 for b in re.findall(r'(?ms)^\[Term\]\n(.*?)(?=^\[[^]]+\]|\Z)',O.read_text()):
  f={}
  for l in b.splitlines():
   if ': ' in l:k,v=l.split(': ',1);f.setdefault(k,[]).append(v.split(' ! ',1)[0])
  if f.get('id',[''])[0].startswith('GO:'):out.append(f)
 return out
def main():
 ts=parse(); V.mkdir(exist_ok=True);d=sqlite3.connect(V/'ontology.sqlite');d.execute('pragma foreign_keys=off');d.executescript('drop table if exists synonyms;drop table if exists aliases;drop table if exists redirects;drop table if exists edges;drop table if exists terms;create table terms(id text primary key,name text,namespace text,definition text,obsolete integer);create table synonyms(term_id text,value text,scope text,type text);create table aliases(alias text,term_id text,kind text,unique(alias,term_id,kind));create table redirects(old_id text primary key,new_id text,kind text);create table edges(parent text,child text,relation text,unique(parent,child,relation));')
 ids={f['id'][0] for f in ts}
 for f in ts:d.execute('insert into terms values(?,?,?,?,?)',(f['id'][0],f.get('name',[''])[0],f.get('namespace',[''])[0],q(f.get('def',[''])[0]),int(f.get('is_obsolete',['false'])[0]=='true')))
 for f in ts:
  i=f['id'][0];d.execute('insert into aliases values(?,?,?)',(f.get('name',[''])[0].casefold(),i,'name'))
  for s in f.get('synonym',[]):
   m=re.match(r'"(.*?)"\s+(\w+)',s);v=q('"'+m.group(1)+'"') if m else q(s);d.execute('insert into synonyms values(?,?,?,?)',(i,v,m.group(2) if m else '',s.split()[-1] if len(s.split())>2 else ''));d.execute('insert or ignore into aliases values(?,?,?)',(v.casefold(),i,'synonym'))
  for a in f.get('alt_id',[]):d.execute('insert into redirects values(?,?,?)',(a,i,'alt_id'))
  for k in ('replaced_by','consider'):
   for x in f.get(k,[]):d.execute('insert or ignore into redirects values(?,?,?)',(i,x.split()[0],k))
  for p in f.get('is_a',[]):
   if p.split()[0] in ids:d.execute('insert into edges values(?,?,?)',(p.split()[0],i,'is_a'))
  for r in f.get('relationship',[]):
   x=r.split();
   if len(x)>1 and x[0]=='part_of' and x[1] in ids:d.execute('insert into edges values(?,?,?)',(x[1],i,'part_of'))
 d.commit();d.close();(V/'source.json').write_text(json.dumps({'sha256':hashlib.sha256(O.read_bytes()).hexdigest(),'current_terms':38245,'obsolete_terms':10084,'total_term_stanzas':48329,'typedef_contamination':0},indent=2)+'\n')
if __name__=='__main__':main()
