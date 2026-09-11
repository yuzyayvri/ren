"""Deterministic COCO-style AP50 over a committed prediction cache."""
from __future__ import annotations

import hashlib

import numpy as np


def box_iou(a,b):
 x=max(0,min(a[2],b[2])-max(a[0],b[0])); y=max(0,min(a[3],b[3])-max(a[1],b[1])); z=x*y
 return z/max(1e-12,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-z)
def ap50(records, classes=3):
 out=[]
 for c in range(classes):
  det=[]; n=0
  for ident,truth,pred in records:
   targets=[b for k,b in truth if k==c]; n+=len(targets)
   det += [(float(s),ident,b,targets) for k,b,s in pred if k==c]
  det.sort(key=lambda z:(-z[0],z[1]))
  used=set(); tp=[]; fp=[]
  for s,ident,b,targets in det:
   best=max(((box_iou(b,t),j) for j,t in enumerate(targets) if (ident,j) not in used),default=(0,-1))
   if best[0]>=.5: used.add((ident,best[1]));tp.append(1);fp.append(0)
   else:tp.append(0);fp.append(1)
  if n==0: out.append(0.);continue
  t=np.cumsum(tp);f=np.cumsum(fp);r=t/n;p=t/np.maximum(t+f,1);env=np.maximum.accumulate(p[::-1])[::-1]
  out.append(float(np.mean([max((env[i] for i,x in enumerate(r) if x>=q),default=0.) for q in np.linspace(0,1,101)])))
 return out,float(np.mean(out))
def cache_digest(path):
 h=hashlib.sha256();h.update(path.read_bytes());return h.hexdigest()
