"""Freeze TXL detector from validation, then perform exactly one test pass."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from ultralytics import YOLO

GRID=np.arange(.05,.951,.05); CLASSES=(0,1,2)
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def iou(a,b):
 x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1)
 return inter/(max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter))
def gt(path):
 from PIL import Image
 w,h=Image.open(path).size; out=[]
 for line in path.parent.parent.parent.joinpath('labels',path.parent.name,path.stem+'.txt').read_text().splitlines():
  c,x,y,bw,bh=map(float,line.split()); out.append((int(c),((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h)))
 return out
def score(rows, threshold):
 tp=fp=fn=0; per={c:[0,0,0] for c in CLASSES};
 for truth,pred in rows:
  used=set()
  for c,box,conf in sorted(pred,key=lambda z:-z[2]):
   choices=[(iou(box,tb),j) for j,(tc,tb) in enumerate(truth) if tc==c and j not in used]
   best=max(choices,default=(0,-1))
   if conf>=threshold and best[0]>=.5: used.add(best[1]); per[c][0]+=1; tp+=1
   elif conf>=threshold: per[c][1]+=1; fp+=1
  for j,(c,_) in enumerate(truth):
   if j not in used: per[c][2]+=1; fn+=1
 fs=[]; rec=[]
 for c in CLASSES:
  t,p,n=per[c]; r=t/(t+n) if t+n else 0; q=t/(t+p) if t+p else 0; fs.append(2*q*r/(q+r) if q+r else 0); rec.append(r)
 return {'macro_f1':float(np.mean(fs)),'min_recall':float(min(rec)),'overall_recall':float(tp/(tp+fn) if tp+fn else 0),'per_class_recall':rec,'tp':tp,'fp':fp,'fn':fn}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); p.add_argument('--model',type=Path,required=True); p.add_argument('--data-root',type=Path,required=True); a=p.parse_args(); freeze=a.run/'detector_freeze.json'; marker=a.run/'test_exposed.marker'; report=a.run/'detector_test_report.json'; a.run.mkdir(parents=True,exist_ok=True)
 if report.exists(): print(json.dumps(json.loads(report.read_text()),indent=2)); return
 model=YOLO(str(a.model),task='detect'); rows=[]
 for path in sorted((a.data_root/'images'/'val').glob('*.png')):
  r=model.predict(str(path),conf=0.001,iou=.70,device='cpu',verbose=False)[0]; pred=[(int(c),tuple(map(float,b)),float(s)) for b,c,s in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.cls.cpu().numpy(),r.boxes.conf.cpu().numpy())]; rows.append((gt(path),pred))
 choices=[(score(rows,t)['macro_f1'],score(rows,t)['min_recall'],score(rows,t)['overall_recall'],-t,t) for t in GRID]; best=max(choices); selected=best[4]; val=score(rows,selected)
 payload={'model':str(a.model),'model_sha256':sha(a.model),'threshold':selected,'nms_iou':.70,'validation':val,'grid':GRID.tolist(),'source_roles':['train','val','test']}; tmp=freeze.with_suffix('.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)); tmp.replace(freeze)
 if marker.exists(): raise RuntimeError('test marker exists without report')
 marker.write_text(json.dumps({'freeze_sha256':sha(freeze),'exposed':True})+'\n')
 rows=[]
 for path in sorted((a.data_root/'images'/'test').glob('*.png')):
  r=model.predict(str(path),conf=selected,iou=.70,device='cpu',verbose=False)[0]; pred=[(int(c),tuple(map(float,b)),float(s)) for b,c,s in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.cls.cpu().numpy(),r.boxes.conf.cpu().numpy())]; rows.append((gt(path),pred))
 out={'model_sha256':payload['model_sha256'],'threshold':selected,'validation':val,'test':score(rows,selected),'gates':{'macro_f1':score(rows,selected)['macro_f1']>=.80,'min_recall':score(rows,selected)['min_recall']>=.60}}
 tmp=report.with_suffix('.tmp'); tmp.write_text(json.dumps(out,indent=2,sort_keys=True)); tmp.replace(report); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
