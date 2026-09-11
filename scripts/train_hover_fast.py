"""Sequential fold1 training / fold2 development loop; no test-fold access."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from hover_fast_model import HoVerFast,IMPLEMENTATION_ID,INITIALIZATION_PROVENANCE
from hover_loss import masked_hover_loss
from pannuke_target_policy import make_target,hover_offsets,POLICY_ID

def arrays(fold):
 b=Path(f'data/tissue/fold{fold}/Fold {fold}');return np.load(b/f'images/fold{fold}/images.npy',mmap_mode='r'),np.load(b/f'masks/fold{fold}/masks.npy',mmap_mode='r')
def main():
 p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=32);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 images,masks=arrays(1);m=HoVerFast();o=torch.optim.Adam(m.parameters(),1e-3);losses=[];skipped=0;t=time.perf_counter()
 for step in range(a.steps):
  idx=step%len(images);s,i,_=make_target(masks[idx],idx)
  x=torch.from_numpy(images[idx].astype('float32').transpose(2,0,1)/255)[None];st=torch.from_numpy(s)[None].long();it=torch.from_numpy(i)[None].long();hv=torch.from_numpy(hover_offsets(i,s))[None]
  o.zero_grad();loss=masked_hover_loss(*m(x),st,it,hv)
  if loss is None: skipped+=1;continue
  loss.backward();o.step();losses.append(float(loss.detach()))
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'implementation':IMPLEMENTATION_ID,'initialization':INITIALIZATION_PROVENANCE,'target_policy':POLICY_ID,'train_fold':1,'development_fold':2,'test_fold_accessed':False,'steps':a.steps,'skipped':skipped,'losses':losses,'seconds':time.perf_counter()-t},indent=2)+'\n')
if __name__=='__main__':main()
