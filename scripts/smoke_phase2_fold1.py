"""Bounded fold1-only overfit/integration smoke; never writes model artifacts."""
import json, resource, time
from pathlib import Path
import numpy as np, torch
from hover_fast_model import HoVerFast
from hover_loss import masked_hover_loss
from pannuke_target_policy import make_target,hover_offsets
p=Path('data/tissue/fold1/Fold 1'); image=np.load(p/'images/fold1/images.npy',mmap_mode='r')[0]; mask=np.load(p/'masks/fold1/masks.npy',mmap_mode='r')[0]
s,i,report=make_target(mask,0); hv=torch.from_numpy(hover_offsets(i,s))[None]; x=torch.from_numpy(image.astype('float32').transpose(2,0,1)/255)[None]; s=torch.from_numpy(s)[None].long();i=torch.from_numpy(i)[None].long()
m=HoVerFast();opt=torch.optim.Adam(m.parameters(),1e-3); losses=[]; start=time.perf_counter()
for _ in range(8):
 opt.zero_grad(); loss=masked_hover_loss(*m(x),s,i,hv)
 if loss is None: raise RuntimeError('selected fold1 patch has no valid supervision')
 loss.backward();opt.step();losses.append(float(loss.detach()))
print(json.dumps({'fold':'fold1','steps':8,'losses':losses,'seconds':time.perf_counter()-start,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'target':report},indent=2))
