"""Verify augmented shards, train the one predeclared 148-column MLP, and select prospectively."""
from __future__ import annotations

import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from appearance_context_features import PatchAppearanceContext
from evaluate_hover_checkpoint import coverage_indices, fold2_arrays
from run_phase2_nonlinear_classifier import predict, score_cache
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from verify_phase2_appearance_context_cache import load_dataset

OUT=Path('artifacts/phase2_appearance_context_classifier_v1'); CACHE=OUT/'cache'; CTRL=Path('artifacts/phase2_nonlinear_classifier_v1'); SEED=20260909
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name):
 return load_dataset(CACHE, name)
def net(): return torch.nn.Sequential(torch.nn.Linear(148,64),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(64,5))
def main():
 random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
 f1=load('fold1_truth'); f2=load('fold2_truth'); pr=load('fold2_predicted')
 for d in (f1,f2,pr): assert d['base_features'].shape[1]==107 and d['added_features'].shape[1]==41 and np.isfinite(d['base_features']).all() and np.isfinite(d['added_features']).all()
 x=np.c_[f1['base_features'],f1['added_features']].astype(np.float32); y=f1['labels'].astype(np.int64)
 sc=StandardScaler().fit(x); np.savez_compressed(OUT/'standardizer.npz',mean=sc.mean_,scale=sc.scale_)
 control=np.load(CTRL/'standardizer.npz'); assert np.allclose(sc.mean_[:107],control['mean'],rtol=0,atol=1e-6) and np.allclose(sc.scale_[:107],control['scale'],rtol=0,atol=1e-6)
 p_cache={}
 for patch in np.unique(pr['patch_index']):
  q=pr['patch_index']==patch; p_cache[int(patch)]={'features':np.c_[pr['base_features'][q],pr['added_features'][q]].astype(np.float32),'ids':pr['instance_id'][q],'prediction':None}
 # Reuse frozen masks and predictions from canonical cache, replacing feature rows only.
 with (CTRL/'fold2_predicted_cache.pkl').open('rb') as h: old=pickle.load(h)
 images2,_=fold2_arrays()
 for i,r in old.items():
  if i not in p_cache:
   ctx=PatchAppearanceContext(images2[i].astype(np.float32)/255); ids=r['ids']; af=np.stack([np.r_[r['features'][j],ctx.features(r['prediction']==int(z))] for j,z in enumerate(ids)]).astype(np.float32) if len(ids) else np.empty((0,148),np.float32); p_cache[i]={'features':af,'ids':ids,'prediction':r['prediction'],'truth':r['truth'],'truth_types':r['truth_types'],'ignored':r['ignored']}
  else: p_cache[i].update({k:r[k] for k in ('prediction','truth','truth_types','ignored')})
 torch.manual_seed(SEED); n=net(); oldstate=torch.load(CTRL/'initial_state.pt',weights_only=True); s=n.state_dict(); s['0.weight'][:,:107]=oldstate['0.weight']; s['0.weight'][:,107:]=0; s['0.bias']=oldstate['0.bias']; s['3.weight']=oldstate['3.weight']; s['3.bias']=oldstate['3.bias']; n.load_state_dict(s); torch.save(n.state_dict(),OUT/'initial_state.pt')
 # Exact initial-logit check against control on the original 107 columns.
 cnet=torch.nn.Sequential(torch.nn.Linear(107,64),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(64,5)); cnet.load_state_dict(oldstate,strict=True); cnet.eval(); n.eval()
 with torch.inference_mode(): assert torch.equal(cnet(torch.from_numpy(((x[:256,:107]-control['mean'])/control['scale']).astype(np.float32))),n(torch.from_numpy(((x[:256]-sc.mean_)/sc.scale_).astype(np.float32))))
 w=torch.tensor(json.loads((CTRL/'report.json').read_text())['class_weights']); opt=torch.optim.AdamW(n.parameters(),lr=1e-3,weight_decay=1e-4); lossfn=torch.nn.CrossEntropyLoss(weight=w); rng=np.random.default_rng(SEED); monitor=set(coverage_indices(fold2_arrays()[1])); hist=[]
 for e in range(1,31):
  n.train(); order=rng.permutation(len(y)); losses=[]
  for st in range(0,len(y),512):
   ix=order[st:st+512]; opt.zero_grad(); z=n(torch.from_numpy(((x[ix]-sc.mean_)/sc.scale_).astype(np.float32))); loss=lossfn(z,torch.from_numpy(y[ix]-1)); loss.backward(); opt.step(); losses.append(float(loss))
  n.eval(); cp=OUT/f'epoch_{e:02d}.pt'; torch.save(n.state_dict(),cp); em=score_cache({i:p_cache[i] for i in monitor},p_cache,sc.mean_,sc.scale_,n); rec={'epoch':e,'fold1_loss':float(np.mean(losses)),'fold1_accuracy':float(accuracy_score(y,predict(n,x,sc.mean_,sc.scale_))),'fold1_macro_f1':float(f1_score(y,predict(n,x,sc.mean_,sc.scale_),average='macro')),'monitor_end_to_end':em,'monitor_end_to_end_macro_f1':em['macro_f1'],'checkpoint_sha256':sha(cp)}; hist.append(rec)
 eligible=[r for r in hist if r['monitor_end_to_end']['classes']['4']['recall']>=.20 and r['monitor_end_to_end']['classes']['4']['f1']>=.15]; pool=eligible or hist; best=max(pool,key=lambda r:(r['monitor_end_to_end_macro_f1'],r['monitor_end_to_end']['matched_instance_typing_accuracy'],-r['fold1_loss'],-r['epoch']))
 n.load_state_dict(torch.load(OUT/f"epoch_{best['epoch']:02d}.pt",weights_only=True)); full=score_cache(p_cache,p_cache,sc.mean_,sc.scale_,n)
 report={'experiment':'phase2-appearance-context-classifier-v1','feature_width':148,'added_feature_width':41,'seed':SEED,'architecture':'148->64 ReLU dropout .20 ->5','optimizer':'AdamW lr=1e-3 weight_decay=1e-4 batch=512 epochs=30','selected_epoch':best['epoch'],'selection':best,'eligible_epochs':[r['epoch'] for r in eligible],'history':hist,'full_fold2_mlp':full,'geometry_invariant':{'tp':40894,'fp':9608,'fn':18475,'detection_f1':0.7444002512036844,'binary_pq':0.596356927353324},'canonical_proposal_cache_sha256':sha(CTRL/'fold2_predicted_cache.pkl'),'fold3_accessed':False}
 (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n'); (OUT/'fold2_augmented_cache.npz').write_bytes(b'') if False else None; print(json.dumps({'selected_epoch':best['epoch'],'full_fold2_macro_f1':full['macro_f1'],'eligible_epochs':[r['epoch'] for r in eligible]}))
if __name__=='__main__': main()
