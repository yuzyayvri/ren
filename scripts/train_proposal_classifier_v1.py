"""Train and evaluate the single proposal-mask classifier intervention."""
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from run_phase2_nonlinear_classifier import (
    digest,
    load_mlp_checkpoint,
    metrics,
    mlp,
    predict,
    score_cache,
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

SEED=20260909; OUT=Path('artifacts/phase2_proposal_trained_classifier_v1'); CONTROL=Path('artifacts/phase2_nonlinear_classifier_v1')
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    with np.load(OUT/'fold1_matched_proposal_cache.npz') as d: x,y={k:d[k] for k in d.files}.values() if False else (d['features'],d['true_class'])
    scaler=StandardScaler().fit(x); mean,scale=scaler.mean_,scaler.scale_; np.savez_compressed(OUT/'standardizer.npz',mean=mean,scale=scale)
    with open(OUT/'fold2_predicted_cache.pkl','rb') as f: pred=pickle.load(f)
    with open(CONTROL/'report.json') as f: control=json.load(f)
    monitor=set(control['monitor_indices']); w=torch.tensor(control['class_weights'],dtype=torch.float32)
    net=mlp(); net.load_state_dict(torch.load(CONTROL/'initial_state.pt',weights_only=True),strict=True); torch.save(net.state_dict(),OUT/'initial_state.pt')
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4); lossfn=torch.nn.CrossEntropyLoss(weight=w); rng=np.random.default_rng(SEED); hist=[]
    for epoch in range(1,31):
        net.train(); order=rng.permutation(len(y)); losses=[]
        for s in range(0,len(y),512):
            ix=order[s:s+512]; opt.zero_grad(); z=net(torch.from_numpy(((x[ix]-mean)/scale).astype(np.float32))); loss=lossfn(z,torch.from_numpy(y[ix]-1)); loss.backward(); opt.step(); losses.append(float(loss))
        net.eval(); em=score_cache({i:pred[i] for i in sorted(monitor)},pred,mean,scale,net); p=predict(net,x,mean,scale); cp=OUT/f'epoch_{epoch:02d}.pt'; torch.save(net.state_dict(),cp)
        hist.append({'epoch':epoch,'fold1_loss':float(np.mean(losses)),'fold1_accuracy':float(accuracy_score(y,p)),'fold1_macro_f1':float(f1_score(y,p,average='macro')),'monitor_end_to_end':em,'monitor_end_to_end_macro_f1':em['macro_f1'],'checkpoint_sha256':digest(cp)})
    eligible=[r for r in hist if r['monitor_end_to_end']['classes']['4']['recall']>=.2 and r['monitor_end_to_end']['classes']['4']['f1']>=.15]; best=max(eligible or hist,key=lambda r:(r['monitor_end_to_end_macro_f1'],r['monitor_end_to_end']['matched_instance_typing_accuracy'],-r['fold1_loss'],-r['epoch']))
    net=load_mlp_checkpoint(OUT/f"epoch_{best['epoch']:02d}.pt"); full=score_cache(pred,pred,mean,scale,net); true=np.load(CONTROL/'true_fold2_cache.npz'); true_metrics=metrics(true['true_class'],predict(net,true['features'],mean,scale))
    report={'experiment':'phase2-proposal-trained-classifier-v1','seed':SEED,'architecture':'107->64 ReLU dropout .20 ->5','optimizer':'AdamW lr=1e-3 weight_decay=1e-4 batch=512 epochs=30','class_weights':w.tolist(),'training_rows':len(y),'training_class_counts':np.bincount(y,minlength=6)[1:].tolist(),'selected_epoch':best['epoch'],'selection':best,'history':hist,'full_fold2':full,'full_fold2_true_mask':true_metrics,'frozen_inputs':{'segmentation_sha256':digest(Path('artifacts/phase2_development_run_v10_segmentation_only/candidate_detection.pt')),'validity_filter_sha256':digest(Path('artifacts/phase2_instance_suppression_v1/validity_filter.npz'))},'fold3_accessed':False,'fold2_is_development_data':True}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n'); (OUT/'summary.md').write_text(f"# Proposal-trained classifier\n\nMatched proposal rows: {len(y)}. Selected epoch: {best['epoch']}. Full-fold2 end-to-end macro-F1: {full['macro_f1']:.6f}; true-mask macro-F1: {true_metrics['macro_f1']:.6f}. Fold3 remained untouched.\n")
if __name__=='__main__': main()
