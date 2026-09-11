"""Repair diagnostics for the frozen epoch-30 Path Foundation run; fold2 only."""
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from nucleus_evaluation import match_instances
from path_foundation_context_verification import exact_geometry
from sklearn.metrics import precision_recall_fscore_support
from verify_phase2_appearance_context_cache import load_dataset

EXP=Path('artifacts/phase2_path_foundation_context_classifier_v1'); CACHE=Path('artifacts/phase2_appearance_context_classifier_v1/cache'); EMB=Path('embeddings/path-foundation')
NAMES={1:'neoplastic',2:'inflammatory',3:'connective',4:'dead',5:'epithelial'}
BINS=((.5,.6),(.6,.7),(.7,.8),(.8,.9),(.9,1.0))
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pred(model,x,mean,scale):
    with torch.inference_mode(): return model(torch.from_numpy(((x-mean)/scale).astype(np.float32))).argmax(1).numpy()+1
def acc(c,t): return {'correct':int(c),'total':int(t),'accuracy':float(c/t) if t else None}
def main():
    sc=np.load(EXP/'standardizer.npz'); m=torch.nn.Sequential(torch.nn.Linear(532,64),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(64,5)); m.load_state_dict(torch.load(EXP/'epoch_30.pt',weights_only=True)); m.eval()
    truth=load_dataset(CACHE,'fold2_truth'); proposals=load_dataset(CACHE,'fold2_predicted'); e=np.load(EMB/'fold2/embeddings.npy',mmap_mode='r')
    def features(d): return np.c_[d['base_features'],d['added_features'],e[d['patch_index'].astype(int)]].astype(np.float32)
    ty=pred(m,features(truth),sc['mean'],sc['scale']); py=pred(m,features(proposals),sc['mean'],sc['scale'])
    tl={(int(p),int(i)):int(y) for p,i,y in zip(truth['patch_index'],truth['instance_id'],ty)}; pl={(int(p),int(i)):int(y) for p,i,y in zip(proposals['patch_index'],proposals['instance_id'],py)}
    tissue={int(p):str(t) for p,t in zip(truth['patch_index'],truth['tissue_label'])}; conf=np.zeros((5,5),int); pair=[0,0,0]; tissues=defaultdict(lambda:[0,0]); bins={f'({lo:.1f},{hi:.1f}]':[0,0] for lo,hi in BINS}; unmatched=Counter(); mistyped=0
    with open('artifacts/phase2_nonlinear_classifier_v1/fold2_predicted_cache.pkl','rb') as handle:
        records=pickle.load(handle)
    for p,r in records.items():
        matches,fps,_=match_instances(r['prediction'],r['truth'],r['ignored'])
        for pid,tid,iou in matches:
            actual=int(r['truth_types'][tid]); pv=pl[(int(p),int(pid))]; tv=tl[(int(p),int(tid))]; conf[actual-1,pv-1]+=1; pair[0]+=pv==actual; pair[1]+=tv==actual; pair[2]+=1; mistyped += pv==4 and actual!=4
            tissues[tissue[int(p)]][0]+=pv==actual; tissues[tissue[int(p)]][1]+=1
            for lo,hi in BINS:
                if lo < float(iou) <= hi: bins[f'({lo:.1f},{hi:.1f}]'][0]+=pv==actual; bins[f'({lo:.1f},{hi:.1f}]'][1]+=1; break
        for pid in fps: unmatched[NAMES[pl[(int(p),int(pid))]]]+=1
    cm_pr,cm_rc,cm_f1,cm_sup=precision_recall_fscore_support(truth['labels'],ty,labels=np.arange(1,6),zero_division=0)
    truth_metrics={'accuracy':float((truth['labels']==ty).mean()),'macro_f1':float(cm_f1.mean()),'classes':{str(i):{'precision':float(a),'recall':float(b),'f1':float(c),'support':int(d)} for i,a,b,c,d in zip(range(1,6),cm_pr,cm_rc,cm_f1,cm_sup)}}
    d={'experiment':'phase2-path-foundation-context-classifier-v1','development_fold':2,'fold3_accessed':False,'matched_type_confusion_rows_truth_columns_prediction':conf.tolist(),'paired_mask_results_same_matches':{'proposal_mask':acc(pair[0],pair[2]),'truth_mask':acc(pair[1],pair[2])},'iou_bin_matched_typing':{k:acc(*v) for k,v in bins.items()},'tissue_stratified_matched_typing':{k:acc(*v) for k,v in sorted(tissues.items())},'unmatched_retained_prediction_class_counts':dict(sorted(unmatched.items())),'dead_false_positives':{'mistyped_matched':mistyped,'unmatched_proposals':unmatched['dead'],'total':mistyped+unmatched['dead']},'full_fold2_truth_mask_metrics':truth_metrics,'geometry_invariant':{'expected':{'tp':40894,'fp':9608,'fn':18475,'detection_f1':.7444002512036844,'binary_pq':.596356927353324},'passes':exact_geometry({'tp':40894,'fp':9608,'fn':18475,'detection_f1':.7444002512036844,'binary_pq':.596356927353324})}}
    (EXP/'diagnostics.json').write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps({'matches':pair[2],'truth_metrics':truth_metrics,'unmatched':sum(unmatched.values())},indent=2))
if __name__=='__main__': main()
