"""Inference-only repaired fold2 evaluator; no fitting or selection."""
import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from appearance_context_features import PatchAppearanceContext
from evaluate_hover_checkpoint import fold2_arrays
from evaluate_instance_suppression import false_positive_categories, retain
from nucleus_evaluation import (
    add_scores,
    empty_score,
    match_instances,
    score_instances,
    summarize_score,
)
from verify_phase2_appearance_context_cache import load_dataset

ROOT=Path('artifacts/phase2_validity_filter_repair_v1'); CACHE=Path('artifacts/phase2_appearance_context_classifier_v1/cache'); PREF=ROOT/'prefilter_fold2'; EMB=np.load('embeddings/path-foundation/fold2/embeddings.npy',mmap_mode='r')
TYPE=Path('artifacts/phase2_path_foundation_context_classifier_v1/epoch_30.pt'); OLD=Path('artifacts/phase2_nonlinear_classifier_v1/fold2_predicted_cache.pkl')
NAMES={1:'neoplastic',2:'inflammatory',3:'connective',4:'dead',5:'epithelial'}
def added_row(amap, ctx, patch, ident, mask):
    value=amap.get((int(patch),int(ident)))
    return value if value is not None else ctx.features(mask)
def main(output_path=None):
    sc=np.load(ROOT/'fit.npz'); head=torch.nn.Sequential(torch.nn.Linear(532,64),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(64,5)); head.load_state_dict(torch.load(TYPE,weights_only=True)); head.eval()
    images,_=fold2_arrays(); old_added=load_dataset(CACHE,'fold2_predicted'); amap={(int(p),int(i)):a for p,i,a in zip(old_added['patch_index'],old_added['instance_id'],old_added['added_features'])}
    pm=json.loads((PREF/'manifest.json').read_text())
    if not pm.get('complete') or pm.get('np_threshold') != .6: raise RuntimeError('prefilter cache incomplete or threshold mismatch')
    records={}
    for meta in pm['shards']:
        p=PREF/f"shard_{int(meta['shard']):04d}.pkl"
        if hashlib.sha256(p.read_bytes()).hexdigest()!=meta['sha256']: raise RuntimeError(f'cache hash mismatch: {p}')
        records.update(pickle.loads(p.read_bytes()))
    if sorted(records)!=list(range(2523)): raise RuntimeError('cache patch coverage mismatch')
    total=empty_score(); cats=Counter(); confusion=np.zeros((5,5),int); unmatched=Counter()
    digest=hashlib.sha256(); ignored_total=ignored_retained=ignored_rejected=0; all_pred=all_truth=0
    tissue_scores={}
    for patch in sorted(records):
        r=records[patch]; base=np.asarray(r['features'],np.float32); validity_input=np.column_stack((base,np.asarray(r['confidence_features'],np.float32))); assert validity_input.shape[1]==111; z=((validity_input-sc['mean'])/sc['scale'])@sc['coefficients'].T+sc['intercept']; prob=(1/(1+np.exp(-z[:,0]))).astype(np.float32) if len(base) else np.empty(0,np.float32); keep=prob>=.35
        kept=np.flatnonzero(keep); missing=[i for i in kept if (int(patch),int(r['ids'][i])) not in amap]; ctx=PatchAppearanceContext(images[int(patch)].astype(np.float32)/255.0) if missing else None; added=np.vstack([added_row(amap,ctx,patch,r['ids'][i],r['prediction']==int(r['ids'][i])) for i in kept]) if len(kept) else np.empty((0,41),np.float32); assert base.shape[1]==107 and added.shape[1]==41; x=np.vstack([np.concatenate((base[i],added[j],EMB[int(patch)])) for j,i in enumerate(kept)]).astype(np.float32) if len(kept) else np.empty((0,532),np.float32)
        with torch.inference_mode(): logits=head(torch.from_numpy(((x-np.load(TYPE.parent/'standardizer.npz')['mean'])/np.load(TYPE.parent/'standardizer.npz')['scale']).astype(np.float32))).numpy() if len(x) else np.empty((0,5),np.float32)
        types=logits.argmax(1)+1 if len(x) else np.empty(0,np.int64)
        ignored_only=np.array([not np.any((r['prediction']==i)&~r['ignored']) for i in r['ids']], dtype=bool)
        ignored_total += int(ignored_only.sum()); ignored_retained += int((ignored_only & keep).sum()); ignored_rejected += int((ignored_only & ~keep).sum())
        filtered=retain(r['prediction'],r['ids'],keep); ptypes={int(r['ids'][i]):int(t) for i,t in zip(kept,types)}
        patch_score=score_instances(filtered,r['truth'],ptypes,r['truth_types'],ignored=r['ignored']); add_scores(total,patch_score)
        tissue_scores.setdefault(str(r['tissue_label']), {'patch_count':0,'score':empty_score(),'gt_support':Counter()})
        ts=tissue_scores[str(r['tissue_label'])]; ts['patch_count']+=1; add_scores(ts['score'],patch_score); ts['gt_support'].update(map(int,r['truth_types']))
        all_pred+=int(np.sum(filtered>0)); all_truth+=len(r['truth_types']); digest.update(np.asarray(filtered,dtype=np.int32).tobytes()); digest.update(np.asarray(sorted(ptypes.items()),dtype=np.int64).tobytes())
        matches,fps,_=match_instances(filtered,r['truth'],r['ignored'])
        for pid,tid,_ in matches: confusion[int(r['truth_types'][tid])-1,ptypes[int(pid)]-1]+=1
        for pid in fps: unmatched[NAMES[ptypes[int(pid)]]]+=1
        for k,v in false_positive_categories(filtered,r['truth'],r['ignored']).items(): cats[k]+=v
    if ignored_total != ignored_retained + ignored_rejected: raise AssertionError('ignored-only accounting invariant failed')
    tissue={k:{'patch_count':v['patch_count'],'ground_truth_class_supports':dict(sorted(v['gt_support'].items())),'metrics':summarize_score(v['score'])} for k,v in sorted(tissue_scores.items())}
    metrics=summarize_score(total); gates={'detection_f1':metrics['detection']['f1']>=.70,'binary_pq':metrics['binary_pq']>=.50,'end_to_end_macro_f1':metrics['macro_f1']>=.55,'dead_recall':metrics['classes']['4']['recall']>=.20,'dead_f1':metrics['classes']['4']['f1']>=.15}
    report={'schema':'phase2-repaired-fold2-evaluation-v2','development_fold':2,'feature_width':532,'np_threshold':.60,'validity_threshold':.35,'metrics':metrics,'tissue_stratified':tissue,'matched_type_confusion_rows_truth_columns_prediction':confusion.tolist(),'unmatched_predictions_by_predicted_class':dict(sorted(unmatched.items())),'false_positive_categories':dict(sorted(cats.items())),'ignored_only_proposals':{'total':ignored_total,'retained_by_filter':ignored_retained,'rejected_by_filter':ignored_rejected,'excluded_from_scored_negatives':True},'gates':gates,'all_gates_pass':all(gates.values()),'prediction_digest':digest.hexdigest(),'fold3_accessed':False,'inference_only':True,'type_classifier_sha256':hashlib.sha256(TYPE.read_bytes()).hexdigest(),'validity_filter_sha256':hashlib.sha256((ROOT/'fit.npz').read_bytes()).hexdigest()}
    output_path = Path(output_path) if output_path else ROOT/'repaired_fold2_report.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(report,sort_keys=True,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    main(parser.parse_args().output)
