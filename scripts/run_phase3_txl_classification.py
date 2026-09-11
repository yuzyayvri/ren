"""Frozen Path Foundation ground-truth-crop TXL classification protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from scripts.phase3_artifact_manifest import (
    atomic_manifest,
    mark_test_exposed,
    require_unexposed,
    sha256,
)
from scripts.phase3_frozen_heads import crop_txl, normalize_for_path_foundation
from scripts.phase3_selection import C_GRID

ROOT = Path("/home/yuzy/ren")
MODEL = ROOT / "models/vision/path-foundation"
MANIFEST = ROOT / "artifacts/phase3_protocol_v1/source_manifest.json"


def _rows(split):
    return json.loads(MANIFEST.read_text())["txl_pbc"]["splits"][split]


def _crops(row):
    image = np.asarray(Image.open(ROOT / row["image"]).convert("RGB"), dtype=np.uint8)
    out, labels = [], []
    for line in (ROOT / row["label"]).read_text().splitlines():
        c, x, y, w, h = map(float, line.split())
        iw, ih = image.shape[1], image.shape[0]
        out.append(crop_txl(image, ((x-w/2)*iw, (y-h/2)*ih, (x+w/2)*iw, (y+h/2)*ih)))
        labels.append(int(c))
    return out, labels


def _config():
    return {"schema":"phase3-txl-frozen-path-foundation-v1", "crop":"10% expansion per side; clip; square median pad; 224 bilinear", "input":"RGB float32 [0,1]", "classes":["WBC","RBC","Platelets"], "model":str(MODEL), "model_tree_sha256":_tree_hash(MODEL), "seed":20260909}


def _tree_hash(path):
    h=hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file(): h.update(str(p.relative_to(path)).encode()); h.update(sha256(p).encode())
    return h.hexdigest()


def _embedder():
    import tensorflow as tf
    model=tf.saved_model.load(str(MODEL)); sig=model.signatures.get("serving_default")
    if sig is None: sig=next(iter(model.signatures.values()))
    name=next(iter(sig.structured_input_signature[1]))
    def run(x):
        out=sig(**{name:tf.convert_to_tensor(normalize_for_path_foundation(x))})
        y=next(iter(out.values())).numpy()
        if y.ndim==3: y=y.mean(axis=1)
        if y.shape[1:] != (384,): raise RuntimeError(f"unexpected Path Foundation output {y.shape}")
        return np.asarray(y,np.float32)
    return run


def _extract(split, run, out):
    rows=_rows(split); ids=[]; ys=[]; chunks=[]
    for row in rows:
        crops, labels=_crops(row)
        if crops: chunks.append(run(np.asarray(crops))); ys.extend(labels); ids.extend([f"{row['id']}:{i}" for i in range(len(labels))])
    emb=np.concatenate(chunks).astype(np.float32) if chunks else np.empty((0,384),np.float32)
    tmp=out.with_suffix('.tmp.npz'); np.savez_compressed(tmp, embeddings=emb, labels=np.asarray(ys,np.int8), identities=np.asarray(ids)); tmp.replace(out)
    return emb, np.asarray(ys,np.int8)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True); ap.add_argument('--batch-size',type=int,default=32); args=ap.parse_args(); run=args.run; run.mkdir(parents=True,exist_ok=True)
    cfg=_config(); cfg_path=run/'config.json'; cfg_path.write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n')
    encoder=_embedder()
    tr,yt=_extract('train',encoder,run/'train_embeddings.npz'); va,yv=_extract('val',encoder,run/'val_embeddings.npz')
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler().fit(tr); ztr=sc.transform(tr); zva=sc.transform(va); candidates=[]
    for c in C_GRID:
        m=LogisticRegression(C=c,class_weight='balanced',max_iter=2000,random_state=20260909,solver='lbfgs').fit(ztr,yt); pred=m.predict(zva); candidates.append((f1_score(yv,pred,average='macro'),-c,c,m,pred))
    score,_,c,model,pred=max(candidates,key=lambda x:x[:2]);
    atomic_manifest(run/'selection.json',stage='phase3-txl-classification',config={'C_grid':list(C_GRID),'selected_C':c,'validation_macro_f1':float(score)},inputs={'source_manifest_sha256':sha256(MANIFEST),'model_tree_sha256':cfg['model_tree_sha256']})
    with (run/'head.pkl.tmp').open('wb') as f: pickle.dump(model,f)
    (run/'head.pkl.tmp').replace(run/'head.pkl'); np.savez_compressed(run/'standardizer.npz.tmp.npz',mean=sc.mean_,scale=sc.scale_); (run/'standardizer.npz.tmp.npz').replace(run/'standardizer.npz')
    require_unexposed(run/'test_exposed.marker'); mark_test_exposed(run/'test_exposed.marker',sha256(run/'selection.json'))
    te,yt2=_extract('test',encoder,run/'test_embeddings.npz'); pt=model.predict((te-sc.mean_)/sc.scale_); rec=recall_score(yt2,pt,average=None,labels=[0,1,2]); report={'validation_macro_f1':float(score),'selected_C':float(c),'test_macro_f1':float(f1_score(yt2,pt,average='macro')),'per_class_recall':rec.tolist(),'confusion_matrix':confusion_matrix(yt2,pt,labels=[0,1,2]).tolist(),'supports':np.bincount(yt2,minlength=3).tolist(),'gates':{'macro_f1':f1_score(yt2,pt,average='macro')>=.8,'recall':bool(np.all(rec>=.6))}}
    tmp=run/'report.json.tmp'; tmp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); tmp.replace(run/'report.json'); print(json.dumps(report,indent=2,sort_keys=True))

if __name__=='__main__': main()
