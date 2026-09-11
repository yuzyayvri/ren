"""Fail-closed recovery state machine; predictor is injected by the caller."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.phase3_ap50 import ap50
from scripts.phase3_prediction_cache import publish, read


def authorize(root, bindings):
 root=Path(root); root.mkdir(parents=True,exist_ok=True); p=root/'recovery_authorization.json'
 if p.exists(): raise RuntimeError('recovery already authorized')
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({'authorization':'one frozen test evaluation with a disclosed recovery replay','bindings':bindings},sort_keys=True,indent=2)+'\n');tmp.replace(p);return p
def capture_once(root, identities, predictions, binding):
 root=Path(root); marker=root/'recovery_attempt.json'
 if marker.exists(): raise RuntimeError('recovery replay already attempted')
 tmp=marker.with_suffix('.tmp');tmp.write_text(json.dumps({'state':'started','count':1})+'\n');tmp.replace(marker)
 try:
  publish(root,identities,predictions['boxes'],predictions['classes'],predictions['confs'],predictions['offsets'],binding)
 except Exception:
  (root/'recovery_failure.marker').write_text('capture failed\n'); raise
 (root/'recovery_success.marker').write_text('capture committed\n')
 return read(root)
def require_reproduction(actual, expected):
 for key,value in expected.items():
  if actual.get(key)!=value: raise RuntimeError(f'reproduction mismatch: {key}')


def capture_production(root, checkpoint, source_manifest, bindings, *, predictor=None):
    """Capture once from the manifest; predictor injection keeps tests offline."""
    root = Path(root)
    auth = authorize(root, bindings)
    manifest = json.loads(Path(source_manifest).read_text())
    rows = manifest["txl_pbc"]["splits"]["test"]
    if [r["id"] for r in rows] != sorted(r["id"] for r in rows):
        raise RuntimeError("canonical test identities are not ordered")
    if predictor is None:
        from ultralytics import YOLO
        predictor = YOLO(str(checkpoint), task="detect")
        def run(path):
            result = predictor.predict(str(path), conf=.001, iou=.70, max_det=300, device="cpu", verbose=False)[0]
            return result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()
    else:
        run = predictor
    boxes, classes, confs, offsets = [], [], [], [0]
    for row in rows:
        b, c, s = run(row["image"])
        boxes.extend(np.asarray(b, dtype=np.float32).tolist()); classes.extend(np.asarray(c, dtype=np.int8).tolist()); confs.extend(np.asarray(s, dtype=np.float32).tolist()); offsets.append(len(boxes))
    capture_once(root, [r["id"] for r in rows], {"boxes": np.asarray(boxes, np.float32).reshape((-1, 4)), "classes": np.asarray(classes, np.int8), "confs": np.asarray(confs, np.float32), "offsets": np.asarray(offsets, np.int64)}, bindings)
    return auth


def score_committed(root, records, originals, bindings):
    """Score only a committed cache; refuse publication on reproduction drift."""
    root = Path(root); read(root)
    actual = _aggregate(records, .65)
    require_reproduction(actual, originals)
    per_class, macro = ap50(records)
    report = {'disclosure': 'one frozen test evaluation with a disclosed recovery replay', 'bindings': bindings, 'original_reproduction': actual, 'ap50_per_class': per_class, 'macro_ap50': macro}
    tmp = root / 'recovered_report.tmp'; tmp.write_text(json.dumps(report, sort_keys=True, indent=2) + '\n'); tmp.replace(root / 'recovered_report.json')
    (root / 'recovery_complete.marker').write_text('recovered report committed\n')
    return report


def _aggregate(records, threshold):
    per = {c: [0, 0, 0] for c in range(3)}
    for ident, truth, pred in records:
        used = set()
        for c, box, conf in sorted(pred, key=lambda x: -x[2]):
            choices = [(box_iou(box, tb), j) for j, (tc, tb) in enumerate(truth) if tc == c and j not in used]
            best = max(choices, default=(0, -1))
            if conf >= threshold and best[0] >= .5: used.add(best[1]); per[c][0] += 1
            elif conf >= threshold: per[c][1] += 1
        for j, (c, _) in enumerate(truth):
            if j not in used: per[c][2] += 1
    tp = sum(x[0] for x in per.values()); fp = sum(x[1] for x in per.values()); fn = sum(x[2] for x in per.values()); fs=[]; recalls=[]
    for t,p,n in per.values():
        r=t/(t+n) if t+n else 0; q=t/(t+p) if t+p else 0; recalls.append(r); fs.append(2*q*r/(q+r) if q+r else 0)
    return {'macro_f1': float(np.mean(fs)), 'min_recall': float(min(recalls)), 'overall_recall': float(tp/(tp+fn) if tp+fn else 0), 'per_class_recall': recalls, 'tp': tp, 'fp': fp, 'fn': fn}


def box_iou(a, b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0])); y=max(0,min(a[3],b[3])-max(a[1],b[1])); z=x*y
    return z/max(1e-12,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-z)
