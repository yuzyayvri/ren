"""One-shot fold1 validity-filter repair with row-level provenance."""
from __future__ import annotations

import hashlib, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from evaluate_hover_checkpoint import load_model
from evaluate_instance_suppression import confidence_features, model_features
from hover_postprocess import extract_instances
from nucleus_evaluation import match_instances
from pannuke_target_policy import make_target
from pilot_hover_fast import fold1_arrays
from train_instance_classifier import FeatureCapture, pooled_features

OUT = Path("artifacts/phase2_validity_filter_repair_v1")
SEG = Path("artifacts/phase2_development_run_v10_segmentation_only/candidate_detection.pt")
EXPECTED = (85276, 46702, 36407, 2167)

def sha_bytes(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def sha_path(p: Path) -> str: return sha_bytes(p.read_bytes())

def main() -> None:
    if OUT.exists() and (OUT / "fit.npz").exists():
        raise RuntimeError("refit output already exists; this path is one-shot")
    OUT.mkdir(parents=True, exist_ok=True)
    images, masks = fold1_arrays()
    model = load_model(SEG); model.eval(); cap = FeatureCapture(model)
    rows=[]; x_parts=[]; y_parts=[]; total=positive=negative=excluded=0
    with __import__('torch').inference_mode():
        for patch_index, (image, mask) in enumerate(zip(images, masks, strict=True)):
            semantic, truth, _ = make_target(mask, patch_index)
            fmap, np_logits, hv = model_features(model, cap, image)
            prediction = extract_instances(np_logits, hv, threshold=.60)
            ids, base = pooled_features(prediction, fmap, image.astype(np.float32)/255.0)
            conf = confidence_features(prediction, np_logits)
            features = np.column_stack((base, conf)).astype(np.float32)
            matches, _, _ = match_instances(prediction, truth, semantic == 255)
            matched = {int(pid) for pid, _, _ in matches}
            for proposal_id, feature in zip(ids.tolist(), features, strict=True):
                evaluable = bool(np.any((prediction == proposal_id) & (semantic != 255)))
                included = evaluable
                label = int(proposal_id in matched) if included else None
                reason = None if included else "ignored_only_or_zero_evaluable"
                ident = f"fold1:{patch_index}:{proposal_id}"
                feature_hash = sha_bytes(np.asarray(feature, dtype=np.float32).tobytes())
                rows.append({"fold":1,"patch_index":patch_index,"proposal_id":int(proposal_id),"evaluable":evaluable,"geometrically_matched":int(proposal_id in matched),"included":included,"training_label":label,"exclusion_reason":reason,"row_identity":ident,"feature_sha256":feature_hash})
                total += 1
                if included:
                    x_parts.append(feature); y_parts.append(label)
                    positive += label; negative += 1-label
                else: excluded += 1
            if (patch_index+1) % 256 == 0: print(f"fold1_refit={patch_index+1}/{len(images)}", flush=True)
    cap.close()
    if (total, positive, negative, excluded) != EXPECTED or total != positive + negative + excluded:
        raise RuntimeError(f"unexpected accounting {(total,positive,negative,excluded)}")
    if len({(r['patch_index'],r['proposal_id']) for r in rows}) != total:
        raise RuntimeError("proposal identity collision")
    x=np.asarray(x_parts,dtype=np.float32); y=np.asarray(y_parts,dtype=np.int8)
    if x.shape != (83109,111) or not np.isfinite(x).all(): raise RuntimeError(f"bad training matrix {x.shape}")
    scaler=StandardScaler().fit(x)
    clf=LogisticRegression(solver="lbfgs",class_weight="balanced",max_iter=500,random_state=20260909).fit(scaler.transform(x),y)
    np.savez_compressed(OUT/"fit.npz",mean=scaler.mean_,scale=scaler.scale_,coefficients=clf.coef_,intercept=clf.intercept_,classes=clf.classes_)
    np.save(OUT/"training_features.npy",x); np.save(OUT/"training_labels.npy",y)
    (OUT/"membership_ledger.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows))
    report={"schema":"phase2-validity-filter-repair-v1","fold":1,"feature_width":111,"feature_definition":"107 pooled proposal features followed by 4 confidence statistics, identical to historical filter","np_threshold":.60,"validity_threshold":.35,"counts":{"total_proposals":total,"included_positive":positive,"included_negative":negative,"excluded":excluded,"included":len(y)},"invariant":total==positive+negative+excluded,"logistic":{"solver":"lbfgs","class_weight":"balanced","max_iter":500,"random_state":20260909,"converged":bool(clf.n_iter_[0]<500),"n_iter":int(clf.n_iter_[0])},"segmentation_checkpoint":str(SEG),"segmentation_sha256":sha_path(SEG),"fit_sha256":sha_path(OUT/"fit.npz"),"scaler_sha256":sha_path(OUT/"fit.npz"),"training_features_sha256":sha_path(OUT/"training_features.npy"),"ledger_sha256":sha_path(OUT/"membership_ledger.jsonl"),"fold3_accessed":False}
    (OUT/"fit_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__ == "__main__": main()
