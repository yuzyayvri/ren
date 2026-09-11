import hashlib
import json
from pathlib import Path

import numpy as np

out = Path('artifacts/phase2_proposal_trained_classifier_v1')
p = out / 'fold1_matched_proposal_cache.npz'
with np.load(p) as d:
    rows = len(d['features'])
    width = d['features'].shape[1]
    assert width == 107 and np.isfinite(d['features']).all()
    assert len(set(zip(d['patch_index'].tolist(), d['proposal_id'].tolist()))) == rows
sha = lambda x: hashlib.sha256(x.read_bytes()).hexdigest()
(out / 'fold1_matched_proposal_manifest.json').write_text(json.dumps({'schema':'fold1-matched-proposal-cache-v1','rows':rows,'feature_width':width,'unique_proposal_matches':rows,'match_rule':'strict one-to-one IoU > 0.5','zero_evaluable_excluded':True,'unmatched_excluded':True,'cache_sha256':sha(p),'segmentation_sha256':sha(Path('artifacts/phase2_development_run_v10_segmentation_only/candidate_detection.pt')),'validity_filter_sha256':sha(Path('artifacts/phase2_instance_suppression_v1/validity_filter.npz')),'fold3_accessed':False}, indent=2) + '\n')
