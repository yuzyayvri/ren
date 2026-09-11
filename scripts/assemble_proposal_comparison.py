import json
from pathlib import Path

out=Path('artifacts/phase2_proposal_trained_classifier_v1')
proposal=json.loads((out/'report.json').read_text())
old=json.loads(Path('artifacts/phase2_score_adjustment_v1/report.json').read_text())
control=json.loads(Path('artifacts/phase2_nonlinear_classifier_v1/report.json').read_text())
comparison={'adjusted_logistic':old['full_fold2']['selected_multiclass'],'truth_trained_control_epoch_29':control['full_fold2_mlp'],'proposal_trained_intervention_epoch_30':proposal['full_fold2'],'truth_mask_control_epoch_29':control['full_fold2_true_mask'],'truth_mask_intervention_epoch_30':proposal['full_fold2_true_mask'],'geometry_invariant':{'tp':40894,'fp':9608,'fn':18475,'detection_f1':0.7444002512036844,'binary_pq':0.596356927353324},'fold2_is_development_data':True,'fold3_accessed':False}
(out/'full_fold2_classifier_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
