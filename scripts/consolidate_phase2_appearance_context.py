"""Write the audit comparison from preserved, already-evaluated reports."""
import json
from pathlib import Path

root=Path('artifacts'); out=root/'phase2_appearance_context_classifier_v1'
aug=json.loads((out/'report.json').read_text()); ctrl=json.loads((root/'phase2_nonlinear_classifier_v1'/'report.json').read_text())
comparison={'experiment':'phase2-classifier-comparison','development_fold':2,'fold3_accessed':False,'models':{'adjusted_logistic':{'source':'artifacts/phase2_instance_suppression_v1/full_fold2_report.json'},'truth_trained_107_mlp':ctrl.get('full_fold2_mlp',{}),'appearance_context_148_mlp':aug.get('full_fold2_mlp',{})},'geometry':{'tp':40894,'fp':9608,'fn':18475,'detection_f1':0.7444002512036844,'binary_pq':0.596356927353324},'retention':{'retained':'appearance_context_148_mlp','reason':'improves end-to-end and valid true-mask macro-F1; rare-class gates and geometry preserved','acceptance':False,'failing_gate':'end-to-end macro-F1 >= 0.55','remaining_gap':0.065886}}
(out/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
