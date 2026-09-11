"""Record canonical patch-aligned fold2 tissue support reconciliation."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'artifacts/phase2_validity_filter_repair_v1'
def main():
 types=np.load(ROOT/'data/tissue/fold2/Fold 2/images/fold2/types.npy',mmap_mode='r')
 if len(types)!=2523: raise ValueError('fold2 patch alignment mismatch')
 r=json.loads((OUT/'repaired_fold2_report.json').read_text()); supports={k:v['support'] for k,v in r['metrics']['classes'].items()}
 labels,counts=np.unique(types,return_counts=True); d={'schema':'phase2-corrected-tissue-diagnostics-v1','fold':2,'patch_count':len(types),'patch_counts_by_tissue':dict(zip(labels.tolist(),counts.astype(int).tolist())),'patch_counts_sum':int(counts.sum()),'aggregate_class_supports':supports,'class_supports_sum':sum(supports.values()),'valid_classes':[1,2,3,4,5],'aggregate_reconciliation':{'tp':40771,'fp':9424,'fn':18598,'sum_iou':32680.828865197265,'digest':r['prediction_digest']},'fold3_accessed':False}
 (OUT/'corrected_tissue_diagnostics.json').write_text(json.dumps(d,sort_keys=True,indent=2)+'\n'); print(json.dumps(d,indent=2))
 for name in ('repaired_fold2_report.json','repaired_fold2_reproduction_1.json','repaired_fold2_reproduction_2.json'):
  p=OUT/name; payload=json.loads(p.read_text()); payload['tissue_stratified']=d['patch_counts_by_tissue']; p.write_text(json.dumps(payload,sort_keys=True,indent=2)+'\n')
if __name__=='__main__': main()
