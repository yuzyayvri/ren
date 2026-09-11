import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from hover_fast_model import HoVerFast
from hover_loss import fixed_type_weights, masked_hover_loss
from nucleus_evaluation import (
 add_scores,
 detection_counts,
 empty_score,
 score_instances,
 summarize_score,
)


class T(unittest.TestCase):
 def test_model_preserves_branch_shapes(self):
  outputs=HoVerFast()(torch.zeros(1,3,64,64))
  self.assertEqual([tuple(x.shape) for x in outputs],[(1,1,64,64),(1,2,64,64),(1,6,64,64)])
 def test_detached_type_head_cannot_update_shared_features(self):
  model=HoVerFast(detach_type_features=True);outputs=model(torch.zeros(1,3,32,32))
  outputs[2].sum().backward()
  self.assertIsNone(model.encoder1[0].weight.grad)
  self.assertIsNotNone(model.type_head.weight.grad)
 def test_separate_type_decoder_preserves_shapes(self):
  outputs=HoVerFast(separate_type_decoder=True)(torch.zeros(1,3,64,64))
  self.assertEqual([tuple(x.shape) for x in outputs],[(1,1,64,64),(1,2,64,64),(1,6,64,64)])
 def test_all_ignored_skips_update(self):
  x=torch.zeros(1,1,2,2,requires_grad=True); s=torch.full((1,2,2),255); i=torch.zeros(1,2,2)
  self.assertIsNone(masked_hover_loss(x,torch.zeros(1,2,2,2),torch.zeros(1,6,2,2),s,i))
 def test_fixed_type_weights_are_finite_and_favor_dead(self):
  weights=fixed_type_weights(torch.device('cpu'))
  self.assertTrue(torch.isfinite(weights).all());self.assertEqual(weights[0],0)
  self.assertGreater(weights[4],weights[1])
 def test_ignored_region_does_not_hide_spurious_object(self):
  p=np.array([[1,1,2],[1,1,2]]); t=np.array([[1,1,0],[1,1,0]]); ig=np.array([[0,0,1],[0,0,1]])
  self.assertEqual(detection_counts(p,t,ig),{'tp':1,'fp':0,'fn':0,'sum_iou':1.0})
 def test_wrong_type_is_fp_and_fn(self):
  a=np.array([[1,1],[0,0]]); r=score_instances(a,a,{1:1},{1:2})
  self.assertEqual((r['classes']['1']['fp'],r['classes']['2']['fn']),(1,1))
 def test_empty_cases_are_defined(self):
  z=np.zeros((2,2),int); self.assertEqual(detection_counts(z,z)['tp'],0)
 def test_aggregate_metrics_include_binary_and_all_classes(self):
  a=np.array([[1,1,0],[0,0,2]]); raw=score_instances(a,a,{1:1,2:2},{1:1,2:2})
  total=empty_score();add_scores(total,raw);summary=summarize_score(total)
  self.assertEqual(summary['detection']['f1'],1.0);self.assertEqual(summary['binary_pq'],1.0)
  self.assertEqual(summary['classes']['1']['support'],1);self.assertEqual(summary['classes']['2']['precision'],1.0)
  self.assertEqual(summary['classes']['3']['f1'],0.0);self.assertAlmostEqual(summary['macro_f1'],0.4)
 def test_unmatched_and_wrong_types_are_end_to_end_errors(self):
  truth=np.array([[1,1,0],[0,0,2]]);pred=np.array([[1,1,3],[0,0,0]])
  raw=score_instances(pred,truth,{1:2,3:3},{1:1,2:2});total=empty_score();add_scores(total,raw);s=summarize_score(total)
  self.assertEqual(s['detection']['tp'],1);self.assertEqual(s['detection']['fp'],1);self.assertEqual(s['detection']['fn'],1)
  self.assertEqual(s['classes']['1']['fn'],1);self.assertEqual(s['classes']['2']['fp'],1)
  self.assertEqual(s['classes']['2']['fn'],1);self.assertEqual(s['classes']['3']['fp'],1)
if __name__=='__main__': unittest.main()
