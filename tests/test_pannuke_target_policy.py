import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from pannuke_target_policy import IGNORE, make_target,hover_offsets

class TargetPolicyTest(unittest.TestCase):
 def test_void_is_ignored_not_negative(self):
  s, i, r = make_target(np.zeros((256,256,6)), 0)
  self.assertTrue((s == IGNORE).all()); self.assertEqual(r['valid_pixels'], 0)
 def test_overlap_excludes_whole_instances(self):
  m=np.zeros((256,256,6)); m[...,5]=1; m[0:2,0:2,0]=7; m[0,0,1]=8
  s, i, r=make_target(m,0)
  self.assertEqual(r['excluded_instances'],2); self.assertEqual(s[1,1],IGNORE)
 def test_disconnected_excludes_whole_instance(self):
  m=np.zeros((256,256,6)); m[...,5]=1; m[0,0,2]=9; m[4,4,2]=9
  s, i, r=make_target(m,0)
  self.assertEqual(r['connective'],1); self.assertEqual(s[0,0],IGNORE)
 def test_hover_offsets_are_centre_relative(self):
  i=np.zeros((256,256),np.int32);s=np.zeros((256,256),np.uint8);i[0,0:3]=1;s[0,0:3]=1
  hv=hover_offsets(i,s); self.assertAlmostEqual(float(hv[0,0,1]),0.0);self.assertLess(hv[0,0,0],0)

if __name__ == '__main__': unittest.main()
