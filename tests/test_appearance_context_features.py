import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from appearance_context_features import appearance_context_features


def test_width_finite_deterministic_and_contrast():
    im=np.zeros((32,32,3),np.float32); im[:]=.2; im[10:20,10:20]=.8
    m=np.zeros((32,32),bool); m[10:20,10:20]=1
    x=appearance_context_features(m,im)
    assert x.shape==(41,) and np.isfinite(x).all() and np.array_equal(x,appearance_context_features(m,im))
    assert x[-7] < 0
def test_tiny_boundary_and_id_invariance():
    im=np.ones((8,8,3),np.float32); a=np.zeros((8,8),bool); a[0,0]=1
    assert np.isfinite(appearance_context_features(a,im)).all()
def test_shape_changes():
    im=np.ones((32,32,3),np.float32); c=np.zeros((32,32),bool); c[10:22,10:22]=1; e=np.zeros_like(c); e[15:17,3:29]=1
    assert appearance_context_features(e,im)[0] > appearance_context_features(c,im)[0]
