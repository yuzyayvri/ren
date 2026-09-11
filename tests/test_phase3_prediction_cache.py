import numpy as np
import pytest

from scripts.phase3_prediction_cache import publish, read


def test_cache_roundtrip_and_empty_image(tmp_path):
 publish(tmp_path,['a','b'],np.array([[0,0,1,1]],np.float32),np.array([1]),np.array([.5],np.float32),np.array([0,1,1]),{'checkpoint':'x'})
 d,_=read(tmp_path);assert d['offsets'].tolist()==[0,1,1]
def test_cache_rejects_tampering(tmp_path):
 publish(tmp_path,['a'],np.array([[0,0,1,1]],np.float32),np.array([1]),np.array([.5],np.float32),np.array([0,1]),{})
 with open(tmp_path/'payload.npz','ab') as f:f.write(b'x')
 with pytest.raises(RuntimeError):read(tmp_path)
def test_cache_single_commit(tmp_path):
 publish(tmp_path,['a'],np.array([[0,0,1,1]],np.float32),np.array([1]),np.array([.5],np.float32),np.array([0,1]),{})
 with pytest.raises(RuntimeError):publish(tmp_path,['a'],np.zeros((0,4)),np.zeros(0),np.zeros(0),np.array([0,0]),{})
