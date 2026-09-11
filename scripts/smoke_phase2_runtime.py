"""Bounded offline forward/backward and timing gate; never trains on source data."""
import json,time,tracemalloc,torch
from hover_fast_model import HoVerFast,IMPLEMENTATION_ID,INITIALIZATION_PROVENANCE
from hover_loss import masked_hover_loss
m=HoVerFast(); x=torch.randn(1,3,256,256); sem=torch.zeros(1,256,256,dtype=torch.long); ins=torch.zeros(1,256,256,dtype=torch.long); sem[:,40:60,40:60]=1;ins[:,40:60,40:60]=1
tracemalloc.start(); t=time.perf_counter(); out=m(x); loss=masked_hover_loss(*out,sem,ins); loss.backward(); current,peak=tracemalloc.get_traced_memory()
print(json.dumps({'implementation':IMPLEMENTATION_ID,'initialization':INITIALIZATION_PROVENANCE,'offline':True,'forward_backward_seconds':time.perf_counter()-t,'python_peak_bytes':peak,'loss':float(loss.detach())},indent=2))
