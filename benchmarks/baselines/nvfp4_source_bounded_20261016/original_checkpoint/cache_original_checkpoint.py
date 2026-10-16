"""Reuse a single hash-checked scratch cache of pinned source tensor ranges."""
from pathlib import Path
import hashlib
import json
import numpy as np
from benchmarks.rocm import benchmark_rocm_nvfp4_checkpoint as checkpoint

root=Path("/home/angstorms/scratch/nvfp4-source-bounded-20261016/original-checkpoint")
root.mkdir(exist_ok=True)
cache=root/"ranges";cache.mkdir(exist_ok=True)
fetch=checkpoint._fetch_range
get_json=checkpoint._get_json

def cached(url,start=None,end=None):
    key=hashlib.sha256(json.dumps([url,start,end]).encode()).hexdigest()
    data_path,meta_path=cache/(key+".bin"),cache/(key+".json")
    if data_path.exists() and meta_path.exists():
        data=data_path.read_bytes();meta=json.loads(meta_path.read_text())
        assert meta["sha256"]==hashlib.sha256(data).hexdigest()
        assert meta["request"]==[url,start,end]
    else:
        data=get_json(url)[1] if start is None else fetch(url,start,end)
        data_path.write_bytes(data)
        meta_path.write_text(json.dumps({"request":[url,start,end],
            "sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)},indent=2)+"\n")
    if start is not None:
        assert len(data)==end-start+1
    return data
checkpoint._fetch_range=lambda url,start,end:cached(url,start,end)
checkpoint._get_json=lambda url:(lambda b:(json.loads(b),b))(cached(url))
for role in ("gate_proj","up_proj"):
    output=root/(role+".npz")
    if output.exists():
        print(role,"existing projection retained",flush=True);continue
    item=checkpoint._load_projection("model.layers.0.mlp."+role+".weight")
    p=item["projection"]
    np.savez(output,codes=p.packed_codes,scale_bits=p.e4m3_scales.view(np.uint8),
             global_scale=np.array(p.global_scale,np.float64),
             bf16_bits=np.asarray(item["bf16"]).view(np.uint16))
    source=item["source"]
    (root/(role+".json")).write_text(json.dumps(source,indent=2)+"\n")
    print(role, p.packed_codes.shape, "pinned source cached",flush=True)
