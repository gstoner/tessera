"""Matched ordinary bounded JIT with and without native dynamic ownership."""
from pathlib import Path
import hashlib
import json
import statistics

root=Path(__file__).parent
comparisons=[]
def key(row):
    return row["dtype"],row["kind"],row["fused"],tuple(row["active_mkn"])

for prepared_name,control_name in (("prepared","control"),("prepared_reverse","control_reverse")):
    prepared,control=(json.loads((root/(name+".json")).read_text()) for name in (prepared_name,control_name))
    for field in ("gpu","compiler_sha256","runtime_sha256","source_sha256"):
        if prepared[field]!=control[field]:raise ValueError("matched "+field+" differs")
    for path,digest in prepared["source_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError("current source differs: "+path)
    pp,cc=({key(row):row for row in packet["rows"]} for packet in (prepared,control))
    if set(pp)!=set(cc) or len(pp)!=48 or len(prepared["rows"])!=48 or len(control["rows"])!=48:
        raise ValueError("matched envelope incomplete")
    expected={(dtype,kind,fused,shape)
        for dtype in ("fp16","bf16")
        for kind in ("rmsnorm","layernorm","softmax")
        for fused in (False,True)
        for shape in ((128,1024,64),(17,35,19),(1,1,1),(63,511,31))}
    if set(pp)!=expected:raise ValueError("declared envelope differs")
    if prepared["schema"]!="tessera.nvidia.bounded_lhs_jit.v1" or control["schema"]!=prepared["schema"]:
        raise ValueError("packet schema differs")
    groups={}
    rows=[]
    for identity in sorted(pp):
        p,c=pp[identity],cc[identity]
        for field in ("image_digests","schedule_digests","abi_ids","contract_digest","capacity_mkn"):
            if p[field]!=c[field]:raise ValueError("matched package "+field+" differs")
        if p["public_binding"]!="prepared_cpp_dynamic_tensor_matmul":
            raise ValueError("prepared arm lost native dynamic owner")
        if c["public_binding"]=="prepared_cpp_dynamic_tensor_matmul" or c["owner_scratch_stats"] is not None:
            raise ValueError("control arm used native dynamic owner")
        if not isinstance(p["owner_scratch_stats"],list) or len(p["owner_scratch_stats"])!=2:
            raise ValueError("prepared scratch witness missing")
        groups.setdefault(identity[:3],[]).append(p)
        rows.append(dict(dtype=identity[0],kind=identity[1],fused=identity[2],active_mkn=list(identity[3]),
            prepared_wall_ms=p["public_warm_wall_median_ms"],control_wall_ms=c["public_warm_wall_median_ms"],
            warm_prepared_over_control=p["public_warm_wall_median_ms"]/c["public_warm_wall_median_ms"],
            producer_event_ms=p["component_event_dispatch_median_ms"]["producer"],
            consumer_event_ms=p["component_event_dispatch_median_ms"]["consumer"]))
    for group in groups.values():
        if any(row["owner_scratch_stats"]!=group[0]["owner_scratch_stats"] for row in group):
            raise ValueError("bounded shape reuse unexpectedly grew retained scratch")
        for field in ("image_digests","schedule_digests","abi_ids","contract_digest"):
            if any(row[field]!=group[0][field] for row in group):
                raise ValueError("active shape altered bounded package identity")
    comparisons.append(dict(prepared=prepared_name,control=control_name,rows=rows,
        median_warm_ratio=statistics.median(row["warm_prepared_over_control"] for row in rows)))
result=dict(comparisons=comparisons,scope="Identical-image native dynamic host ownership; complete warm ordinary JIT wall, separate dispatch events; no kernel or format promotion")
(root/"analysis.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
for value in comparisons:print(value["prepared"],value["median_warm_ratio"])
