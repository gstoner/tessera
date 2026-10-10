"""Identity-checked native-owner versus canonical descriptor comparison."""
from pathlib import Path
import hashlib
import json
import statistics

root=Path(__file__).parent
pairs=[("prepared_row","control_row"),("prepared_col","control_col"),
       ("prepared_row_reverse","control_row_reverse"),("prepared_col_reverse","control_col_reverse")]
def key(row):
    return row["dtype"],row["kind"],row["fused"],tuple(row["shape_mkn"]),row["rhs_layout"]
comparisons=[]
for prepared_name,control_name in pairs:
    prepared,control=(json.loads((root/(name+".json")).read_text())
                      for name in (prepared_name,control_name))
    for field in ("gpu","compiler_sha256","runtime_sha256","source_sha256"):
        if prepared[field]!=control[field]:raise ValueError("matched "+field+" differs")
    for path,digest in prepared["source_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError("current source differs: "+path)
    pp,cc=({key(row):row for row in packet["rows"]} for packet in (prepared,control))
    if set(pp)!=set(cc) or len(pp)!=24:raise ValueError("incomplete matched envelope")
    rows=[]
    for identity in sorted(pp):
        p,c=pp[identity],cc[identity]
        for field in ("image_digests","consumer_abi","contract_digest","compiler_path"):
            if p[field]!=c[field]:raise ValueError("matched package "+field+" differs")
        if p["public_binding"]!="prepared_cpp_tensor_matmul":
            raise ValueError("prepared arm did not execute native owner")
        if c["public_binding"]=="prepared_cpp_tensor_matmul":
            raise ValueError("control unexpectedly used prepared owner")
        rows.append(dict(dtype=identity[0],kind=identity[1],fused=identity[2],
            shape_mkn=list(identity[3]),rhs_layout=identity[4],
            prepared_wall_ms=p["warm_wall_median_ms"],control_wall_ms=c["warm_wall_median_ms"],
            warm_prepared_over_control=p["warm_wall_median_ms"]/c["warm_wall_median_ms"],
            prepared_producer_event_ms=p["resident_event_dispatch_median_ms"]["producer"],
            prepared_consumer_event_ms=p["resident_event_dispatch_median_ms"]["consumer"]))
    comparisons.append(dict(prepared=prepared_name,control=control_name,rows=rows,
        median_warm_ratio=statistics.median(row["warm_prepared_over_control"] for row in rows)))
result=dict(comparisons=comparisons,
    scope="Static primal FP16/BF16 norm/softmax->matmul native host ownership; identical physical images; no kernel or format promotion")
(root/"analysis.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
for value in comparisons:print(value["prepared"],value["median_warm_ratio"])
