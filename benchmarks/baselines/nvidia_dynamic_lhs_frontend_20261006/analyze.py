"""Check packet identity, capacity reuse and component timing attribution."""
from pathlib import Path
import hashlib
import json
import statistics

root=Path(__file__).parent
packet=json.loads((root/"timing.json").read_text())
if len(packet["rows"])!=48 or packet["gpu"].split(",")[-1].strip()!="12.0":
    raise ValueError("incomplete exact-device packet")
for path,digest in packet["source_sha256"].items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
        raise ValueError("current source differs: "+path)
groups={}
for row in packet["rows"]:
    key=(row["dtype"],row["kind"],row["fused"])
    groups.setdefault(key,[]).append(row)
if len(groups)!=12:raise ValueError("producer/storage/epilogue envelope incomplete")
summaries=[]
for key,rows in groups.items():
    if {tuple(r["active_mkn"]) for r in rows}!={(128,1024,64),(17,35,19),(1,1,1),(63,511,31)}:
        raise ValueError("shape reuse envelope incomplete")
    for field in ("image_digests","schedule_digests","abi_ids","contract_digest","capacity_mkn"):
        if any(row[field]!=rows[0][field] for row in rows):
            raise ValueError("bounded reuse identity changed: "+field)
    summaries.append(dict(dtype=key[0],kind=key[1],fused=key[2],
        median_replay_wall_ms=statistics.median(r["replay_wall_median_ms"] for r in rows),
        max_abs_error=max(r["max_abs_error"] for r in rows)))
result=dict(groups=summaries,device=packet["gpu"],rows=len(packet["rows"]),
    max_abs_error=max(r["max_abs_error"] for r in packet["rows"]),
    replay_wall_range_ms=[min(r["replay_wall_median_ms"] for r in packet["rows"]),
                          max(r["replay_wall_median_ms"] for r in packet["rows"])],
    component_event_range_ms={role:[
        min(r["component_event_dispatch_median_ms"][role] for r in packet["rows"]),
        max(r["component_event_dispatch_median_ms"][role] for r in packet["rows"])]
        for role in ("producer","consumer")},
    scope="Numerical/native bounded package reuse and separate component dispatch attribution; no performance promotion")
(root/"analysis.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print(json.dumps({k:v for k,v in result.items() if k!="groups"},indent=2))
