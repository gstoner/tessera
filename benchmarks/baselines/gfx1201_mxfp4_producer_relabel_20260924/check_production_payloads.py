"""Rebuild each recorded production WMMA kernel twice; compare payload digests to the packet."""
import hashlib, json, sys
from dataclasses import replace
from pathlib import Path
from tessera.compiler.rocm_mxfp4_native import package_mxfp4_w4a8_wmma, select_mxfp4_schedule

packet = json.loads(Path("benchmarks/baselines/gfx1201_mxfp4_kstep_prefill_20260922/evidence.json").read_text())
ok = True
for row in packet["rows"]:
    if row["engine"] != "tessera":
        continue
    _, shape = row["case"].split("_", 1)
    m, n, k = (int(x) for x in shape.split("x"))
    rec = row["metadata"]["schedule"]
    sched = replace(select_mxfp4_schedule(m, n, k), **{f: rec[f] for f in (
        "group_m", "split_k", "stages", "waves_per_eu", "cache_modifier", "lds_pad_dwords", "k_step_schedule")})
    digests = []
    for _ in range(2):
        pkg = package_mxfp4_w4a8_wmma(m, n, k, schedule=sched)
        digests.append(hashlib.sha256(pkg.image.payload).hexdigest())
    same = digests[0] == digests[1] == row["metadata"]["image_sha256"]
    ok &= same
    print(f"{row['case']}: producer={pkg.image.pipeline_name} rebuild={digests[0][:16]} rebuild2={digests[1][:16]} recorded={row['metadata']['image_sha256'][:16]} identical={same}")
print("ALL_IDENTICAL" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
