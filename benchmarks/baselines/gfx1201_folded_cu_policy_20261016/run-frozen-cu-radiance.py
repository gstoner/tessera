"""Run production selector versus immutable WGP compiler, without Target edits."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

root = Path.cwd()
out = Path(__file__).resolve().parent / "frozen-cu-radiance"
out.mkdir(exist_ok=True)
current = Path(os.environ["TESSERA_OPT"]).resolve()
control = out.parent / "native-control/tessera-opt"
expected = os.environ["FROZEN_CU_HEAD"]
def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()
def state():
    return {"head": git("rev-parse", "HEAD"), "status": git("status", "--porcelain"),
            "compiler_sha256": hashlib.sha256(current.read_bytes()).hexdigest(),
            "control_sha256": hashlib.sha256(control.read_bytes()).hexdigest()}
before = state()
assert before["head"] == expected and not before["status"], before
assert before["control_sha256"] == "4e5eeed3ea09f496d6f67a144338ae647281618a2d0503e572ba9fbf7139fb6a"
(out / "source-before.json").write_text(json.dumps(before, indent=2)+"\n")
cases = [(256,n,k) for k in (1536,2048,5120) for n in (8192,16384)]
for reverse in (False, True):
    name = "reverse" if reverse else "forward"
    primary, reference = (control,current) if reverse else (current,control)
    command = [sys.executable, "benchmarks/rocm/record_gfx1201_folded_native_package.py",
        "--tessera-opt", str(primary), "--llvm-bin", os.environ["TESSERA_LLVM_BIN"],
        "--native-reference-opt", str(reference), "--compare-native-workgroup-mode",
        "--share-native-resident-frames", "--hip-graph-windows", "--trials", "7",
        "--radiance-module", "/home/angstorms/scratch/tessera-legacy-20260929/mxfp4-independent-build/radiance_mxfp4_fp8.so",
        "--radiance-revision", "dfdfa3832922c9a4253133f09c1f5c0d39748fc7",
        "--output", str(out / (name+".json"))]
    for m,n,k in (list(reversed(cases)) if reverse else cases):
        command += ["--case", f"prefill:{m}x{n}x{k}"]
    (out / (name+"-command.json")).write_text(json.dumps(command,indent=2)+"\n")
    with (out / (name+".log")).open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    assert state() == before, state()
    packet = json.loads((out / (name+".json")).read_text())
    assert packet["dirty_worktree"] is False and packet["source_commit"] == expected
    for row in packet["rows"]:
        frames = row["shared_native_resident_frames"]
        assert frames["closed"] and frames["leases"] == 0 and frames["copies"] == 3
        identities = {arm["image_evidence"]["resident_frame_identity"]
                      for arm in row["rows"] if arm["engine"].startswith("native_")}
        assert identities == {frames["pointer_identity_sha256"]}
summary = []
packets = {name:json.loads((out/(name+".json")).read_text()) for name in ("forward","reverse")}
for shape in cases:
    ratios = []
    for name in ("forward","reverse"):
        row = next(r for r in packets[name]["rows"] if r["shape_mnk"] == list(shape))
        times = {r["engine"]:r["device_window_per_launch_median_ms"] for r in row["graph_rows"]}
        ratio = times["native_mlir"]/times["native_reference_control"]
        ratios.append(1/ratio if name == "reverse" else ratio)
    summary.append({"shape_mnk":shape, "cu_over_wgp_captured_device_ratio":ratios})
(out/"paired-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
(out/"source-after.json").write_text(json.dumps(state(),indent=2)+"\n")
print(json.dumps(summary,indent=2),flush=True)
