from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
out=Path(__file__).resolve().parent
driver=Path("/proc/707273/cmdline")
while driver.exists():
    try:
        command=driver.read_bytes()
    except FileNotFoundError:
        break
    if b"run-frozen-cu-grid.py" not in command:break
    time.sleep(5)
expected="8e53d33556d04066c4a6cd0335b7be1b164b1f80"
before=json.loads((out/"source-before.json").read_text())
after=json.loads((out/"source-after.json").read_text())
assert before==after and before["head"]==expected and not before["status"]
assert len(json.loads((out/"paired-summary.json").read_text()))==22
def state():
    return {"head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "status":subprocess.check_output(["git","status","--porcelain"],text=True).strip(),
        "compiler_sha256":hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest()}
initial=state()
assert initial["head"]==expected and not initial["status"]
record={"revision":expected,"source_before":initial,"state":"running","phases":[]}
def save():
    (out/"post-grid-validation.json").write_text(json.dumps(record,indent=2)+"\n")
phases=[
 ("owning-device",[sys.executable,"-m","pytest","-q","tests/device/rocm/test_mxfp4_folded_native_package.py"]),
 ("format-images",[sys.executable,str(out/"compare-format-controls.py")]),
 ("full-unit",[sys.executable,"-m","pytest","-q","tests/unit","-m","not slow"]),
]
save()
for name,command in phases:
    phase={"name":name,"command":command,"state":"running","started":time.time()}
    record["phases"].append(phase);save()
    start=time.perf_counter()
    with (out/(name+".log")).open("w") as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    phase.update(state="passed" if result.returncode==0 else "failed",
        returncode=result.returncode,elapsed_seconds=time.perf_counter()-start,
        tail="\n".join((out/(name+".log")).read_text().splitlines()[-8:]))
    record["source_after"]=state();save()
    assert record["source_after"]==initial,record
    if result.returncode:
        record["state"]="failed";save();sys.exit(result.returncode)
record["state"]="passed";save()
