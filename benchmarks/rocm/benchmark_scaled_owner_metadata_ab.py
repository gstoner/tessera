"""Matched gfx1201 public-owner metadata A/B, with unchanged native packages."""
import argparse
import ctypes as c
import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from tessera import runtime as rt
from tessera.compiler import native_scaled_program as native
from benchmarks.rocm import benchmark_scale_only_batch as recorder

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if rt._rocm_live_arch()!="gfx1201":raise RuntimeError("owning gfx1201 required")
    name="tessera.compiler.native_scaled_metadata_control"
    spec=importlib.util.spec_from_file_location(name,args.control)
    legacy=importlib.util.module_from_spec(spec);sys.modules[name]=legacy
    spec.loader.exec_module(legacy)
    candidate=native.PreparedScaledProgram
    runs=[]
    try:
        for round_index,order in enumerate((("control","candidate"),("candidate","control"))):
            for arm in order:
                owner=legacy.PreparedScaledProgram if arm=="control" else candidate
                native.PreparedScaledProgram=owner;recorder.PreparedScaledProgram=owner
                rows=[recorder.record(role,kind,warm_samples=9) for role in range(4)
                      for kind in ("primal","jvp","vjp")]
                runs.append({"round":round_index,"arm":arm,"rows":rows})
                args.output.write_text(json.dumps({"runs":runs},indent=2,allow_nan=False)+"\n")
    finally:
        native.PreparedScaledProgram=candidate;recorder.PreparedScaledProgram=candidate
    pairs=[]
    for round_index in range(2):
        a=next(run for run in runs if run["round"]==round_index and run["arm"]=="control")
        b=next(run for run in runs if run["round"]==round_index and run["arm"]=="candidate")
        for control,new in zip(a["rows"],b["rows"],strict=True):
            for field in ("kind","batch_owner_operand","input_shapes","output_shapes",
                          "native_program_sha256","member_image_sha256","frontend_graph_sha256"):
                assert control[field]==new[field],field
            pairs.append({"round":round_index,"kind":new["kind"],
                "batch_owner_operand":new["batch_owner_operand"],
                "control_over_candidate_warm_wall":control["warm_public_wall_median_ms"]/new["warm_public_wall_median_ms"]})
    hip=rt._load_hip_for_launch()
    ordinal=c.c_int();device=c.create_string_buffer(256);uuid=(c.c_ubyte*16)()
    if hip is None or hip.hipInit(0) or hip.hipGetDevice(c.byref(ordinal)) or hip.hipDeviceGetName(device,256,ordinal.value) or hip.hipDeviceGetUuid(c.byref(uuid),ordinal.value):
        raise RuntimeError("owning live GPU identity unavailable")
    lib=Path(rt._load_rocm_native_movement_runtime()._name)
    import os
    packet={"schema":"tessera.scaled_owner_metadata_ab.v1","architecture":"gfx1201",
        "device":device.value.decode(),"device_uuid":bytes(uuid).hex(),"device_ordinal":ordinal.value,
        "compiler_sha256":hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        "runtime_sha256":hashlib.sha256(lib.read_bytes()).hexdigest(),
        "control_source_sha256":hashlib.sha256(args.control.read_bytes()).hexdigest(),
        "candidate_source_sha256":hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        "recorder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "warm_samples_per_profile":9,"runs":runs,"paired":pairs,
        "median_control_over_candidate_warm_wall":statistics.median(row["control_over_candidate_warm_wall"] for row in pairs),
        "timing_scope":"Nine public warm wall samples per profile; unchanged native program/image/Graph hashes. Two counterbalanced rounds. Native event windows recorded separately, not a kernel speedup comparison. Cold wall timing is diagnostic only and excluded from the ratio."}
    args.output.write_text(json.dumps(packet,indent=2,allow_nan=False)+"\n")

if __name__=="__main__":main()
