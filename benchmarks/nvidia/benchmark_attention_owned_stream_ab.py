"""Matched public attention owner A/B; kernel packages remain identical."""
import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from tessera.compiler import resident_attention
from benchmarks.nvidia.record_jit_multiresult_owner import record
from tests.device.nvidia.test_jit_multiresult_attention_vjp import function

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    name="tessera.compiler.resident_attention_context_control"
    spec=importlib.util.spec_from_file_location(name,args.control)
    legacy=importlib.util.module_from_spec(spec)
    sys.modules[name]=legacy
    spec.loader.exec_module(legacy)
    candidate=resident_attention.ResidentAttentionTape
    runs=[]
    # Portable program digests include cold/warm compile state. Pin each actual
    # traced package once so both runtime arms consume identical serialized bytes.
    cls=type(function(False,False))
    compile_program=cls.compile_native_attention_vjp
    packages={}
    def pinned(self,*inputs,**options):
        source=self._specialized_autodiff_module(inputs,{}).to_mlir()
        key=(source,tuple(sorted(options.items())))
        if key not in packages:
            packages[key]=compile_program(self,*inputs,**options)
        return packages[key]
    cls.compile_native_attention_vjp=pinned
    try:
        for round_index,order in enumerate((("control","candidate"),("candidate","control"))):
            for arm in order:
                resident_attention.ResidentAttentionTape=legacy.ResidentAttentionTape if arm=="control" else candidate
                packet=record(samples=3,reps=20)
                packet["arm"]=arm;packet["round"]=round_index
                runs.append(packet)
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps({"runs":runs},indent=2,allow_nan=False)+"\n")
    finally:
        resident_attention.ResidentAttentionTape=candidate
        cls.compile_native_attention_vjp=compile_program
    paired=[]
    for round_index in range(2):
        a=next(p for p in runs if p["round"]==round_index and p["arm"]=="control")
        b=next(p for p in runs if p["round"]==round_index and p["arm"]=="candidate")
        assert a["gpu"]==b["gpu"] and a["binaries"]==b["binaries"]
        for control,candidate_row in zip(a["rows"],b["rows"],strict=True):
            keys=("shape","bias","causal","compact","active","program_digest","checkpoint_identity")
            assert all(control[k]==candidate_row[k] for k in keys)
            assert control["ancestry"]==candidate_row["ancestry"]
            paired.append({"round":round_index,**{k:control[k] for k in keys},
                "control_over_candidate":{k:control["medians"][k]/candidate_row["medians"][k]
                    for k in ("capture_wall_ms","backward_wall_ms","paired_wall_ms")}})
    result={"schema":"tessera.attention_owned_stream_ab.v1",
        "control_source_sha256":hashlib.sha256(args.control.read_bytes()).hexdigest(),
        "candidate_source_sha256":hashlib.sha256(Path(resident_attention.__file__).read_bytes()).hexdigest(),
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runs":runs,"paired":paired,
        "median_control_over_candidate":{key:statistics.median(row["control_over_candidate"][key] for row in paired)
            for key in ("capture_wall_ms","backward_wall_ms","paired_wall_ms")},
        "timing_scope":"Public capture/backward/pair wall time; unchanged packages have independent resident event-dispatch windows. Two counterbalanced rounds, not a kernel speedup claim."}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")

if __name__=="__main__":main()
