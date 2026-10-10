"""Native program events are already averaged; benchmark units must preserve them."""
import json
from pathlib import Path
from types import SimpleNamespace
from benchmarks.rocm import record_independent_scaled_primal as recorder

def test_recorder_preserves_native_per_invocation_event_value(tmp_path,monkeypatch):
    row={"mask":None,"prefixes":[[],[],[],[]],"output_prefix":[],
         "transposeB":False,"encoded":False,"kind":"primal","package":{}}
    package_path=tmp_path/"packages.json"
    package_path.write_text(json.dumps({"recorder_sha256":"digest",
        "compiler_sha256":"compiler","rows":[row for _ in range(69)]}))
    monkeypatch.setattr(recorder,"digest",lambda path:"digest")
    monkeypatch.setattr(recorder,"rt",SimpleNamespace(
        _rocm_live_arch=lambda:"gfx1201",
        _load_rocm_native_movement_runtime=lambda:SimpleNamespace(_name="/mock/library")))
    class Hip:
        def __getattr__(self,name):return lambda *args:0
    monkeypatch.setattr(recorder.c,"CDLL",lambda *args:Hip())
    monkeypatch.setattr(recorder,"NativeScaledProgram",SimpleNamespace(
        from_manifest=lambda value:SimpleNamespace(images=(b"image",))))
    class Owner:
        def __init__(self,package,values,**kwargs):self.values=values;self.generation=0
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def update(self,values):self.values=values
        def invoke(self,*,repeats=1,timed=False):
            self.generation+=1
            # The C ABI has already divided its event window by repeats.
            return self.generation,7.5 if timed else None
        def read(self,generation):
            if generation!=self.generation:raise RuntimeError("stale generation")
            return recorder.expected(self.values,row)
    monkeypatch.setattr(recorder,"PreparedScaledProgram",Owner)
    output=tmp_path/"result.json"
    recorder.measure(package_path,output)
    rows=json.loads(output.read_text())["rows"]
    assert len(rows)==69
    assert all(row["native_launch_window_samples_ms"]==[7.5]*5 for row in rows)
    assert all(row["native_launch_window_median_ms"]==7.5 for row in rows)
