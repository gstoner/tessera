"""Interleaved duration admission preserves complete pairs and rejected history."""
import pytest
from benchmarks.rocm.record_gfx1201_interleaved_compiler_formats import collect_interleaved_windows

class FakeGraphs:
    def __init__(self,short_first=False):
        self.short_first=short_first
        self.calls=[]
    def window(self,engine,launches,*,bracketed):
        self.calls.append((engine,launches))
        # Slow startup calibrations must not make later short windows admissible.
        per_launch=.5 if len(self.calls)<=6 else .01
        return dict(device_window_ms=launches*per_launch,launches=launches)

def test_short_pair_series_is_retained_and_both_arms_recaptured():
    graph=FakeGraphs();checked=[]
    samples,orders,admission=collect_interleaved_windows(graph,{"candidate":"a","reference":"b"},
        windows=3,min_window_ms=6,verify_output=checked.append)
    assert admission["status"]=="duration_admitted"
    assert admission["rejected_series"]
    for rejected in admission["rejected_series"]:
        assert len(rejected["samples"]["candidate"])==3
        assert len(rejected["samples"]["reference"])==3
        assert rejected["orders"]==[["candidate","reference"],["reference","candidate"],["candidate","reference"]]
    assert {s["launches"] for arm in samples.values() for s in arm}=={admission["launches"]}
    assert min(s["device_window_ms"] for arm in samples.values() for s in arm)>=6
    assert len(checked)==len(graph.calls)

def test_no_duration_admission_when_launch_cap_cannot_cover_floor():
    graph=FakeGraphs()
    with pytest.raises(RuntimeError,match="no timing series admitted"):
        collect_interleaved_windows(graph,{"candidate":"a","reference":"b"},
            windows=3,min_window_ms=1e6,verify_output=lambda _:None)

def test_nonfinite_calibration_is_rejected():
    class Broken:
        def window(self,*args,**kwargs):return {"device_window_ms":float("nan")}
    with pytest.raises(RuntimeError,match="positive finite"):
        collect_interleaved_windows(Broken(),{"candidate":"a","reference":"b"},
            windows=3,min_window_ms=6,verify_output=lambda _:None)
