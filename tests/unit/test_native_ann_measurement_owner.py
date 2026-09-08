"""A measurement can select only its own scoped native artifacts and domain."""
from types import SimpleNamespace as NS
from pathlib import Path
import hashlib
import pytest
from tessera.compiler.native_ann_gpu import NativeANNDeviceRegistration


def owner_and_reports(monkeypatch, speedup=1.1, budget=True):
    from tessera.compiler.emit import candidate
    owner=object.__new__(NativeANNDeviceRegistration)
    owner.closed=False
    owner.runner=NS(closed=False,_ready=lambda:None,bounds=(.001,.001),rewrite_admitted=budget)
    owner.region=NS(backend='rocm',chip='gfx1151',logical=NS(pair=NS(digest='pair'),input_bound=1.0,absolute_budget=.001),original_digest='o',transformed_digest='t')
    owner.candidates=(NS(name='original'),NS(name='rewrite'))
    monkeypatch.setattr(candidate,'arbitrate',lambda *args,force:next(c for c in owner.candidates if c.name==force))
    root=Path(__file__).resolve().parents[2]
    names=('python/tessera/compiler/native_ann.py','python/tessera/compiler/native_ann_gpu.py',
        'python/tessera/compiler/native_gpu_storage.py','src/transforms/lib/NativeTapeToGPUPass.cpp')
    base=dict(backend='rocm',chip='gfx1151',pair='pair',original='o',transformed='t',input_bound=1.0,absolute_budget=.001,
        bounds=['0.001','0.001'],sources={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names},
        recorder_sha256=hashlib.sha256((root/'benchmarks/record_native_ann_execution.py').read_bytes()).hexdigest(),
        timing_domain='warm_package_h2d_dispatch_d2h_host_wall',numerical_verified=True,
        samples_ms=[[speedup]*31,[1.0]*31],medians_ms=[speedup,1.0],speedup=speedup)
    return owner,[dict(base,pid=i+1) for i in range(9)]


@pytest.mark.parametrize('speedup,budget,expected',[(1.1,True,'rewrite'),(1.0,True,'original'),(1.1,False,'original')])
def test_scoped_selection_keeps_performance_and_numerical_gates(monkeypatch,speedup,budget,expected):
    owner,reports=owner_and_reports(monkeypatch,speedup,budget)
    selected,evidence=owner.select_from_measurements(reports)
    assert selected.name==expected and evidence['production_promoted'] is False


@pytest.mark.parametrize('field,value',[('chip','gfx1201'),('original','foreign'),('input_bound',True),('absolute_budget',.1),('sources',{}),('recorder_sha256','changed')])
def test_foreign_evidence_refuses_before_arbiter(monkeypatch,field,value):
    from tessera.compiler.emit import candidate
    owner,reports=owner_and_reports(monkeypatch)
    monkeypatch.setattr(candidate,'arbitrate',lambda *a,**kw:pytest.fail('foreign measurement reached arbiter'))
    for report in reports:report[field]=value
    with pytest.raises(ValueError):owner.select_from_measurements(reports)


def test_later_run_cannot_replace_numeric_identity_with_boolean(monkeypatch):
    owner,reports=owner_and_reports(monkeypatch)
    reports[-1]['input_bound']=True
    with pytest.raises(ValueError):owner.select_from_measurements(reports)
