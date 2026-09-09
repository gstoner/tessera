"""Pending owned-state copyback excludes readers and retains failed work."""
import threading
from types import SimpleNamespace
import pytest
from tessera.compiler.native_source_state import OwnedSourceGPUState


def owner_with_pending(*,copy_status=0,query_status=0):
    owner=OwnedSourceGPUState.__new__(OwnedSourceGPUState)
    owner._lock=threading.RLock()
    owner._closed=owner._poisoned=False
    owner._readers=0
    owner._quarantine=[]
    calls=[]
    view=SimpleNamespace(__cuda_array_interface__=dict(shape=(4,),typestr='<f4',data=(128,True)))
    def check(code):
        if code:raise RuntimeError(f'driver status {code}')
    owner._frame=SimpleNamespace(_ready=lambda:None,results=[view],check=check,
        sync=lambda:calls.append('sync') or 0,close=lambda:calls.append('owner-close'))
    result=SimpleNamespace(poll=lambda:True,results=[view],close=lambda:calls.append('result-close'))
    owner._pending=[result,7,None]
    def function(cu,hip,args):
        def run(*values):
            calls.append(cu)
            if cu=='cuEventCreate':values[0]._obj.value=16
            if cu=='cuMemcpyDtoDAsync_v2':return copy_status
            if cu=='cuEventQuery':return query_status
            return 0
        return run
    owner._event_function=function
    return owner,result,calls


def test_pending_copyback_blocks_readers_and_retains_result():
    owner,result,calls=owner_with_pending(query_status=600)
    with pytest.raises(ValueError,match='pending'):
        with owner.read():pass
    assert owner.poll_step() is None
    assert owner._pending[0] is result
    assert 'sync' not in calls
    assert 'result-close' not in calls
    owner.close()
    assert calls.index('sync')<calls.index('result-close')


def test_failed_copyback_poison_retains_until_synchronized_close():
    owner,result,calls=owner_with_pending(copy_status=700)
    with pytest.raises(RuntimeError,match='700'):owner.poll_step()
    assert owner._poisoned and owner._pending[0] is result
    assert 'result-close' not in calls
    with pytest.raises(ValueError,match='poisoned'):owner.poll_step()
    owner.close()
    assert calls.index('sync')<calls.index('result-close')


def test_completed_copyback_transfers_result_ownership():
    owner,result,calls=owner_with_pending()
    assert owner.poll_step() is result
    assert owner._pending is None
    owner.close()
    assert 'result-close' not in calls
