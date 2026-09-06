"""Host lock/closure contracts only; device snapshot proof uses owning GPUs."""
import threading
from types import SimpleNamespace
import pytest
from tessera.compiler.native_storage_pair import NativeStoragePair


def pair():
    obj=NativeStoragePair.__new__(NativeStoragePair)
    obj._frame_lock=threading.RLock()
    obj._closed=False
    obj._frames=[]
    obj.binding=SimpleNamespace(close=lambda:None)
    return obj


def test_closed_pair_cannot_capture_or_reopen():
    obj=pair()
    obj.close()
    obj.close()
    with pytest.raises(ValueError,match='closed'):
        obj.capture(None)
    with pytest.raises(ValueError,match='closed'):
        obj(None)


def test_close_releases_owner_lock_before_waiting_for_frame():
    obj=pair()
    frame_lock=threading.RLock()
    entered=threading.Event()
    closing=threading.Event()
    def close_frame():
        closing.set()
        with frame_lock:
            pass
    obj._frames=[SimpleNamespace(close=close_frame)]
    def backward():
        with frame_lock:
            entered.set()
            assert closing.wait(2)
            with pytest.raises(ValueError,match='closed'):
                obj(None)
    errors=[]
    def checked(fn):
        try:
            fn()
        except BaseException as error:
            errors.append(error)
    first=threading.Thread(target=checked,args=(backward,),daemon=True)
    second=threading.Thread(target=checked,args=(obj.close,),daemon=True)
    first.start()
    assert entered.wait(2)
    second.start()
    first.join(3)
    second.join(3)
    assert not first.is_alive() and not second.is_alive(), 'owner/frame lock inversion'
    assert not errors


def test_partial_backward_allocation_failure_preserves_persistent_frame():
    import ctypes as ct
    from tessera.compiler.native_device_tape import NativeDeviceTape
    obj=NativeDeviceTape.__new__(NativeDeviceTape)
    obj._lock=threading.RLock()
    obj._ready=lambda:None
    obj.primal=obj._input=SimpleNamespace(shape=(4,))
    old=object()
    obj.buffers=[old]
    obj.check=lambda code:None
    obj.sync=lambda:0
    attempts=[]
    freed=[]
    def alloc(pointer,size):
        attempts.append(size)
        if len(attempts)==2:
            raise RuntimeError('out of memory')
        ct.cast(pointer,ct.POINTER(ct.c_void_p))[0]=ct.c_void_p(123)
        return 0
    obj.alloc=alloc
    obj.free=lambda pointer:freed.append(pointer.value) or 0
    with pytest.raises(RuntimeError,match='out of memory'):
        obj.backward(None)
    assert obj.buffers==[old]
    assert freed==[123]


@pytest.mark.parametrize('shape',[(True,),(-1,),(1<<62,),(1<<32,1<<32)])
def test_persistent_allocation_refuses_invalid_byte_extents(shape):
    from tessera.compiler.native_device_tape import _Buffer
    with pytest.raises(ValueError):
        _Buffer(SimpleNamespace(),shape)
