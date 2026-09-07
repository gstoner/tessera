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


@pytest.mark.parametrize('fail_launch', [False, True])
def test_backward_releases_temporary_primal_and_preserves_results(fail_launch):
    import ctypes as ct
    from tessera.compiler.native_device_tape import NativeDeviceTape
    obj = NativeDeviceTape.__new__(NativeDeviceTape)
    obj._lock = threading.RLock()
    obj._ready = lambda: None
    obj.closed = False
    obj.width = 4
    obj.primal = obj._input = SimpleNamespace(shape=(4,), pointer=ct.c_void_p(128))
    obj.children = []
    obj.buffers = [obj.primal]
    obj.check = lambda code: None
    obj.sync = lambda: 0
    allocated, freed, temporaries, results = [], [], [], []

    def alloc(pointer, size):
        address = 256 * (len(allocated) + 1)
        allocated.append(address)
        ct.cast(pointer, ct.POINTER(ct.c_void_p))[0] = ct.c_void_p(address)
        return 0

    def pair(value, seed, primal, derivative, width):
        temporaries.append(primal)
        if fail_launch and seed == 7:
            raise RuntimeError('injected launch failure')

    obj.alloc = alloc
    obj.free = lambda pointer: freed.append(pointer.value) or 0
    obj.pair = pair
    for seed in range(16):
        if fail_launch and seed == 7:
            with pytest.raises(RuntimeError, match='injected'):
                obj.backward(seed)
        else:
            results.append(obj.backward(seed))
        assert obj.buffers == [obj.primal, *results]
        assert all(result.pointer.value not in freed for result in results)
        assert all(not temporary.pointer.value for temporary in temporaries)
    assert len(freed) == 16 + int(fail_launch)
    assert len(set(freed)) == len(freed)
    assert all(result.__cuda_array_interface__['data'][0] for result in results)
    obj.close()
    assert obj.closed and not obj.buffers
    assert all(not result.pointer.value for result in results)


@pytest.mark.parametrize('dtype,size,typestr', [('fp32', 16, '<f4'), ('fp64', 32, '<f8')])
def test_owned_tape_buffer_uses_declared_storage_width(dtype, size, typestr):
    import ctypes as ct
    from types import SimpleNamespace
    from tessera.compiler.native_device_tape import _Buffer
    allocations = []
    def allocate(pointer, nbytes):
        allocations.append(nbytes)
        ct.cast(pointer, ct.POINTER(ct.c_void_p))[0] = ct.c_void_p(4096)
        return 0
    frame = SimpleNamespace(closed=False, buffers=[], alloc=allocate, check=lambda status: None)
    buffer = _Buffer(frame, (4,), dtype)
    assert allocations == [size]
    assert buffer.nbytes == size
    assert buffer.__cuda_array_interface__['typestr'] == typestr
