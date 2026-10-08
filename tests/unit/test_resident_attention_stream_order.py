"""Producer ordering and event ownership for saved-LSE attention."""
import ctypes as ct
import pytest
from tessera.compiler.resident_attention import ResidentAttentionTape

def frame():
    value=ResidentAttentionTape.__new__(ResidentAttentionTape)
    value.asynchronous=False
    value._pending_sources=[]
    value._consumer_streams=set()
    value.context=ct.c_void_p(100)
    value._stream=ct.c_void_p(200)
    calls=[]
    def context(stream,dest):
        calls.append(("context",stream.value));dest._obj.value=100;return 0
    def create(dest,flags):
        calls.append(("create",flags));dest._obj.value=300;return 0
    value._stream_context=context
    value._event_create=create
    value._event_record=lambda event,stream:calls.append(("record",event.value,stream.value)) or 0
    value._stream_wait=lambda stream,event,flags:calls.append(("wait",stream.value,event.value,flags)) or 0
    value._event_destroy=lambda event:calls.append(("destroy",event.value)) or 0
    return value,calls

@pytest.mark.parametrize("producer",[1,2,400])
def test_wait_orders_declared_producer_and_destroys_event(producer):
    owner,calls=frame()
    owner._order_producer_stream(producer)
    assert calls==[("context",producer),("create",2),("record",300,producer),
                   ("wait",200,300,0),("destroy",300)]

@pytest.mark.parametrize("producer",[None,200])
def test_no_event_for_completed_or_same_stream(producer):
    owner,calls=frame()
    owner._order_producer_stream(producer)
    assert calls==([] if producer is None else [("context",200)])

@pytest.mark.parametrize("producer",[0,-1,True,"400",1<<64])
def test_invalid_stream_is_rejected_before_driver_work(producer):
    owner,calls=frame()
    with pytest.raises(ValueError,match="producer stream is invalid"):
        owner._order_producer_stream(producer)
    assert calls==[]

def test_other_context_is_rejected_before_creating_event():
    owner,calls=frame()
    def context(stream,dest):dest._obj.value=101;return 0
    owner._stream_context=context
    with pytest.raises(ValueError,match="another context"):
        owner._order_producer_stream(400)
    assert calls==[]

@pytest.mark.parametrize("failed",["record","wait"])
def test_driver_failure_destroys_event(failed):
    owner,calls=frame()
    if failed=="record":owner._event_record=lambda *args:1
    else:owner._stream_wait=lambda *args:1
    with pytest.raises(RuntimeError,match="CUDA status"):
        owner._order_producer_stream(400)
    assert calls[-1]==("destroy",300)


def test_group_waits_once_per_stream_but_repeats_for_next_call():
    owner,calls=frame()
    def resident(value,shape,*,producer_streams):
        producer_streams.append(value)
        return 512
    owner._resident=resident
    values=(400,400,None,500,400)
    assert owner._resident_group(values,[(1,)]*len(values))==[512]*len(values)
    assert [call for call in calls if call[0]=="record"]==[
        ("record",300,400),("record",300,500)]
    owner._resident_group((400,),((1,),))
    assert [call for call in calls if call[0]=="record"][-1]==("record",300,400)
    assert len([call for call in calls if call[0]=="record"])==3


def test_group_validates_all_buffers_before_ordering_any_stream():
    owner,calls=frame()
    def resident(value,shape,*,producer_streams):
        if value=="invalid":
            raise ValueError("allocation extent")
        producer_streams.append(value)
        return 512
    owner._resident=resident
    with pytest.raises(ValueError,match="allocation extent"):
        owner._resident_group((400,"invalid"),((1,),(1,)))
    assert calls==[]

def test_async_snapshot_orders_future_producer_writes_after_private_reads():
    owner,calls=frame()
    owner.asynchronous=True
    owner._last_producer_streams={400}
    source=object()
    owner._protect_snapshot_sources((source,))
    assert owner._pending_sources==[source]
    assert calls==[("create",2),("record",300,200),("wait",400,300,0),("destroy",300)]

def test_async_source_group_requires_declared_stream_before_dependencies():
    owner,calls=frame();owner.asynchronous=True
    def resident(value,shape,*,producer_streams):
        producer_streams.append(None);return 512
    owner._resident=resident
    with pytest.raises(ValueError,match="declared producer"):
        owner._resident_group((None,),((1,),))
    assert calls==[]

def test_wait_on_tracks_context_checked_consumer_until_close():
    import threading
    owner,calls=frame()
    owner._lock=threading.RLock();owner._ready=lambda:None
    owner.wait_on(400)
    assert owner._consumer_streams=={400}
    assert calls==[("context",400),("create",2),("record",300,200),
                   ("wait",400,300,0),("destroy",300)]
    owner.closed=False;owner._jvp_binding=None;owner._modules=[]
    owner._stream_sync=lambda stream:calls.append(("sync",stream.value)) or 0
    owner._release=lambda:calls.append(("release",))
    owner._stream_destroy=lambda stream:calls.append(("stream_destroy",stream.value)) or 0
    owner.close()
    assert calls[-3:]==[("sync",400),("release",),("stream_destroy",200)]
    assert owner.closed and not owner._consumer_streams
