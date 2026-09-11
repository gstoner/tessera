import pytest
from test_native_reader_retirement import setup
from tessera.compiler.native_stream_epoch import NativeStreamEpoch


def test_epoch_orders_readers_and_writer_without_host_wait():
    owner, native = setup()
    epoch = NativeStreamEpoch(owner.frame, native, owner._reader_buffers, owner._submission)
    with epoch.read(21):
        with pytest.raises(ValueError, match="scopes"):
            with epoch.write(22):
                pass
    with epoch.write(22):
        with pytest.raises(ValueError, match="closed"):
            with epoch.read(23):
                pass
    assert ("stream_wait", 22) in native.calls
    assert not any(call[0] in ("stream_sync", "event_wait") for call in native.calls)
    epoch.wait()


def test_epoch_record_failure_keeps_writer_retryable_after_completion():
    owner, native = setup()
    epoch = NativeStreamEpoch(owner.frame, native, owner._reader_buffers, owner._submission)
    native.record_failure = True
    with pytest.raises(RuntimeError, match="injected"):
        with epoch.write(21):
            pass
    with pytest.raises(RuntimeError, match="unproven"):
        with epoch.write(22):
            pass
    assert not epoch.retiring
    native.record_failure = False
    epoch.wait()
    with epoch.write(22):
        pass
    epoch.wait()
