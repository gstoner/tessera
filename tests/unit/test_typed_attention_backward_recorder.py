"""Reproducible half-word input encoding and prelaunch packet integrity."""
import json

import ml_dtypes
import numpy as np
import pytest

from benchmarks.nvidia import record_typed_attention_backward as recorder


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_recorder_roundtrip_preserves_every_source_word_and_matched_control_value(tmp_path, storage):
    words = np.arange(65536, dtype=np.uint16).reshape(1, 1, 256, 256)
    path = tmp_path/"input.npz"
    np.savez_compressed(path, **{name: words for name in ("q", "k", "v", "do")},
                        bias=np.zeros((1,), np.float32), seed=np.zeros((1,), np.float32))
    native = recorder.decode(path, storage, False)
    dtype = ml_dtypes.bfloat16 if storage == "bf16" else np.float16
    for name in ("q", "k", "v", "do"):
        np.testing.assert_array_equal(native[name].view(np.uint16), words)
    with np.errstate(invalid="ignore"):
        control = recorder.decode(path, storage, True)
        expected = words.view(dtype).astype(np.float32)
    for name in ("q", "k", "v", "do"):
        np.testing.assert_array_equal(control[name], expected)


@pytest.mark.parametrize("storage,control", [("fp8", False), ("f16", "yes")])
def test_invalid_codec_arguments_precede_file_access(tmp_path, storage, control):
    with pytest.raises(ValueError, match="explicit"):
        recorder.decode(tmp_path/"missing.npz", storage, control)


@pytest.mark.parametrize("bad_dtype", [np.uint32, np.int16, np.dtype(">u2")])
def test_wrong_physical_word_encoding_is_not_silently_reinterpreted(tmp_path, bad_dtype):
    path = tmp_path/"input.npz"
    np.savez_compressed(path, **{name: np.zeros((1,), bad_dtype) for name in ("q", "k", "v", "do")},
                        bias=np.zeros((1,), np.float32), seed=np.zeros((1,), np.float32))
    with pytest.raises(ValueError, match="raw u16"):
        recorder.decode(path, "f16", False)


def test_input_hash_mismatch_stops_before_any_native_launch(tmp_path, monkeypatch):
    (tmp_path/"input.npz").write_bytes(b"changed input bytes")
    index = tmp_path/"index.json"
    index.write_text(json.dumps({"rows": [{"input_file": "input.npz", "input_sha256": "0"*64}]}))
    monkeypatch.setattr(recorder, "device", lambda: "host-only integrity test; no device evidence")
    def forbidden(*args, **kwargs):
        pytest.fail("launched a packet with mismatched input bytes")
    monkeypatch.setattr(recorder.rt, "launch", forbidden)
    with pytest.raises(AssertionError):
        recorder.replay(index)
