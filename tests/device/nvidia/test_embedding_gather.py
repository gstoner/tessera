"""AR-DECODE-PROGRAM-1 S1c: embedding row gather on sm_120.

The kernel is compiled Graph -> Schedule -> Tile -> NVIDIA by the native
passes (``NativeEmbeddingGather.h``). A gather moves bits, so every check is
bitwise (compared as uint32 words, which also covers NaN payloads and -0.0)
against NumPy's ``take`` and against the S0 decode oracle's own embedding
table. A token outside ``[0, V)`` is refused through the status word with no
rows copied back.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._support.nvidia import nvidia_cuda_host_ready

PIPELINE = "tessera-nvidia-pipeline-sm120"
NAMES = ("embed", "token", "x")


@pytest.fixture(autouse=True)
def _sm120():
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    from tessera import runtime as rt
    assert rt._nvidia_device_name() == "sm_120"


_PACKAGES: dict = {}


def _launch(table, tokens, out=None):
    from tessera import runtime as rt
    from tessera.compiler import nvidia_native as native
    (vocab, hidden), batch = table.shape, tokens.shape[0]
    key = (vocab, hidden, batch)
    if key not in _PACKAGES:
        _PACKAGES[key] = native.package_embedding_gather(
            NAMES, vocab=vocab, hidden=hidden, batch=batch, pipeline_name=PIPELINE)
    package = _PACKAGES[key]
    out = np.full((batch, hidden), -7.0, np.float32) if out is None else out
    status = np.zeros(1, np.int32)
    artifact = rt.RuntimeArtifact(
        tile_ir=package.tile_ir, target_ir=package.target_ir, metadata={"target": "nvidia_sm120"},
        native_image=package.image, launch_descriptor=package.descriptor)
    bindings = {"embed": np.ascontiguousarray(table, np.float32),
                "token": np.ascontiguousarray(tokens, np.int32), "x": out, "status": status,
                "V": vocab, "H": hidden, "B": batch}
    result = rt.launch(rt.RuntimeArtifact.from_json(artifact.to_json()), bindings)
    return result, out, status


def _bits(x):
    return np.ascontiguousarray(x, np.float32).view(np.uint32)


def _gather(table, tokens):
    result, out, status = _launch(table, tokens)
    assert result.get("ok") is True and status[0] == 0, result
    np.testing.assert_array_equal(_bits(out), _bits(np.take(table, tokens, axis=0)))
    return out


def test_decode_oracle_embedding_rows():
    """The oracle's prompt and greedy tokens gather its own (f32-stored) table bit for bit."""
    from tessera.models import ar_decode as A
    weights = A.tiny_fixture_weights()
    trace = A.reference_decode(A.AR_DECODE_TINY, weights, A.tiny_fixture_contract(), A.TINY_PROMPT, 32)
    table = np.asarray(weights.embed, np.float32)
    tokens = np.asarray(list(A.TINY_PROMPT) + list(trace.tokens), np.int32)
    _gather(table, tokens)


@pytest.mark.parametrize("vocab,hidden,batch", [
    (1, 1, 1), (7, 5, 3), (512, 64, 3), (300, 127, 9), (300, 129, 9), (1000, 4096, 2), (32000, 33, 17),
])
def test_random_tables_match_numpy_take(vocab, hidden, batch):
    rng = np.random.default_rng(vocab * 31 + hidden * 7 + batch)
    table = rng.standard_normal((vocab, hidden)).astype(np.float32)
    tokens = rng.integers(0, vocab, size=batch).astype(np.int32)
    tokens[0] = vocab - 1  # the last row is in range
    _gather(table, tokens)


def test_repeated_tokens_and_special_values_move_as_bits():
    table = np.arange(16 * 40, dtype=np.float32).reshape(16, 40)
    table[3, :4] = [np.nan, -0.0, np.inf, -np.inf]
    table[3].view(np.uint32)[4] = 0x7FC00123  # a NaN payload a float compare would hide
    out = _gather(table, np.array([3, 0, 3, 15, 3], np.int32))
    assert out[0].view(np.uint32)[4] == 0x7FC00123
    assert np.signbit(out[2, 1]) and out[2, 1] == 0


@pytest.mark.parametrize("bad", [-1, 16, 2**31 - 1, -(2**31)])
def test_out_of_range_token_is_refused_and_nothing_is_copied(bad):
    table = np.ones((16, 40), np.float32)
    out = np.full((3, 40), -7.0, np.float32)
    result, out, status = _launch(table, np.array([2, bad, 5], np.int32), out)
    assert result.get("ok") is not True, result
    assert "refused" in str(result.get("reason", result)), result
    assert status[0] == 1
    assert (out == -7.0).all()


def test_mismatched_buffer_is_refused():
    from tessera import runtime as rt
    from tessera.compiler import nvidia_native as native
    package = native.package_embedding_gather(NAMES, vocab=8, hidden=4, batch=2, pipeline_name=PIPELINE)
    artifact = rt.RuntimeArtifact(
        tile_ir=package.tile_ir, target_ir=package.target_ir, metadata={"target": "nvidia_sm120"},
        native_image=package.image, launch_descriptor=package.descriptor)
    bindings = {"embed": np.zeros((7, 4), np.float32), "token": np.zeros(2, np.int32),
                "x": np.zeros((2, 4), np.float32), "status": np.zeros(1, np.int32),
                "V": 8, "H": 4, "B": 2}
    try:
        result = rt.launch(artifact, bindings)
    except (RuntimeError, ValueError):
        return
    assert result.get("ok") is not True, result
