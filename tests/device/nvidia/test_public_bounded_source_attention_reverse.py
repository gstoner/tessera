"""Bounded public source AD uses one compiler-owned capacity program on SM120."""
from contextlib import ExitStack
import json
import os
import subprocess
import sys

import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tessera._jit_boundary import TesseraJitError
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.device.nvidia.test_source_attention_reverse import oracle
from tests.unit.test_native_attention_prefix_program import (
    prefix_attention, prefix_attention_permuted, prefix_attention_swapped_outputs)

pytestmark = pytest.mark.hardware_nvidia


def _values(storage, sq, sk, d, dv, seed):
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    rng = np.random.default_rng(seed)
    shapes = ((1, 2, sq, d), (1, 1, sk, d), (1, 1, sk, dv), (1, 2, sq, dv), (1, 2, sq))
    return tuple(rng.normal(0, .25, shape).astype(np.float32 if i == 4 else dtype)
                 for i, shape in enumerate(shapes))


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("variant", ["normal", "permuted", "swapped"])
@pytest.mark.parametrize("wide", [False, True])
def test_public_bounded_source_capacity_replay(storage, variant, wide, monkeypatch):
    assert rt._nvidia_device_name() == "sm_120"
    cap = (17, 19) if wide else (7, 11)
    d, dv = (257, 13) if wide else (4, 3)
    function = (prefix_attention_permuted if variant == "permuted" else
                prefix_attention_swapped_outputs if variant == "swapped" else prefix_attention)
    requested = (2, 0) if variant == "permuted" else (2, 0, 1) if variant == "swapped" else (0, 1, 2)
    names = ("v", "q", "k") if variant == "permuted" else ("q", "k", "v")
    fn = ts.jit(target="nvidia_sm120", autodiff="reverse", wrt=tuple(names[i] for i in requested))(function)
    owner = handle = images = None
    retained = []
    for step, (sq, sk) in enumerate(((1, 1), cap, (3, 2), (1, cap[1]), (cap[0], 1))):
        values = _values(storage, sq, sk, d, dv, 304 + step)
        q, k, v, do, dlse = values
        roots = (v, q, k) if variant == "permuted" else (q, k, v)
        seeds = (dlse, do) if variant == "swapped" else (do, dlse)
        expected = oracle(*values)
        expected = (expected[2], expected[0], expected[1]) if variant == "permuted" else expected
        def forbidden(*args, **kwargs):
            pytest.fail("bounded warm frame invoked eager source, tracing, compilation or preparation")
        with monkeypatch.context() as guard:
            if step:
                guard.setattr(subprocess, "run", forbidden)
                guard.setattr(subprocess, "Popen", forbidden)
                guard.setattr(fn, "_fn", forbidden)
                guard.setattr(fn, "_specialized_autodiff_module", forbidden)
                guard.setattr(owner, "_prepare", forbidden)
            result = fn.native_backward(*roots, out_cotangents=seeds, sequence_bounds=cap)
        assert len(fn._native_prepared_vjp_calls) == 1
        current = next(iter(fn._native_prepared_vjp_calls.values()))
        execution = fn.last_backward_execution
        if not step:
            owner, handle, images = current, current._handle, execution["image_digests"]
        assert current is owner and current._handle == handle and execution["image_digests"] == images
        assert execution["runtime_dims"] == (1, 2, 1, sq, sk, d, dv)
        assert execution["shape_bounds"] == (1, 2, 1, *cap, d, dv)
        assert execution["execution_certificate"]["source_reexecution"] == "prohibited"
        assert execution["execution_certificate"]["evidence_scope"] == "exact_device"
        for got, index in zip(result, requested, strict=True):
            np.testing.assert_allclose(got.astype(np.float32), expected[index].astype(np.float32), rtol=.03, atol=.002)
        for old, saved in retained:
            np.testing.assert_array_equal(old, saved)
            assert all(not np.shares_memory(old, x) for x in result)
        retained.extend((x, x.copy()) for x in result)
        if step == 1:
            with ExitStack() as stack:
                sessions = tuple(stack.enter_context(NvidiaDeviceSession()) for _ in range(5))
                resident = tuple(s.upload(x) for s, x in zip(sessions, (*roots, *seeds), strict=True))
                with monkeypatch.context() as guard:
                    guard.setattr(subprocess, "run", forbidden)
                    guard.setattr(subprocess, "Popen", forbidden)
                    guard.setattr(fn, "_fn", forbidden)
                    guard.setattr(fn, "_specialized_autodiff_module", forbidden)
                    actual = fn.native_backward(*resident[:3], out_cotangents=resident[3:], sequence_bounds=cap)
                for got, prior in zip(actual, result, strict=True):
                    np.testing.assert_array_equal(got, prior)
        artifact = rt.RuntimeArtifact.from_json(fn.native_backward_runtime_artifact().to_json())
        assert artifact.metadata["native_program"]["schema"].endswith(".v4" if wide else ".v3")
        with monkeypatch.context() as guard:
            guard.setattr(subprocess, "run", forbidden)
            guard.setattr(subprocess, "Popen", forbidden)
            replay = rt.launch(artifact, (*roots, *seeds))
        assert replay["ok"], replay
        for got, prior in zip(replay["output"], result, strict=True):
            np.testing.assert_array_equal(got, prior)
    # A changed frame exceeding the same capacity must never trigger recompilation.
    values = _values(storage, cap[0] + 1, 2, d, dv, 999)
    q, k, v, do, dlse = values
    roots = (v, q, k) if variant == "permuted" else (q, k, v)
    seeds = (dlse, do) if variant == "swapped" else (do, dlse)
    with monkeypatch.context() as guard:
        guard.setattr(subprocess, "run", forbidden)
        guard.setattr(subprocess, "Popen", forbidden)
        guard.setattr(fn, "_specialized_autodiff_module", forbidden)
        with pytest.raises(ValueError):
            fn.native_backward(*roots, out_cotangents=seeds, sequence_bounds=cap)
    assert fn.last_backward_execution is None and len(fn._native_prepared_vjp_calls) == 1
    with pytest.raises(TesseraJitError, match="no persisted"):
        fn.native_backward_runtime_artifact()


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("wide", [False, True])
def test_bounded_public_export_fresh_process(storage, wide, tmp_path):
    assert rt._nvidia_device_name() == "sm_120"
    cap = (17, 19) if wide else (7, 11)
    d, dv = (257, 13) if wide else (4, 3)
    fn = ts.jit(target="nvidia_sm120", autodiff="reverse")(prefix_attention)
    first = _values(storage, 1, 1, d, dv, 77)
    fn.native_backward(*first[:3], out_cotangents=first[3:], sequence_bounds=cap)
    program = tmp_path / "program.json"
    program.write_text(fn.native_backward_runtime_artifact().to_json())
    frames = ((1, 1), cap, (3, 2))
    arrays, expected = {}, []
    for frame, (sq, sk) in enumerate(frames):
        values = _values(storage, sq, sk, d, dv, 99 + frame)
        expected.append(oracle(*values))
        arrays.update({f"{frame}_{i}": x.astype(np.float32) for i, x in enumerate(values)})
    inputs, outputs = tmp_path / "inputs.npz", tmp_path / "outputs.npz"
    np.savez(inputs, **arrays)
    script = '''
import json, os, subprocess, sys
def forbidden(*a, **kw):
    raise AssertionError("fresh portable bounded replay invoked external tools")
subprocess.run = subprocess.Popen = subprocess.check_output = forbidden
os.environ["TESSERA_OPT"] = os.environ["TESSERA_NVIDIA_OPT"] = "/unavailable/compiler"
import numpy as np, ml_dtypes
from tessera import runtime as rt
from tessera.compiler.prepared_source_attention_reverse import PreparedSourceAttentionReverse
artifact = rt.RuntimeArtifact.from_json(open(sys.argv[1]).read())
loaded = np.load(sys.argv[2])
types = {"f16": np.float16, "bf16": ml_dtypes.bfloat16, "f32": np.float32}
manifest = artifact.metadata["native_program"]
buffers = manifest["plan"]["buffers"]
outputs = {}
with PreparedSourceAttentionReverse.from_manifest(manifest) as owner:
    retained = owner.runtime_artifact()
    handle = None
    for frame in range(3):
        values = tuple(loaded[f"{frame}_{i}"].astype(types[buffers[i]["storage"]]) for i in range(5))
        result = rt.launch(retained, values)
        assert result["ok"], result
        assert result["physical_attestation"]["device_arch"] == "sm_120"
        assert len(result["component_receipts"]) == int(sys.argv[4])
        if handle is None: handle = owner._handle
        assert owner._handle == handle
        outputs.update({f"{frame}_{i}": x.astype(np.float32) for i, x in enumerate(result["output"])})
np.savez(sys.argv[3], **outputs)
print(json.dumps({"frames": 3, "ok": True}))
'''
    child = subprocess.run([sys.executable, "-c", script, str(program), str(inputs), str(outputs),
                            str(5 if wide else 4)], text=True, capture_output=True, timeout=90, env=dict(os.environ))
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(child.stdout.strip().splitlines()[-1])["frames"] == 3
    restored = np.load(outputs)
    for frame, references in enumerate(expected):
        for i, ref in enumerate(references):
            np.testing.assert_allclose(restored[f"{frame}_{i}"], ref.astype(np.float32), rtol=.03, atol=.002)
