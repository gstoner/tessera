"""Public original-source attention VJP and compiler-free runtime replay on SM120."""
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


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("variant", ["normal", "permuted", "swapped", "wide"])
def test_public_source_attention_reverse_and_portable_replay(storage, variant, monkeypatch):
    assert rt._nvidia_device_name() == "sm_120", "owning SM120 required"
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    sq, sk, d, dv = (17, 19, 257, 13) if variant == "wide" else (3, 5, 4, 3)
    rng = np.random.default_rng(20261019)
    shapes = ((1, 2, sq, d), (1, 1, sk, d), (1, 1, sk, dv), (1, 2, sq, dv), (1, 2, sq))
    values = tuple(rng.normal(0, .3, shape).astype(np.float32 if i == 4 else dtype)
                   for i, shape in enumerate(shapes))
    q, k, v, do, dlse = values
    function = (prefix_attention_permuted if variant == "permuted" else
                prefix_attention_swapped_outputs if variant == "swapped" else prefix_attention)
    roots = (v, q, k) if variant == "permuted" else (q, k, v)
    seeds = (dlse, do) if variant == "swapped" else (do, dlse)
    requested = (2, 0) if variant == "permuted" else (2, 0, 1) if variant == "swapped" else (0, 1, 2)
    names = ("v", "q", "k") if variant == "permuted" else ("q", "k", "v")
    wrt = tuple(names[i] for i in requested)
    expected = oracle(*values)
    original = (expected[2], expected[0], expected[1]) if variant == "permuted" else expected
    expected = tuple(original[i] for i in requested)
    fn = ts.jit(target="nvidia_sm120", autodiff="reverse", wrt=wrt)(function)
    first = fn.native_backward(*roots, out_cotangents=seeds)
    for actual, ref in zip(first, expected, strict=True):
        np.testing.assert_allclose(actual.astype(np.float32), ref.astype(np.float32), rtol=.03, atol=.002)
    execution = fn.last_backward_execution
    assert execution["compiler_path"] == "nvidia_sm120_source_attention_reverse_compiled"
    assert execution["family"] == "source_normalization_attention"
    assert execution["execution_certificate"]["evidence_scope"] == "exact_device"
    assert execution["physical_attestation"]["device_arch"] == "sm_120"
    artifact = rt.RuntimeArtifact.from_json(fn.native_backward_runtime_artifact().to_json())
    def forbidden(*args, **kwargs):
        pytest.fail("warm or portable source attention reverse invoked compiler/eager source")
    with monkeypatch.context() as guard:
        guard.setattr(subprocess, "run", forbidden)
        guard.setattr(subprocess, "Popen", forbidden)
        guard.setattr(fn, "_fn", forbidden)
        warm = fn.native_backward(*roots, out_cotangents=seeds)
        replay = rt.launch(artifact, (*roots, *seeds))
    assert replay["ok"], replay
    for collection in (warm, replay["output"]):
        for actual, prior in zip(collection, first, strict=True):
            np.testing.assert_array_equal(actual, prior)
            assert not np.shares_memory(actual, prior)
    with ExitStack() as stack:
        sessions = tuple(stack.enter_context(NvidiaDeviceSession()) for _ in range(5))
        resident = tuple(s.upload(x) for s, x in zip(sessions, (*roots, *seeds), strict=True))
        actual = fn.native_backward(*resident[:3], out_cotangents=resident[3:])
        assert fn.last_backward_execution["execution_certificate"]["proof_mode"] == "abstract_metadata_structural"
        for got, ref in zip(actual, first, strict=True):
            np.testing.assert_array_equal(got, ref)
    changed = tuple((x.astype(np.float32) * .7).astype(x.dtype) for x in values)
    qc, kc, vc, doc, lc = changed
    roots_changed = (vc, qc, kc) if variant == "permuted" else (qc, kc, vc)
    seeds_changed = (lc, doc) if variant == "swapped" else (doc, lc)
    expected_changed = oracle(*changed)
    if variant == "permuted":
        expected_changed = (expected_changed[2], expected_changed[0], expected_changed[1])
    outputs = fn.native_backward(*roots_changed, out_cotangents=seeds_changed)
    for got, index in zip(outputs, requested, strict=True):
        np.testing.assert_allclose(got.astype(np.float32), expected_changed[index].astype(np.float32), rtol=.03, atol=.002)


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_public_source_reverse_revisits_static_shape_and_clears_failed_export(storage):
    assert rt._nvidia_device_name() == "sm_120"
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    fn = ts.jit(target="nvidia_sm120", autodiff="reverse")(prefix_attention)
    rng = np.random.default_rng(103)
    images = []
    for sq in (3, 7, 3):
        shapes = ((1, 2, sq, 4), (1, 1, 5, 4), (1, 1, 5, 3), (1, 2, sq, 3), (1, 2, sq))
        values = tuple(rng.normal(0, .3, s).astype(np.float32 if i == 4 else dtype) for i, s in enumerate(shapes))
        actual = fn.native_backward(*values[:3], out_cotangents=values[3:])
        for got, ref in zip(actual, oracle(*values), strict=True):
            np.testing.assert_allclose(got.astype(np.float32), ref.astype(np.float32), rtol=.03, atol=.002)
        images.append(fn.last_backward_execution["image_digests"])
    assert images[0] == images[2] and len(fn._native_prepared_vjp_calls) == 2
    bad_artifact = fn.native_backward_runtime_artifact()
    bad_artifact.metadata["native_program"]["plan"]["outputs"].reverse()
    assert not rt.launch(bad_artifact, values)["ok"]
    assert fn.native_backward_runtime_artifact().metadata["native_program"]["plan"]["outputs"] == [11, 9, 10]
    with pytest.raises(ValueError, match="storage"):
        fn.native_backward(*values[:3], out_cotangents=(values[3], values[4].astype(dtype)))
    assert fn.last_backward_execution is None
    with pytest.raises(TesseraJitError, match="no persisted"):
        fn.native_backward_runtime_artifact()


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_source_reverse_fresh_process_without_compiler(storage, tmp_path):
    assert rt._nvidia_device_name() == "sm_120"
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    rng = np.random.default_rng(11)
    shapes = ((1, 2, 3, 4), (1, 1, 5, 4), (1, 1, 5, 3), (1, 2, 3, 3), (1, 2, 3))
    values = tuple(rng.normal(0, .3, s).astype(np.float32 if i == 4 else dtype) for i, s in enumerate(shapes))
    fn = ts.jit(target="nvidia_sm120", autodiff="reverse")(prefix_attention)
    expected = fn.native_backward(*values[:3], out_cotangents=values[3:])
    artifact = tmp_path / "program.json"
    artifact.write_text(fn.native_backward_runtime_artifact().to_json())
    roots, outputs = tmp_path / "roots.npz", tmp_path / "outputs.npz"
    np.savez(roots, **{str(i): value.astype(np.float32) for i, value in enumerate(values)})
    script = '''
import json, os, subprocess, sys
def forbidden(*args, **kwargs):
    raise AssertionError("portable source reverse invoked external tooling")
subprocess.run = subprocess.Popen = subprocess.check_output = forbidden
os.environ["TESSERA_OPT"] = os.environ["TESSERA_NVIDIA_OPT"] = "/unavailable/compiler"
import numpy as np, ml_dtypes
from tessera import runtime as rt
artifact = rt.RuntimeArtifact.from_json(open(sys.argv[1]).read())
loaded = np.load(sys.argv[2])
types = {"f16": np.float16, "bf16": ml_dtypes.bfloat16, "f32": np.float32}
buffers = artifact.metadata["native_program"]["plan"]["buffers"]
values = tuple(loaded[str(i)].astype(types[buffers[i]["storage"]]) for i in range(5))
result = rt.launch(artifact, values)
assert result["ok"], result
assert result["physical_attestation"]["device_arch"] == "sm_120"
assert len(result["component_receipts"]) == 4
np.savez(sys.argv[3], **{str(i): x.astype(np.float32) for i, x in enumerate(result["output"])})
print(json.dumps({"ok": True, "execution_kind": result["execution_kind"], "components": 4}))
'''
    child = subprocess.run([sys.executable, "-c", script, str(artifact), str(roots), str(outputs)],
                           text=True, capture_output=True, timeout=90, env=dict(os.environ))
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(child.stdout.strip().splitlines()[-1])["components"] == 4
    restored = np.load(outputs)
    for i, prior in enumerate(expected):
        np.testing.assert_array_equal(restored[str(i)], prior.astype(np.float32))
