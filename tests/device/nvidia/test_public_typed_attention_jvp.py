"""Public typed attention JVP compilation, portable replay and resident ownership."""
from pathlib import Path
import os

import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.native_attention_program import NativeAttentionJVPProgram
from tests.device.nvidia.test_public_typed_attention_vjp import function
from tests.device.nvidia.test_saved_lse_half_backward_kernel import compare
from benchmarks.nvidia.benchmark_jit_attention_vjp import download

pytestmark = [pytest.mark.hardware_nvidia, pytest.mark.compiler_route]

def reference(q,k,v,dq,dk,dv,causal=True,bias=None,dbias=None):
 q,k,v,dq,dk,dv=[x.astype(np.float64) for x in (q,k,v,dq,dk,dv)]
 groups=q.shape[1]//k.shape[1]
 k,v,dk,dv=[np.repeat(x,groups,axis=1) for x in (k,v,dk,dv)]
 scale=q.shape[-1]**-.5
 scores=scale*(q@k.swapaxes(-1,-2))
 ds=scale*(dq@k.swapaxes(-1,-2)+q@dk.swapaxes(-1,-2))
 if bias is not None:scores=scores+bias.astype(np.float64)
 if dbias is not None:ds=ds+dbias.astype(np.float64)
 if causal:
  sq,sk=q.shape[2],k.shape[2]
  mask=np.arange(sk)[None,:]<=np.arange(sq)[:,None]+max(sk-sq,0)
  scores=np.where(mask,scores,-np.inf)
 maximum=scores.max(axis=-1,keepdims=True);ex=np.exp(scores-maximum);den=ex.sum(axis=-1,keepdims=True);p=ex/den
 lse=(maximum+np.log(den))[...,0];dlse=np.sum(p*ds,axis=-1)
 return (p@v,lse),((p*(ds-dlse[...,None]))@v+p@dv,dlse)


@pytest.mark.parametrize("storage", ["float16", "bfloat16"])
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("active", [("q", "k", "v"), ("v",), ("q",)])
def test_public_typed_jvp_portable_resident_replay(storage, saved, bias, causal, active, monkeypatch):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    dtype = np.dtype(ml_dtypes.bfloat16 if storage == "bfloat16" else storage)
    rng = np.random.default_rng(20261017)
    values = {n: rng.normal(0, .7, shape).astype(dtype) for n, shape in {
        "q": (1, 2, 3, 16), "k": (1, 1, 5, 16), "v": (1, 1, 5, 16)}.items()}
    if bias:
        values["bias"] = rng.normal(0, .1, (1, 2, 1, 5)).astype(np.float32)
    directions = {n: rng.normal(0, .2, x.shape).astype(x.dtype) for n, x in values.items()}
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=active)(function(saved, bias, causal))
    program = fn.compile_native_attention_jvp(**values, compiler=Path(os.environ["TESSERA_OPT"]),
                                            llvm_bin=Path(os.environ["TESSERA_LLVM_BIN"]))
    program = NativeAttentionJVPProgram.from_json(program.to_json(), expected_digest=program.program_digest)
    def forbidden(*args, **kwargs):
        pytest.fail("portable resident replay attempted compilation")
    monkeypatch.setattr("subprocess.run", forbidden)
    with NvidiaDeviceSession() as session:
        roots = {n: session.upload(x) for n, x in values.items()}
        tangents = tuple(session.upload(directions[n]) for n in active)
        expected = reference(*(values[n] for n in ("q", "k", "v")),
            *(directions[n] if n in active else np.zeros_like(values[n]) for n in ("q", "k", "v")),
            causal=causal, bias=values.get("bias"))
        with program.capture(**roots) as frame:
            result = frame.jvp(*tangents)
            result = result if saved else (result,)
            wanted = expected[1] if saved else expected[1][:1]
            held = tuple(download(session, item) for item in result)
            for actual, target in zip(held, wanted, strict=True):
                compare(actual, target)
            # A second derivative call must not overwrite the retained first result.
            frame.jvp(*tangents)
            for prior, item in zip(held, result, strict=True):
                np.testing.assert_array_equal(download(session, item), prior)


@pytest.mark.parametrize("storage", ["float16", "bfloat16"])
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("subset", [False, True])
def test_public_typed_jvp_ordinary_prepared_host_route(storage, saved, bias, subset, monkeypatch):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    dtype = np.dtype(ml_dtypes.bfloat16 if storage == "bfloat16" else storage)
    rng = np.random.default_rng(20261017)
    values = {n: rng.normal(0, .7, shape).astype(dtype) for n, shape in {
        "q": (1, 2, 3, 16), "k": (1, 1, 5, 16), "v": (1, 1, 5, 16)}.items()}
    if bias:
        values["bias"] = rng.normal(0, .1, (1, 2, 1, 5)).astype(np.float32)
    active = (("bias", "q") if bias else ("v",)) if subset else tuple(values)
    directions = {n: rng.normal(0, .2, x.shape).astype(x.dtype) if n in active
                  else np.zeros_like(x) for n, x in values.items()}
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=active)(function(saved, bias, True))
    def forbidden(*args, **kwargs):
        pytest.fail("warm prepared JVP attempted compilation or concrete frontend evaluation")
    try:
        first = fn.native_jvp(**values, tangents=tuple(directions[n] for n in active))
        from tessera.compiler.native_attention_jvp_runtime import execute_unprepared
        package = next(iter(fn._native_jvp_packages.values()))
        metadata = package.contract["steps"][0]["child_metadata"]
        # Root order follows the authored function, not semantic Q/K/V order.
        frontend = ("bias", "v", "q", "k") if bias else ("q", "k", "v")
        unprepared = execute_unprepared(metadata,tuple(values[n] for n in frontend)+tuple(directions[n] for n in active))
        for got,wanted in zip(unprepared,first,strict=True):
            for a,b in zip(got if saved else (got,),wanted if saved else (wanted,),strict=True):
                np.testing.assert_array_equal(a,b)
        held = tuple(tuple(x.copy() for x in value) if saved else value.copy() for value in first)
        for generation in (1, 2):
            current = {n: x*generation for n, x in values.items()}
            expected = reference(*(current[n] for n in ("q", "k", "v")),
                *(directions[n] for n in ("q", "k", "v")),
                bias=current.get("bias"), dbias=directions.get("bias"))
            with monkeypatch.context() as guard:
                guard.setattr("subprocess.run", forbidden)
                guard.setattr(fn, "_fn", forbidden)
                guard.setattr(fn, "_trace_frontend_capture", forbidden)
                primal, tangent = fn.native_jvp(**current, tangents=tuple(directions[n] for n in active))
            for actuals, targets in zip((primal, tangent), expected, strict=True):
                actuals = actuals if saved else (actuals,)
                targets = targets if saved else targets[:1]
                for i, (actual, target) in enumerate(zip(actuals, targets, strict=True)):
                    assert actual.dtype == (np.dtype("float32") if i else dtype)
                    compare(actual, target)
            for actual, prior in zip(first, held, strict=True):
                for x, y in zip(actual if saved else (actual,), prior if saved else (prior,), strict=True):
                    np.testing.assert_array_equal(x, y)
        assert fn.last_jvp_execution["execution_kind"] == "native_gpu"
    finally:
        fn.close_native_storage()


@pytest.mark.parametrize("storage", ["float16", "bfloat16"])
@pytest.mark.parametrize("saved", [False, True])
def test_public_typed_jvp_orders_pending_producer_stream(storage, saved, monkeypatch):
    from contextlib import ExitStack
    from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed, queue_delayed_upload
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    dtype = np.dtype(ml_dtypes.bfloat16 if storage == "bfloat16" else storage)
    rng = np.random.default_rng(20261017)
    shapes = {"q": (1, 2, 3, 16), "k": (1, 1, 5, 16), "v": (1, 1, 5, 16)}
    values = {n: rng.normal(0, .4, shape).astype(dtype) for n, shape in shapes.items()}
    seeds = {n: rng.normal(0, .1, shape).astype(dtype) for n, shape in shapes.items()}
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=("q", "k", "v"))(
        function(saved, False, True))
    try:
        with ExitStack() as stack:
            sessions = {n: stack.enter_context(NvidiaDeviceSession()) for n in shapes}
            roots = {n: session.upload(np.zeros(shapes[n], dtype)) for n, session in sessions.items()}
            directions = tuple(Borrowed(sessions[n].upload(seeds[n])) for n in shapes)
            for session in sessions.values():
                assert session.synchronize() == 0
            borrowed = {n: Borrowed(value) for n, value in roots.items()}
            fn.native_jvp(**borrowed, tangents=directions)
            held = []
            callback = queue_delayed_upload(sessions["k"], roots["k"], values["k"], sessions["k"].stream, held)
            for n in ("q", "v"):
                import ctypes as ct
                assert sessions[n].lib.tessera_nvidia_device_upload(ct.c_void_p(roots[n].ptr),
                    ct.c_void_p(values[n].ctypes.data), values[n].nbytes, ct.c_void_p(sessions[n].stream)) == 0
            assert not held
            def forbidden(*args, **kwargs):
                pytest.fail("resident typed replay attempted compilation or host coercion")
            with monkeypatch.context() as guard:
                guard.setattr("subprocess.run", forbidden)
                guard.setattr(fn, "_fn", forbidden)
                guard.setattr(fn, "_trace_frontend_capture", forbidden)
                result = fn.native_jvp(**borrowed, tangents=directions)
            assert held and callback is not None
            expected = reference(*values.values(), *seeds.values())
            for actuals, targets in zip(result, expected, strict=True):
                for actual, target in zip(actuals if saved else (actuals,), targets if saved else targets[:1], strict=True):
                    compare(actual, target)
            assert fn.last_jvp_execution["host_preparation"] == "native_ordered_resident_snapshot"
    finally:
        fn.close_native_storage()


@pytest.mark.parametrize("storage", ["float16", "bfloat16"])
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_bounded_typed_jvp_reuses_one_portable_image_from_trace_at_one(storage, saved, bias, monkeypatch):
    from contextlib import ExitStack
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    dtype = np.dtype(ml_dtypes.bfloat16 if storage == "bfloat16" else storage)
    rng = np.random.default_rng(20261017)
    def values(sq, sk):
        roots = {n: rng.normal(0, .4, shape).astype(dtype) for n, shape in {
            "q": (1, 2, sq, 16), "k": (1, 1, sk, 16), "v": (1, 1, sk, 16)}.items()}
        if bias:
            roots["bias"] = rng.normal(0, .1, (1, 2, 1, 1)).astype(np.float32)
        seeds = {n: rng.normal(0, .2, x.shape).astype(x.dtype) for n, x in roots.items()}
        return roots, seeds
    traced, _ = values(1, 1)
    active = tuple(traced)
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=active)(function(saved, bias, True))
    program = fn.compile_native_attention_jvp(**traced, compiler=Path(os.environ["TESSERA_OPT"]),
        llvm_bin=Path(os.environ["TESSERA_LLVM_BIN"]), sequence_bounds=(7, 9))
    program = NativeAttentionJVPProgram.from_json(program.to_json(), expected_digest=program.program_digest)
    identity = (program.program_digest, program.tangent.binding_digest,
                program.pair.forward.image.image_digest)
    def forbidden(*args, **kwargs):
        pytest.fail("bounded typed replay attempted compilation")
    monkeypatch.setattr("subprocess.run", forbidden)
    with ExitStack() as stack:
        session = stack.enter_context(NvidiaDeviceSession())
        retained = []
        for sq, sk in ((1, 1), (3, 4), (7, 9), (7, 1), (1, 9), (3, 4)):
            current, seeds = values(sq, sk)
            expected = reference(*(current[n] for n in ("q", "k", "v")),
                *(seeds[n] for n in ("q", "k", "v")),
                bias=current.get("bias"), dbias=seeds.get("bias"))
            resident = {n: session.upload(x) for n, x in current.items()}
            directions = tuple(session.upload(seeds[n]) for n in active)
            frame = stack.enter_context(program.capture(**resident))
            tangent = frame.jvp(*directions)
            for actuals, targets in zip((frame.primal, tangent), expected, strict=True):
                for actual, target in zip(actuals if saved else (actuals,), targets if saved else targets[:1], strict=True):
                    compare(download(session, actual), target)
            actuals = tangent if saved else (tangent,)
            retained.append((actuals, tuple(download(session, x) for x in actuals)))
            for results, snapshot in retained:
                for actual, prior in zip(results, snapshot, strict=True):
                    np.testing.assert_array_equal(download(session, actual), prior)
            assert identity == (program.program_digest, program.tangent.binding_digest,
                                program.pair.forward.image.image_digest)
        oversized, _ = values(8, 9)
        with pytest.raises(ValueError, match="envelope"):
            program.capture(**{n: session.upload(x) for n, x in oversized.items()})


@pytest.mark.parametrize("storage", ["float16", "bfloat16"])
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
def test_public_typed_jvp_fresh_process_has_no_compiler_dependency(storage, saved, bounded, tmp_path):
    import subprocess
    import sys
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    dtype = np.dtype(ml_dtypes.bfloat16 if storage == "bfloat16" else storage)
    rng = np.random.default_rng(20261017)
    def inputs(sq, sk):
        return {n: rng.normal(0, .4, shape).astype(dtype) for n, shape in {
            "q": (1, 2, sq, 16), "k": (1, 1, sk, 16), "v": (1, 1, sk, 16)}.items()}
    values = inputs(3, 5)
    traced = inputs(1, 1) if bounded else values
    seeds = {n: rng.normal(0, .1, x.shape).astype(dtype) for n, x in values.items()}
    expected = reference(*values.values(), *seeds.values())
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=("q", "k", "v"))(
        function(saved, False, True))
    program = fn.compile_native_attention_jvp(**traced, compiler=Path(os.environ["TESSERA_OPT"]),
        llvm_bin=Path(os.environ["TESSERA_LLVM_BIN"]), sequence_bounds=(7, 9) if bounded else None)
    path = tmp_path/"program.json"
    path.write_text(program.to_json())
    payload = {n: x.view(np.uint16) for n, x in values.items()}
    payload.update({"d"+n: x.view(np.uint16) for n, x in seeds.items()})
    for prefix, group in zip(("primal", "tangent"), expected, strict=True):
        payload.update({prefix+str(i): x for i, x in enumerate(group if saved else group[:1])})
    data = tmp_path/"data.npz"
    np.savez(data, **payload)
    child = '\nimport os\nfrom pathlib import Path\nimport subprocess\nimport sys\nimport ml_dtypes\nimport numpy as np\nfrom tessera.compiler.native_attention_program import NativeAttentionJVPProgram\nfrom tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession\nfrom benchmarks.nvidia.benchmark_jit_attention_vjp import download\nfrom tests.device.nvidia.test_saved_lse_half_backward_kernel import compare\ndef forbidden(*args,**kwargs):\n    raise AssertionError("fresh portable replay attempted a compiler subprocess")\nsubprocess.run=forbidden\nsubprocess.check_output=forbidden\nfor key in ("TESSERA_OPT","TESSERA_NVIDIA_OPT","TESSERA_LLVM_BIN","TESSERA_NATIVE_STORAGE_LLVM_BIN"):\n    os.environ[key]="/missing-compiler-for-portable-replay"\nprogram=NativeAttentionJVPProgram.from_json(Path(sys.argv[1]).read_text(),expected_digest=sys.argv[2])\ndtype=np.dtype(ml_dtypes.bfloat16 if sys.argv[4]=="bfloat16" else sys.argv[4])\ndata=np.load(sys.argv[3])\nvalues={n:data[n].view(dtype) for n in ("q","k","v")}\ndirections=tuple(data["d"+n].view(dtype) for n in ("q","k","v"))\nwith NvidiaDeviceSession() as session:\n    roots={n:session.upload(x) for n,x in values.items()}\n    seeds=tuple(session.upload(x) for x in directions)\n    with program.capture(**roots) as frame:\n        result=frame.jvp(*seeds)\n        for prefix,group in (("primal",frame.primal),("tangent",result)):\n            for i,actual in enumerate(group if program.saved_lse else (group,)):\n                compare(download(session,actual),data[prefix+str(i)])\nprint("PASS compiler-free fresh typed JVP "+program.program_digest)\n'
    result = subprocess.run([sys.executable, "-c", child, str(path), program.program_digest,
                             str(data), storage], text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout+result.stderr
    assert "PASS compiler-free fresh typed JVP "+program.program_digest in result.stdout


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_textual_typed_jvp_with_fp32_output_reaches_public_owner(storage, saved, bias, monkeypatch):
    from tessera.compiler.native_attention_program import compile_attention_program
    from tests.unit.test_saved_lse_half_backward import source
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    dtype = np.dtype(np.float16 if storage == "f16" else ml_dtypes.bfloat16)
    rng = np.random.default_rng(20261017)
    values = {n: rng.normal(0, .4, shape).astype(dtype) for n, shape in {
        "q": (1, 2, 3, 4), "k": (1, 1, 5, 4), "v": (1, 1, 5, 3)}.items()}
    if bias:
        values["bias"] = rng.normal(0, .1, (1, 2, 1, 5)).astype(np.float32)
    directions = {n: rng.normal(0, .1, x.shape).astype(x.dtype) for n, x in values.items()}
    original = source(storage, saved=saved, bias=bias, output="f32").replace('"reverse"', '"forward"')
    program = compile_attention_program(original, tuple(range(len(values))),
        compiler=Path(os.environ["TESSERA_OPT"]), llvm_bin=Path(os.environ["TESSERA_LLVM_BIN"]),
        input_names=tuple(values))
    artifact = rt.RuntimeArtifact(graph_ir=original, target_ir=program.tangent.arena_ir,
        metadata=dict(target="nvidia_sm120", compiler_path="nvidia_sm120_attention_jvp_compiled",
            execution_kind="native_gpu", execution_mode="cuda_runtime", executable=True,
            arg_names=[f"primal_{i}" for i in range(len(values))]+[f"tangent_{i}" for i in range(len(values))],
            program_json=program.to_json(), program_digest=program.program_digest))
    artifact = rt.RuntimeArtifact.from_json(artifact.to_json())
    expected = reference(*(values[n] for n in ("q", "k", "v")),
        *(directions[n] for n in ("q", "k", "v")),
        bias=values.get("bias"), dbias=directions.get("bias"))
    def forbidden(*args, **kwargs):
        pytest.fail("textual typed public replay attempted compilation")
    monkeypatch.setattr("subprocess.run", forbidden)
    receipt = rt.launch(artifact, tuple(values.values())+tuple(directions.values()))
    assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
    for actuals, targets in zip(receipt["output"], expected, strict=True):
        for actual, target in zip(actuals if saved else (actuals,), targets if saved else targets[:1], strict=True):
            assert actual.dtype == np.dtype("float32")
            compare(actual, target)


@pytest.mark.parametrize("storage", ["float16", "bfloat16"])
@pytest.mark.parametrize("violation", ["capacity", "alignment", "stream", "storage_flag", "output_flag", "saved_flag"])
def test_typed_jvp_native_owner_refusal_preserves_valid_generation(storage, violation):
    import ctypes as ct
    from tessera.compiler.native_attention_jvp_runtime import prepared, clear_prepared
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    dtype = np.dtype(ml_dtypes.bfloat16 if storage == "bfloat16" else storage)
    rng = np.random.default_rng(20261017)
    values = tuple(rng.normal(0, .3, shape).astype(dtype) for shape in (
        (1, 2, 3, 16), (1, 1, 5, 16), (1, 1, 5, 16)))
    seeds = tuple(rng.normal(0, .1, x.shape).astype(dtype) for x in values)
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=("q", "k", "v"))(
        function(True, False, True))
    try:
        first = fn.native_jvp(*values, tangents=seeds)
        package = next(iter(fn._native_jvp_packages.values()))
        metadata = package.contract["steps"][0]["child_metadata"]
        owner = prepared(metadata)
        handle = owner.handle
        if violation.endswith("_flag"):
            program = owner.program
            lib = owner._library()
            fimage = ct.create_string_buffer(program.pair.forward.image.payload)
            timage = ct.create_string_buffer(program.tangent.image)
            dims = (ct.c_int64*7)(*program.pair.forward.descriptor.provenance["shape"])
            mapping = (ct.c_int*3)(*program.input_indices)
            roles = (ct.c_int*3)(*program.active)
            code = 2 if storage == "float16" else 3
            storage_code = (3 if code == 2 else 2) if violation == "storage_flag" else code
            output_code = 1 if violation == "output_flag" else code
            saved_code = 2 if violation == "saved_flag" else 1
            invalid = ct.c_uint64()
            status = lib.tessera_nvidia_attention_jvp_prepare_typed(
                fimage, len(program.pair.forward.image.payload), program.pair.forward.descriptor.entry_symbol.encode(),
                timage, len(program.tangent.image), program.tangent.entry.encode(),
                b"/invalid-sizer-must-not-be-opened", program.tangent.sizer.encode(),
                dims, None, mapping, roles, 3, storage_code, output_code, saved_code, ct.byref(invalid))
            try:
                assert status != 0 and invalid.value == 0
                reason = lib.tessera_nvidia_attention_jvp_last_error().decode()
                expected_reason = ("invalid prepared image or ABI" if violation == "storage_flag"
                                   else "saved-LSE" if violation == "saved_flag" else "storage")
                assert expected_reason in reason
            finally:
                if invalid.value:
                    lib.tessera_nvidia_attention_jvp_close(invalid.value)
        else:
            with NvidiaDeviceSession() as session:
                roots = [session.upload(x) for x in (*values, *seeds)]
                assert session.synchronize() == 0
                owner.invoke(metadata, roots)
                data = dict(roots[1].__cuda_array_interface__)
                if violation == "capacity":
                    small = session.empty((1,), dtype)
                    data["data"] = (small.ptr, False)
                elif violation == "alignment":
                    data["data"] = (roots[1].ptr+1, False)
                else:
                    data["stream"] = 0
                class Invalid:
                    dtype = roots[1].dtype
                    @property
                    def __cuda_array_interface__(self):
                        return data
                broken = [*roots]
                broken[1] = Invalid()
                with pytest.raises((ValueError, RuntimeError)):
                    owner.invoke(metadata, broken)
                assert owner.last_device_ms is None
                restored = owner.invoke(metadata, roots)
                for got, wanted in zip(restored, first, strict=True):
                    for a, b in zip(got, wanted, strict=True):
                        np.testing.assert_array_equal(a, b)
        assert owner.handle == handle
        again = fn.native_jvp(*values, tangents=seeds)
        for got, wanted in zip(again, first, strict=True):
            for a, b in zip(got, wanted, strict=True):
                np.testing.assert_array_equal(a, b)
    finally:
        fn.close_native_storage()
        clear_prepared()
