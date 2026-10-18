"""Dispatch and source effect admission; mock owners are not device evidence."""
from types import SimpleNamespace

import pytest

from tessera import runtime as rt


@pytest.mark.parametrize("target", ["apple_gpu", "x86", "rocm_gfx1151", "rocm_gfx1201", "nvidia_sm120"])
@pytest.mark.parametrize("failure", [False, True])
def test_native_owner_checks_its_provider_without_generic_discovery(target, failure, monkeypatch):
    kind = "native_cpu" if target == "x86" else "native_gpu"
    row = SimpleNamespace(executable=True, executor_id="test_native_owner", owns_stream=True,
                          target=target, compiler_path="test_owned_program", execution_kind=kind,
                          execution_mode="owned_provider")
    called = []
    def execute(artifact, args):
        called.append((artifact, args))
        if failure:
            raise RuntimeError("owning provider rejected context")
        return rt.NativeProgramExecution((17,), ({"ok": True, "execution_kind": kind},))
    def forbidden(*args, **kwargs):
        pytest.fail("native owner dispatch probed the unrelated generic C runtime")
    monkeypatch.setattr(rt, "backend_capabilities", forbidden)
    monkeypatch.setattr(rt, "_exec_row_for_metadata", lambda metadata: row)
    monkeypatch.setattr(rt, "_executor_table", lambda: {row.executor_id: execute})
    monkeypatch.setattr(rt, "_physical_execution_attestation", lambda **kwargs: None)
    artifact = rt.RuntimeArtifact(metadata={"target": target, "compiler_path": row.compiler_path,
                                            "executable": True, "execution_kind": kind})
    result = rt.launch(artifact, ())
    assert len(called) == 1
    if failure:
        assert result["ok"] is False and "owning provider rejected context" in result["reason"]
    else:
        assert result["ok"] and result["output"] == (17,)
        assert result["physical_attestation"] is None


@pytest.mark.parametrize("dropout", [.25, float("nan")])
def test_source_dropout_refused_before_concrete_frontend_certificate(dropout):
    import numpy as np
    import tessera as ts
    from tessera._jit_boundary import TesseraJitError
    from tessera.compiler.source_attention_reverse_frontend import execute
    from tessera.compiler.native_vjp_composites import _ATTENTION_SOURCE_DECLARATION
    from tests.unit.test_native_attention_prefix_program import prefix_attention
    roots = (np.zeros((1, 2, 3, 4), np.float16), np.zeros((1, 1, 5, 4), np.float16),
             np.zeros((1, 1, 5, 3), np.float16))
    module = ts.jit(target="nvidia_sm120")(prefix_attention)._traced_autodiff_module(roots, {})
    module.functions[0].body[1].kwargs["dropout_p"] = dropout
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported stateful attention reached a concrete frontend certificate")
    jit = SimpleNamespace(_ordered_inputs=lambda *args, **kwargs: roots, frontend_differential=forbidden)
    with pytest.raises(TesseraJitError, match="before frontend certification"):
        execute(jit, module, (), {}, (None, None), _ATTENTION_SOURCE_DECLARATION)
