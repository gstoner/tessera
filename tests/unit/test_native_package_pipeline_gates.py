"""Package compiler admission retains every other capability gate."""
from dataclasses import replace

import pytest

from tessera.compiler import pipeline_gates as pg
from tests.unit.test_native_artifact_contract import _image, _descriptor


def package(target="nvidia_sm120"):
    if target == "rocm_gfx1201":
        image = _image(target=target, architecture="gfx1201",
                       pipeline_name="tessera-lower-to-rocm",
                       binary_format="hsaco", device_libraries=())
    else:
        image = _image()
    return image, _descriptor(image)


def toolchain(target, image, descriptor):
    return next(g for g in pg.evaluate(target, "matmul", native_image=image,
                                      launch_descriptor=descriptor)
                if g.gate == pg.GATE_TOOLCHAIN)


@pytest.mark.parametrize("target", ["nvidia_sm120", "rocm_gfx1201"])
@pytest.mark.parametrize("state", ["cold", "warm_cache", "prepackaged"])
def test_checked_package_retains_compiler_admission_without_driver(target, state, monkeypatch):
    monkeypatch.setattr(pg.shutil, "which", lambda *a, **k: None)
    image, descriptor = package(target)
    image = replace(image, compile_state=state)
    plain = pg.evaluate(target, "matmul")
    checked = pg.evaluate(target, "matmul", native_image=image,
                          launch_descriptor=descriptor)
    assert next(g for g in plain if g.gate == pg.GATE_TOOLCHAIN).status == pg.STATUS_FAIL
    assert next(g for g in checked if g.gate == pg.GATE_TOOLCHAIN).status == pg.STATUS_PASS
    assert [g for g in plain if g.gate != pg.GATE_TOOLCHAIN] == [
        g for g in checked if g.gate != pg.GATE_TOOLCHAIN]
    first_other = next((g for g in plain if g.gate != pg.GATE_TOOLCHAIN and
                        g.status == pg.STATUS_FAIL), None)
    assert pg.first_failing_gate(target, "matmul", native_image=image,
                                 launch_descriptor=descriptor) == first_other


@pytest.mark.parametrize("field,value", [
    ("image_digest", "f" * 64), ("entry_symbol", "missing_entry"), ("abi_id", "wrong.v1"),
])
def test_stale_package_binding_cannot_use_live_compiler(field, value, monkeypatch):
    monkeypatch.setattr(pg.shutil, "which", lambda *a, **k: "/real/compiler")
    image, descriptor = package()
    result = toolchain("nvidia_sm120", image, replace(descriptor, **{field: value}))
    assert result.status == pg.STATUS_FAIL
    assert "invalid native compiler evidence" in result.detail


@pytest.mark.parametrize("case", ["target", "architecture", "format", "payload", "missing_image", "missing_abi"])
def test_package_evidence_rejects_mismatched_or_incomplete_contract(case):
    image, descriptor = package()
    target = "nvidia_sm120"
    if case == "target":
        target = "nvidia_sm90"
    elif case == "architecture":
        image = replace(image, architecture="sm_90")
        descriptor = _descriptor(image)
    elif case == "format":
        image = replace(image, binary_format="metallib")
        descriptor = _descriptor(image)
    elif case == "payload":
        # A stale cached identity cannot hide mutated bytes.
        object.__setattr__(image, "payload", b"changed provider bytes")
    elif case == "missing_image":
        image = None
    elif case == "missing_abi":
        descriptor = None
    assert toolchain(target, image, descriptor).status == pg.STATUS_FAIL


def test_materialized_package_does_not_rewrite_dashboard_truth(monkeypatch):
    monkeypatch.setattr(pg.shutil, "which", lambda *a, **k: "/real/compiler")
    image, descriptor = package()
    with pg.deterministic_host_for_dashboard():
        assert toolchain("nvidia_sm120", image, descriptor).status == pg.STATUS_FAIL
        assert pg.evaluate("nvidia_sm120", "matmul", native_image=image,
                           launch_descriptor=descriptor) == pg.evaluate("nvidia_sm120", "matmul")


@pytest.mark.parametrize("target", ["cpu", "x86", "apple_cpu", "apple_gpu"])
def test_sibling_host_compiler_policy_retains_its_own_contract(target):
    image, descriptor = package()
    assert pg.evaluate(target, "matmul", native_image=image,
                       launch_descriptor=descriptor) == pg.evaluate(target, "matmul")


@pytest.mark.parametrize("executable,blocked", [(False, False), (True, True)])
def test_package_evidence_cannot_admit_nonexecutable_bundle_or_failed_primary(executable, blocked, monkeypatch):
    from types import SimpleNamespace
    from tessera.compiler.canonical_compile import _result_from_bundle
    image, descriptor = package()
    if blocked:
        monkeypatch.setitem(pg._EVALUATORS, pg.GATE_CODEGEN,
            lambda *a: pg.GateResult(pg.GATE_CODEGEN, pg.STATUS_FAIL, "unimplemented operation"))
    bundle = SimpleNamespace(
        request=SimpleNamespace(target=image.target), native_image=image,
        launch_descriptor=descriptor, executable=executable,
        runtime_status="artifact_only", execution_kind="artifact_only")
    result = _result_from_bundle(bundle, None, ())
    assert not result.executable and not result.program_executable
