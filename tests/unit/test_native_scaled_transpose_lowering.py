"""Faithful native Schedule/Tile/Target/image projection and corruption gates."""
import base64
from dataclasses import replace
import json
import os
import re
import subprocess
import pytest
from tests.unit.test_native_scaled_transpose_export import source
from tessera.compiler.native_scaled_program import package_native_scaled_vjp


def native(text, pipeline):
    opt = os.environ.get("TESSERA_OPT")
    if not opt:
        pytest.skip("matching MLIR compiler required")
    return subprocess.run([opt, "--pass-pipeline=builtin.module("+pipeline+")"],
                          input=text, text=True, capture_output=True, timeout=180)


PREFIX = ("tessera-autodiff-paired{export-scaled-transpose=true "
          "select-scaled-transpose-member=0},tessera-graph-to-schedule")


@pytest.mark.parametrize("policy", ["shared_rhs_rows", "independent_rhs", "shared_lhs"])
@pytest.mark.parametrize("nk", [False, True])
def test_actual_reduction_descends_through_schedule_tile_and_target(policy, nk):
    graph = source(policy, nk=nk)
    scheduled = native(graph, PREFIX)
    assert scheduled.returncode == 0, scheduled.stderr
    assert "schedule.artifact" in scheduled.stdout
    assert 'algorithm = "serial_per_scale_element"' in scheduled.stdout
    tile = native(scheduled.stdout, "tessera-schedule-to-tile")
    assert tile.returncode == 0, tile.stderr
    assert "tile.structured_reduction_kernel" in tile.stdout
    assert "memref.load" in tile.stdout and "arith.bitcast" in tile.stdout
    assert not re.search(r"^\s+%.*tensor.extract", tile.stdout, re.MULTILINE)
    assert not re.search(r"^\s+%.*arith.extf.*f8", tile.stdout, re.MULTILINE)
    target = native(tile.stdout, "tessera-rocm-executable{family=reduction input=tile output=target arch=gfx1201}")
    assert target.returncode == 0, target.stderr
    assert "tessera_rocm.structured_reduction" in target.stdout
    assert not re.search(r"^\s+tile.structured_reduction_kernel", target.stdout, re.MULTILINE)


def test_schedule_rejects_changed_scalar_reduction():
    scheduled = native(source("shared_lhs"), PREFIX)
    assert scheduled.returncode == 0, scheduled.stderr
    changed, count = re.subn(r"(?m)^(\s+%[^=\n]+ = arith.constant) 128 : index$",
                             r"\1 127 : index", scheduled.stdout, count=1)
    assert count == 1
    assert changed != scheduled.stdout
    tile = native(changed, "tessera-schedule-to-tile")
    assert tile.returncode != 0
    assert "Schedule differs" in tile.stderr


def test_tile_body_seal_and_sibling_architecture_are_checked():
    tile = native(source("shared_lhs"), PREFIX+",tessera-schedule-to-tile")
    assert tile.returncode == 0, tile.stderr
    changed, count = re.subn(r'body_hash = "[0-9a-f]{64}"',
                            'body_hash = "'+"0"*64+'"', tile.stdout)
    assert count == 1
    bad = native(changed, "tessera-rocm-executable{family=reduction input=tile output=target arch=gfx1201}")
    assert bad.returncode != 0
    assert "sealed Tile contract" in bad.stderr
    sibling = native(tile.stdout, "tessera-rocm-executable{family=reduction input=tile output=target arch=gfx1151}")
    assert sibling.returncode != 0
    assert "owning gfx1201" in sibling.stderr


@pytest.mark.parametrize("policy", ["shared_rhs_rows", "independent_rhs", "shared_lhs"])
@pytest.mark.parametrize("nk", [False, True])
def test_native_reverse_images_and_member_launch_binding(policy, nk):
    if not os.environ.get("TESSERA_OPT"):
        pytest.skip("matching MLIR compiler required")
    package = package_native_scaled_vjp(source(policy, (3, 2), nk))
    program = json.loads(package.program_json)
    assert program["kind"] == "scale_vjp" and program["gradient_roles"] == [3, 2]
    assert all(image.startswith(b"\x7fELF") for image in package.images)
    assert all(step["operation"] == "tensor.generate" for step in program["steps"])
    assert program["outputs"] == [6, 5]
    assert all(json.loads(raw)["geometry"][3:] == [128, 1, 1]
               for raw in package.members_json)


def test_equal_shape_gradient_roles_cannot_be_swapped_without_output_binding():
    if not os.environ.get("TESSERA_OPT"):
        pytest.skip("matching MLIR compiler required")
    graph = source("independent_rhs").replace("7x", "2x")
    package = package_native_scaled_vjp(graph)
    program = json.loads(package.program_json)
    assert program["buffers"][2]["shape"] == program["buffers"][3]["shape"]
    program["gradient_roles"] = [3, 2]
    encoded = base64.b64encode(json.dumps(program).encode()).decode()
    members = []
    for raw in package.members_json:
        member = json.loads(raw)
        member["program_base64"] = encoded
        members.append(json.dumps(member))
    changed = replace(package, program_json=json.dumps(program), members_json=tuple(members))
    with pytest.raises(ValueError, match="requested input storage"):
        changed.validate()

@pytest.mark.parametrize("policy", ["shared_rhs_rows", "independent_rhs", "shared_lhs"])
@pytest.mark.parametrize("nk", [False, True])
def test_wave_schedule_preserves_native_body_and_checked_launch_geometry(policy, nk):
    if not os.environ.get("TESSERA_OPT"):
        pytest.skip("matching native compiler required")
    graph = source(policy, (3, 2), nk)
    pipeline = PREFIX.replace("tessera-graph-to-schedule",
                              "tessera-graph-to-schedule{scale-transpose-wave=true}")
    tile = native(graph, pipeline+",tessera-schedule-to-tile")
    assert tile.returncode == 0, tile.stderr
    assert 'algorithm = "wave_per_scale_element"' in tile.stdout
    assert "gpu.shuffle xor" in tile.stdout
    assert "scf.for" in tile.stdout and "memref.load" in tile.stdout
    package = package_native_scaled_vjp(graph, schedule="wave_per_scale_element")
    program = json.loads(package.program_json)
    for raw, step in zip(package.members_json, program["steps"], strict=True):
        member = json.loads(raw)
        count = program["buffers"][step["output"]]["elements"]
        assert member["scale_adjoint_schedule"] == "wave_per_scale_element"
        assert member["geometry"] == [count,1,1,32,1,1]
    assert program["gradient_roles"] == [3,2]


@pytest.mark.parametrize("wave", [False, True])
def test_compensated_outer_policy_is_sealed_and_native(wave):
    pipeline = PREFIX.replace(
        "tessera-graph-to-schedule",
        "tessera-graph-to-schedule{scale-transpose-wave=true}" if wave
        else "tessera-graph-to-schedule")
    scheduled = native(source("shared_rhs_rows", (3,), shape=(2,3,17,129,1536)), pipeline)
    assert scheduled.returncode == 0, scheduled.stderr
    assert 'schedule.outer_accumulation = "compensated_fp32"' in scheduled.stdout
    tile = native(scheduled.stdout, "tessera-schedule-to-tile")
    assert tile.returncode == 0, tile.stderr
    # Two carried values retain both sum and rounding residual across shared axes.
    assert re.search(r"iter_args\([^\n]+\) -> \(f32, f32\)", tile.stdout)
    assert "arith.subf" in tile.stdout
    changed = scheduled.stdout.replace("compensated_fp32", "plain_fp32")
    bad = native(changed, "tessera-schedule-to-tile")
    assert bad.returncode != 0


def test_public_schedule_option_is_explicit_and_package_cache_isolated(monkeypatch):
    from tessera.compiler import native_vjp_plugins as plugins
    from tessera.compiler import native_scaled_program as packages
    from tessera._jit_boundary import TesseraJitError
    monkeypatch.delenv("TESSERA_ROCM_SCALE_VJP_SCHEDULE", raising=False)
    assert plugins._scaled_transpose_schedule() == "serial_per_scale_element"
    monkeypatch.setenv("TESSERA_ROCM_SCALE_VJP_SCHEDULE", "wave_per_scale_element")
    assert plugins._scaled_transpose_schedule() == "wave_per_scale_element"
    monkeypatch.setenv("TESSERA_ROCM_SCALE_VJP_SCHEDULE", "unknown")
    with pytest.raises(TesseraJitError, match="unsupported native scale-VJP schedule"):
        plugins._scaled_transpose_schedule()
    calls = []
    def package(graph, *, schedule):
        calls.append(schedule)
        return object()
    monkeypatch.setattr(packages, "package_native_scaled_vjp", package)
    plugins._cached_scaled_transpose_package.cache_clear()
    try:
        serial = plugins._cached_scaled_transpose_package("graph", "compiler", 1, 2, "serial_per_scale_element")
        wave = plugins._cached_scaled_transpose_package("graph", "compiler", 1, 2, "wave_per_scale_element")
        assert serial is not wave
        assert plugins._cached_scaled_transpose_package("graph", "compiler", 1, 2, "serial_per_scale_element") is serial
        assert calls == ["serial_per_scale_element", "wave_per_scale_element"]
    finally:
        plugins._cached_scaled_transpose_package.cache_clear()
