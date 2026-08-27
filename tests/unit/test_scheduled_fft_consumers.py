"""Content-addressed FFT Schedule -> Tile package boundary."""

from __future__ import annotations

import copy

import pytest

from tessera.compiler.scheduled_fft import (
    lower_scheduled_fft,
    validate_scheduled_fft_metadata,
)
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

# These lower through the production `tessera-opt`, which the CI unit lane does
# not build. The library correctly RAISES rather than silently degrading, so
# the tests must skip there rather than fail.
_needs_opt = pytest.mark.skipif(
    find_tessera_opt() is None, reason="tessera-opt not built"
)


@pytest.mark.parametrize(
    ("target", "op_name", "shape", "axis", "n", "strategy", "radices"),
    [
        ("x86", "tessera.fft", (2, 16), -1, None, "radix2", (2, 2, 2, 2)),
        ("x86", "tessera.irfft", (2, 51), -1, 100, "mixed_radix", (2, 5, 5)),
        ("x86", "tessera.fft", (2, 255), -1, None, "bluestein", ()),
        ("x86", "tessera.fft", (2, 768), -1, None, "mixed_radix", (4, 4, 4, 4, 3)),
        ("rocm", "tessera.fft", (3, 100), -1, None, "mixed_radix", (4, 5, 5)),
        ("rocm", "tessera.fft", (2, 68), -1, None, "mixed_radix", (4, 17)),
        ("rocm", "tessera.fft", (257,), -1, None, "bluestein", ()),
        ("rocm", "tessera.rfft", (4, 16, 3), 1, None, "mixed_radix", (4, 2)),
    ],
)
@_needs_opt
def test_fft_lowers_to_one_content_addressed_tile_package(
    target, op_name, shape, axis, n, strategy, radices
):
    artifact = lower_scheduled_fft(
        target=target,
        op_name=op_name,
        input_shape=shape,
        axis=axis,
        n=n,
    )

    assert artifact.strategy == strategy
    assert artifact.radix_sequence == radices
    assert artifact.schedule_ir.count(artifact.schedule_digest) == 3
    assert artifact.tile_ir.count("tile.fft_kernel") == 1
    assert "schedule.fft" not in artifact.tile_ir
    assert "schedule.artifact" not in artifact.tile_ir
    validate_scheduled_fft_metadata(
        artifact.to_metadata(), target=target, input_shape=shape
    )


@pytest.mark.parametrize(
    "field",
    [
        "tile_digest",
        "schedule_digest",
        "strategy",
        "residency",
        "workspace_policy",
        "twiddle_policy",
        "batch",
        "physical_length",
        "real_transform_policy",
    ],
)
@_needs_opt
def test_fft_runtime_contract_rejects_tampering(field):
    artifact = lower_scheduled_fft(
        target="rocm", op_name="tessera.fft", input_shape=(3, 100)
    )
    metadata = copy.deepcopy(artifact.to_metadata())
    if field in {"tile_digest", "schedule_digest"}:
        metadata[field] = "0" * 64
    elif field == "strategy":
        metadata[field] = "bluestein"
    elif field in {"residency", "workspace_policy", "twiddle_policy"}:
        metadata[field] = "stale_policy"
    else:
        metadata[field] = 4

    with pytest.raises(ValueError, match="FFT package"):
        validate_scheduled_fft_metadata(metadata, target="rocm", input_shape=(3, 100))


@_needs_opt
def test_schedule_to_tile_rederives_and_rejects_stale_fft_policy():
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("production tessera-opt is not built")
    artifact = lower_scheduled_fft(
        target="rocm", op_name="tessera.fft", input_shape=(3, 100)
    )
    stale = artifact.schedule_ir.replace(
        'kernel_family = "gfx1151_stockham_bluestein_v6"',
        'kernel_family = "stale_kernel"',
    )

    with pytest.raises(RuntimeError, match="FFT policy was altered after hashing"):
        run_tessera_opt(tool, stale, "--tessera-schedule-to-tile")


@_needs_opt
def test_rocm_bluestein_artifact_describes_persistent_native_plan():
    artifact = lower_scheduled_fft(
        target="rocm", op_name="tessera.fft", input_shape=(2, 257)
    )

    assert artifact.workspace_elems == 4 * 1024
    assert artifact.workspace_policy == "persistent_plan_4m"
    assert artifact.residency == "persistent_device_plan"
    assert artifact.twiddle_policy == "persistent_device_chirp_fft"
    assert artifact.kernel_family == "gfx1151_stockham_bluestein_v6"


@_needs_opt
def test_rocm_small_power_of_two_artifact_selects_batched_fused_lds():
    artifact = lower_scheduled_fft(
        target="rocm", op_name="tessera.fft", input_shape=(56, 256)
    )

    assert artifact.batch == 56
    assert artifact.residency == "persistent_device_plan_fused_lds_batch"
    assert artifact.kernel_family == "gfx1151_stockham_bluestein_v6"
    assert 'residency = "persistent_device_plan_fused_lds_batch"' in artifact.tile_ir


@_needs_opt
@pytest.mark.parametrize("target", ["x86", "rocm"])
def test_even_real_fft_artifact_names_half_length_physical_transform(target):
    artifact = lower_scheduled_fft(
        target=target, op_name="tessera.rfft", input_shape=(3, 256)
    )

    assert artifact.length == 256
    assert artifact.physical_length == 128
    assert artifact.real_transform_policy == "packed_even_n2_hermitian_v1"
    assert artifact.hermitian_layout == "half_spectrum_nyquist_explicit"
    assert "physical_length = 128 : i64" in artifact.tile_ir
    if target == "rocm":
        assert artifact.residency == "persistent_device_plan_fused_lds_hermitian_batch"
        assert (
            'residency = "persistent_device_plan_fused_lds_hermitian_batch"'
            in artifact.tile_ir
        )


@_needs_opt
@pytest.mark.parametrize("target", ["x86", "rocm"])
def test_odd_real_fft_artifact_keeps_explicit_full_complex_fallback(target):
    artifact = lower_scheduled_fft(
        target=target, op_name="tessera.rfft", input_shape=(2, 255)
    )

    assert artifact.physical_length == 255
    assert artifact.real_transform_policy == "full_complex_hermitian_fallback"


@_needs_opt
@pytest.mark.parametrize("architecture", ["gfx1200", "gfx1250"])
def test_unmeasured_rdna4_fft_profiles_fail_closed(architecture):
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("production tessera-opt is not built")
    source = f'''module attributes {{tessera.target = "rocm", tessera.arch = "{architecture}"}} {{
  func.func @fft(%x: tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>> {{
    %0 = "tessera.fft"(%x) {{axis = -1 : i64}} : (tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>>
    return %0 : tensor<16xcomplex<f32>>
  }}
}}
'''

    with pytest.raises(RuntimeError, match="Zen 5, gfx1151, or exact sm120"):
        run_tessera_opt(tool, source, "--tessera-graph-to-schedule")
