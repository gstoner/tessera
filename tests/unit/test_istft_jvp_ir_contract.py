"""ODS-WIRE-2: the ISTFT forward product is built from compiler IR.

`tessera.istft_jvp` is produced by `ISTFTOp::buildTangent` under
`--tessera-autodiff-forward` and consumed by the GraphToSchedule arm in
`PMPasses.cpp`, which binds the product's semantic contract (resolved axis,
n_fft, hop, centering, output length, storage, and the tangent activity it
reads from the IR) into one hashed `schedule.jvp_contract`. The native JVP
plugin builds the ISTFT package from that contract.

Before this slice the plugin re-derived the same contract from the source op's
Python kwargs, so C++ forward AD and the plugin were two authorities for one
JVP (Decision #31). The kwargs derivation now survives only as a **declared
oracle** (`native_jvp_plugins._source_kwargs_spectral_arguments` for ISTFT):
every package re-derives it and refuses a disagreement, and this file is its
differential test. Host-free tests need `tessera-opt` only; the numeric tests
run on the exact device (Zen 5 AVX-512, gfx1151, gfx1201) and skip elsewhere.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import tessera
from tessera.compiler import native_jvp_plugins as plugins
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tessera.compiler.scheduled_spectral import lower_scheduled_spectral

pytestmark = pytest.mark.skipif(
    find_tessera_opt() is None, reason="production tessera-opt is not built"
)

_WRT = {"spectrum": 0, "window": 1}

# Configurations the forward transform and the scheduled ISTFT program both
# admit. Window-only differentiation is absent on purpose: the forward
# transform itself rejects it today (TangentInterface), before this consumer.
_CONFIGS = {
    "defaults": ((4, 5), 8, dict(hop=4)),
    "centered_cropped_ortho": (
        (6, 9), 12, dict(hop=3, n_fft=16, center=True, length=10, norm="ortho")
    ),
    "batched_last_axis": ((2, 5, 5), 8, dict(hop=4, axis=2)),
    "batched_inner_axis": ((5, 5, 3), 8, dict(hop=4, axis=1)),
    "forward_norm": ((3, 4, 5), 8, dict(hop=2, norm="forward")),
}
_WRT_SETS = (("spectrum",), ("spectrum", "window"))
_PROFILES = (
    ("x86", "zen5_avx512"),
    ("rocm", "gfx1151"),
    ("rocm", "gfx1201"),
    ("nvidia_sm120", "sm120"),
)


def _paired(shape, window_length, kwargs, wrt, window_dtype=np.float32):
    @tessera.jit(target="x86", autodiff="jvp", wrt=wrt)
    def product(spectrum, window):
        return tessera.ops.istft(spectrum, window, **kwargs)

    spectrum = np.zeros(shape, np.complex64)
    window = np.ones(window_length, window_dtype)
    module = product._traced_autodiff_module((spectrum, window), {})
    source = module.functions[0].body[0]
    return module, product._compile_jvp_module(module), source, spectrum, window


def _build(module, paired, source, spectrum, window, wrt, target, arch):
    return plugins.build_native_jvp_family_artifact(
        source=source, primal_inputs=[spectrum, window],
        wrt_indices=tuple(sorted(_WRT[name] for name in wrt)), target=target,
        architecture=arch, execution_mode="host_free_contract",
        source_graph_ir=module.to_mlir(), paired_jvp_ir=paired,
        arg_names=("primal_0", "primal_1", "tangent_0", "tangent_1"),
    )


@pytest.mark.parametrize("target,arch", _PROFILES)
@pytest.mark.parametrize("wrt", _WRT_SETS, ids=lambda w: "+".join(w))
@pytest.mark.parametrize("config", sorted(_CONFIGS))
def test_ir_contract_lowers_to_the_oracle_program(config, wrt, target, arch):
    """The differential test for the declared oracle: the package's scheduled
    spectral program is the one the compiler's contract names, and the kwargs
    oracle lowers to the byte-identical program."""
    shape, window_length, kwargs = _CONFIGS[config]
    module, paired, source, spectrum, window = _paired(
        shape, window_length, kwargs, wrt
    )
    contract = plugins.istft_jvp_contract_from_paired_ir(
        paired, target=target, architecture=arch
    )
    assert hashlib.sha256(
        contract_text(contract).encode()
    ).hexdigest() == contract["artifact_hash"]
    wrt_indices = tuple(sorted(_WRT[name] for name in wrt))
    assert contract["active_tangents"] == ",".join(map(str, wrt_indices))

    plan, artifact = _build(
        module, paired, source, spectrum, window, wrt, target, arch
    )
    child = plan.steps[0]["child_metadata"]["scheduled_spectral"]
    storage = "f32"
    oracle = lower_scheduled_spectral(**plugins._source_kwargs_spectral_arguments(
        source=source, primal_inputs=[spectrum, window], storage=storage,
        target=target, architecture=arch,
    )).to_metadata()
    compiled = lower_scheduled_spectral(**plugins.istft_jvp_spectral_arguments(
        contract, primal_inputs=[spectrum, window], wrt_indices=wrt_indices,
        target=target, architecture=arch,
    )).to_metadata()
    assert compiled == oracle
    assert child == compiled
    schedule = artifact.contract["schedule_program"]
    assert schedule["graph_schedule_artifact"] == contract["artifact_hash"]
    assert schedule["graph_schedule_consumer"] == (
        "tessera-graph-to-schedule:tessera.istft_jvp"
    )


def contract_text(contract):
    keys = (
        "schema", "kind", "target", "arch", "spectrum", "window", "tangent",
        "axis", "logical_length", "hop", "frames", "center", "onesided",
        "pad_mode", "output_length", "normalization", "spectrum_layout",
        "window_broadcast", "numeric_storage", "numeric_accum",
        "active_tangents", "mutation_lineage",
    )
    assert set(contract) == set(keys) | {"artifact_hash"}
    return ";".join(f"{key}={contract[key]}" for key in keys)


def test_activity_is_read_from_the_ir_not_the_request():
    module, paired, source, spectrum, window = _paired(
        (4, 5), 8, dict(hop=4), ("spectrum",)
    )
    contract = plugins.istft_jvp_contract_from_paired_ir(
        paired, target="x86", architecture="zen5_avx512"
    )
    assert contract["active_tangents"] == "0"
    with pytest.raises(ValueError, match="activity .* disagrees with wrt"):
        plugins.istft_jvp_spectral_arguments(
            contract, primal_inputs=[spectrum, window], wrt_indices=(0, 1),
            target="x86", architecture="zen5_avx512",
        )


def test_a_disagreeing_oracle_refuses_the_package(monkeypatch):
    module, paired, source, spectrum, window = _paired(
        (4, 5), 8, dict(hop=4), ("spectrum", "window")
    )
    real = plugins._source_kwargs_spectral_arguments

    def drifted(**kwargs):
        arguments = real(**kwargs)
        return {**arguments, "normalization": "ortho"}

    monkeypatch.setattr(plugins, "_source_kwargs_spectral_arguments", drifted)
    with pytest.raises(ValueError, match="lower to different spectral programs"):
        _build(module, paired, source, spectrum, window,
               ("spectrum", "window"), "x86", "zen5_avx512")


def test_the_package_needs_the_compiler_contract():
    module, paired, source, spectrum, window = _paired(
        (4, 5), 8, dict(hop=4), ("spectrum", "window")
    )
    with pytest.raises(ValueError, match="requires the scheduled tessera.istft_jvp"):
        plugins.plan_native_jvp_family(
            source=source, primal_inputs=[spectrum, window], wrt_indices=(0, 1),
            target="x86", architecture="zen5_avx512", execution_mode="x",
        )


@pytest.mark.parametrize("target,arch", [("rocm", "gfx1200"), ("x86", "zen4"),
                                         ("apple_gpu", "apple7")])
def test_unproven_profiles_fail_closed(target, arch):
    _, paired, *_ = _paired((4, 5), 8, dict(hop=4), ("spectrum",))
    with pytest.raises(ValueError, match="no exact Schedule profile"):
        plugins.istft_jvp_contract_from_paired_ir(
            paired, target=target, architecture=arch
        )


def test_reduced_precision_window_is_refused_while_types_disagree():
    """The frontend types an f16-window ISTFT result as f32 while the native
    packages emit window storage. The Schedule consumer refuses that
    disagreement (a dtype is a semantic key, #21a) instead of letting the
    package output contradict the IR type."""
    _, paired, *_ = _paired((4, 5), 8, dict(hop=4), ("spectrum", "window"),
                            window_dtype=np.float16)
    with pytest.raises(ValueError, match="SPECTRAL_JVP_SCHEDULE_REFUSED"):
        plugins.istft_jvp_contract_from_paired_ir(
            paired, target="x86", architecture="zen5_avx512"
        )


def test_a_profile_already_on_the_module_is_refused():
    _, paired, *_ = _paired((4, 5), 8, dict(hop=4), ("spectrum",))
    tagged = paired.replace(
        "module attributes {", 'module attributes {tessera.target = "x86", ', 1
    )
    with pytest.raises(ValueError, match="already names a target profile"):
        plugins.istft_jvp_contract_from_paired_ir(
            tagged, target="x86", architecture="zen5_avx512"
        )


# ── Exact-device differential: the IR-built package against the reference ──

@tessera.jit(target="x86", autodiff="jvp", wrt=("spectrum", "window"))
def _x86_both(spectrum, window):
    return tessera.ops.istft(spectrum, window, hop=4)


@tessera.jit(target="x86", autodiff="jvp", wrt=("spectrum",))
def _x86_spectrum_centered(spectrum, window):
    return tessera.ops.istft(spectrum, window, hop=3, n_fft=16, center=True,
                             length=10, norm="ortho")


@tessera.jit(target="rocm", autodiff="jvp", wrt=("spectrum", "window"))
def _rocm_both(spectrum, window):
    return tessera.ops.istft(spectrum, window, hop=4)


@tessera.jit(target="rocm", autodiff="jvp", wrt=("spectrum",))
def _rocm_spectrum_centered(spectrum, window):
    return tessera.ops.istft(spectrum, window, hop=3, n_fft=16, center=True,
                             length=10, norm="ortho")


def _case(name, rng):
    if name == "both":
        shape, window_length, kwargs = (4, 5), 8, dict(hop=4)
    else:
        shape, window_length, kwargs = (6, 9), 12, dict(
            hop=3, n_fft=16, center=True, length=10, norm="ortho"
        )
    spectrum = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
        np.complex64
    )
    spectrum[..., (0, -1)] = spectrum[..., (0, -1)].real
    window = (np.hanning(window_length) + 0.2).astype(np.float32)
    dspectrum = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
        np.complex64
    )
    dspectrum[..., (0, -1)] = dspectrum[..., (0, -1)].real
    dwindow = rng.normal(size=window.shape).astype(np.float32)
    return spectrum, window, dspectrum, dwindow, kwargs


def _check_against_reference(compiled, name, rtol):
    rng = np.random.default_rng(4231 if name == "both" else 4241)
    spectrum, window, dspectrum, dwindow, kwargs = _case(name, rng)
    both = name == "both"
    tangents = (dspectrum, dwindow) if both else (dspectrum,)
    primal, tangent = compiled.native_jvp(spectrum, window, tangents=tangents)
    eps = np.float32(2.0e-3)
    step_window = eps * dwindow if both else 0.0
    plus = tessera.ops.istft(spectrum + eps * dspectrum, window + step_window, **kwargs)
    minus = tessera.ops.istft(spectrum - eps * dspectrum, window - step_window, **kwargs)
    oracle = (np.asarray(plus) - np.asarray(minus)) / (2.0 * eps)
    np.testing.assert_allclose(
        primal, tessera.ops.istft(spectrum, window, **kwargs), rtol=rtol, atol=rtol
    )
    np.testing.assert_allclose(tangent, oracle, rtol=3e-3, atol=3e-3)
    (package,) = compiled._native_jvp_packages.values()
    schedule = package.contract["schedule_program"]
    assert len(schedule["graph_schedule_artifact"]) == 64
    assert schedule["graph_schedule_consumer"] == (
        "tessera-graph-to-schedule:tessera.istft_jvp"
    )


@pytest.mark.compiler_avx512
@pytest.mark.parametrize("name,compiled", [
    ("both", _x86_both), ("centered", _x86_spectrum_centered),
])
def test_x86_ir_built_istft_product_matches_reference(name, compiled):
    from tessera import runtime
    if not runtime._x86_elementwise_available():
        pytest.skip("AVX-512 spectral package unavailable on this host")
    lib = runtime._load_x86_elementwise()
    if lib is None or not hasattr(lib, "tessera_x86_istft_jvp_f32"):
        pytest.skip("AVX-512 ISTFT window-JVP image is stale")
    _check_against_reference(compiled, name, rtol=3e-5)


@pytest.mark.hardware_rocm
@pytest.mark.parametrize("name,compiled", [
    ("both", _rocm_both), ("centered", _rocm_spectrum_centered),
])
def test_rocm_ir_built_istft_product_matches_reference(name, compiled):
    from tessera import runtime
    chip = runtime._rocm_live_arch()
    if runtime._tessera_opt_path() is None or chip not in ("gfx1151", "gfx1201"):
        pytest.skip("exact gfx1151 or gfx1201 compiler/device required")
    _check_against_reference(compiled, name, rtol=3e-4)


def test_x86_window_product_refuses_geometry_its_symbol_cannot_express():
    """The x86/ROCm window-product symbols take no n_fft/center/length; a
    centered or cropped window product is refused before launch instead of
    overrunning the cropped output buffer."""
    from tessera import runtime

    module, paired, source, spectrum, window = _paired(
        (6, 9), 12, dict(hop=3, n_fft=16, center=True, length=10),
        ("spectrum", "window"),
    )
    plan, _ = _build(module, paired, source, spectrum, window,
                     ("spectrum", "window"), "x86", "zen5_avx512")
    child = dict(plan.steps[0]["child_metadata"])
    with pytest.raises(ValueError, match="supports only n_fft == window"):
        runtime._execute_compiled_spectral_jvp(
            runtime.RuntimeArtifact(metadata=child),
            (spectrum, window, spectrum, window.copy()), target="x86",
        )
