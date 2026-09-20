"""`ROCMExecutablePipeline.cache_key` is the one authority for ROCm compile-cache identity.

ROCM-PIPELINE-KEY-1. The method used to be a Decision #29 unconsumed
declaration -- no caller anywhere -- listing ten of the config's fourteen
fields by hand. The four it omitted (`sched_groups`, `lds_pad_dwords`,
`lds_copy_width`, `lds_copy_elide`) all change the generated kernel, and
`lds_copy_elide` selects a *deliberately wrong* all-zero ceiling-probe kernel
whose staging copy never reads global memory. A consumer wired to that key
would have served the probe kernel for a real request.

It never fired only because the real compile cache in `rocm_native` keyed on
its own separate string that happened to include all four -- which is the
failure Decision #31 names: two spellings of one identity, and the one nothing
ran was the one that drifted.

So the fix is not "add the four fields". It is to make a field impossible to
omit (derive the key from the pipeline option string every knob already flows
through) and to give the method its consumer. These tests are what would fail
if either half regressed:

  * every field of the config changes the key, checked field by field against
    a table this file also proves complete, so a *new* knob that forgets the
    key fails here rather than on a device;
  * the native cache key reads the config only through `cache_key()`, so a
    second spelling cannot come back.
"""

from __future__ import annotations

import dataclasses

import pytest

from tessera.compiler.rocm_native import _native_cache_key
from tessera.compiler.rocm_pipeline import (
    ROCMExecutablePipeline,
    ROCMInputLevel,
    ROCMOutputLevel,
)


#: field name -> (extra base overrides, alternative value).
#:
#: The overrides exist because two fields are not independently legal:
#: `depth_cooperative` is rejected by `__post_init__` off the
#: `depth_attention` family, and a family must be promoted on the arch it is
#: paired with. Every pair below is a config the pipeline actually accepts --
#: the test constructs both sides, so an illegal pair fails loudly rather than
#: quietly skipping a field.
_FIELD_ALTERNATIVES: dict[str, tuple[dict[str, object], object]] = {
    "family": ({}, "softmax"),
    "input_level": ({}, ROCMInputLevel.GRAPH),
    "output_level": ({}, ROCMOutputLevel.TARGET),
    # matmul is promoted on both AMD parts, so this pair is legal; proof never
    # transfers between them, and neither may a cached kernel.
    "arch": ({}, "gfx1201"),
    "staging": ({}, "lds"),
    "lds_waves": ({}, (2, 4)),
    "k_unroll": ({}, 2),
    "sched_groups": ({}, 2),
    "lds_pad_dwords": ({}, 0),
    "lds_copy_width": ({}, 4),
    "lds_copy_elide": ({}, True),
    # Added by PR #787 (LDS issue depth / double-buffering / schedule
    # description / transposable B). They needed no change to `cache_key`:
    # the derived key picked them up from `pass_pipeline` on the rebase, and
    # this table is the only thing that had to be touched -- which is the
    # completeness guard below doing its job rather than a gap in it.
    "lds_copy_depth": ({}, 2),
    "lds_double_buffer": ({}, True),
    "lds_sched_valu_per_mma": ({}, 2),
    "lds_b_row_major": ({}, True),
    "tile_q": ({}, 128),
    "tile_kv": ({}, 128),
    "depth_cooperative": ({"family": "depth_attention"}, True),
}


def _base(**overrides: object) -> ROCMExecutablePipeline:
    kwargs: dict[str, object] = {"family": "matmul"}
    kwargs.update(overrides)
    return ROCMExecutablePipeline(**kwargs)  # type: ignore[arg-type]


def test_alternatives_table_covers_every_field() -> None:
    """A new knob must be added here, so the per-field check below cannot
    silently stop covering the config."""
    declared = {f.name for f in dataclasses.fields(ROCMExecutablePipeline)}
    assert declared == set(_FIELD_ALTERNATIVES), (
        "ROCMExecutablePipeline fields and the cache-key alternatives table "
        f"disagree: only in config {sorted(declared - set(_FIELD_ALTERNATIVES))}, "
        f"only in table {sorted(set(_FIELD_ALTERNATIVES) - declared)}"
    )


@pytest.mark.parametrize("field", sorted(_FIELD_ALTERNATIVES))
def test_every_config_field_changes_the_cache_key(field: str) -> None:
    extra, alternative = _FIELD_ALTERNATIVES[field]
    default = _base(**extra)
    assert getattr(default, field) != alternative, (
        f"the alternative for {field!r} equals the default; it cannot prove "
        "the field is keyed"
    )
    changed = _base(**extra, **{field: alternative})
    assert default.cache_key() != changed.cache_key(), (
        f"{field!r} changes the generated kernel but not the cache key: two "
        "different kernels would share a cache entry"
    )


def test_cache_key_is_stable_across_equal_configs() -> None:
    assert _base().cache_key() == _base().cache_key()


def test_the_ceiling_probe_never_shares_a_key_with_production() -> None:
    """`lds_copy_elide` is the field with teeth.

    It emits a kernel whose staging copy writes a constant instead of reading
    global memory, so every output is zero. It exists to bound what a copy
    optimisation can buy and is never a production path -- which is exactly
    why it must never collide with one.
    """
    production = _base(staging="lds")
    probe = _base(staging="lds", lds_copy_elide=True)
    assert production.cache_key() != probe.cache_key()
    assert _native_cache_key(
        production, **_NATIVE_CONTEXT
    ) != _native_cache_key(probe, **_NATIVE_CONTEXT)


_NATIVE_CONTEXT: dict[str, str] = {
    "tile_ir": "module {}",
    "directive": "tessera_rocm.wmma_gemm",
    "library_identity": "ocml:deadbeef:full",
    "tool_digest": "0" * 64,
}


@pytest.mark.parametrize("field", sorted(_FIELD_ALTERNATIVES))
def test_native_cache_key_separates_every_config_field(field: str) -> None:
    extra, alternative = _FIELD_ALTERNATIVES[field]
    default = _native_cache_key(_base(**extra), **_NATIVE_CONTEXT)
    changed = _native_cache_key(
        _base(**extra, **{field: alternative}), **_NATIVE_CONTEXT
    )
    assert default != changed


@pytest.mark.parametrize(
    "context_field", sorted(_NATIVE_CONTEXT), ids=sorted(_NATIVE_CONTEXT)
)
def test_native_cache_key_separates_its_non_config_halves(context_field: str) -> None:
    """What is compiled and what compiles it are keyed too -- a toolchain or
    device-library change must miss rather than serve a kernel built by
    different code (Decision #11)."""
    config = _base()
    changed = dict(_NATIVE_CONTEXT)
    changed[context_field] = _NATIVE_CONTEXT[context_field] + "-other"
    assert _native_cache_key(config, **_NATIVE_CONTEXT) != _native_cache_key(
        config, **changed
    )


def test_native_cache_key_reads_the_config_only_through_cache_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decision #31, made checkable: one authority for config identity.

    With `cache_key` pinned to a constant, two configs that differ in every
    keyed knob must collide -- which is only true if `_native_cache_key` reads
    no config field of its own. A re-introduced second spelling fails here.
    """
    monkeypatch.setattr(
        ROCMExecutablePipeline, "cache_key", lambda self: ("pinned",)
    )
    left = _base()
    right = _base(
        family="softmax",
        arch="gfx1201",
        staging="lds",
        input_level=ROCMInputLevel.GRAPH,
        output_level=ROCMOutputLevel.TARGET,
        lds_waves=(4, 4),
        k_unroll=3,
        sched_groups=5,
        lds_pad_dwords=0,
        lds_copy_width=8,
        lds_copy_elide=True,
        tile_q=128,
        tile_kv=32,
    )
    assert _native_cache_key(left, **_NATIVE_CONTEXT) == _native_cache_key(
        right, **_NATIVE_CONTEXT
    )


# ---------------------------------------------------------------------------
# The consumers in `runtime.py`.
#
# Six sites build a `ROCMExecutablePipeline` and cache what it compiles. Each
# used to hand-spell a parallel key tuple beside the config -- the shape that
# drifted inside `cache_key` itself. All six were complete when audited
# (2026-09-20), including `staging` at the canonical GEMM and `chip` in all 73
# callers of the shared family helper; what they lacked was anything that would
# *keep* them complete. These two tests are that.
# ---------------------------------------------------------------------------

import ast
import subprocess as _subprocess
import types
from pathlib import Path

_RUNTIME = Path(__file__).resolve().parents[2] / "python" / "tessera" / "runtime.py"


def _functions_building_a_rocm_pipeline() -> dict[str, ast.FunctionDef]:
    tree = ast.parse(_RUNTIME.read_text())
    found: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "ROCMExecutablePipeline"
            ):
                found[node.name] = node  # type: ignore[assignment]
                break
    return found


def test_every_runtime_pipeline_site_keys_on_the_authority() -> None:
    """A site that builds a ROCm pipeline and consults neither `cache_key()`
    nor `_rocm_lane_config` (which returns it) is either not caching, or
    caching on a second spelling of the config."""
    sites = _functions_building_a_rocm_pipeline()
    assert len(sites) == 7, (
        "expected the six lane sites plus `_rocm_lane_config`, found "
        f"{sorted(sites)}"
    )
    missing = [
        name
        for name, node in sites.items()
        if not any(
            (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "cache_key"
            )
            or (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "_rocm_lane_config"
            )
            for inner in ast.walk(node)
        )
    ]
    assert not missing, (
        f"{missing} build a ROCm pipeline but do not key on its identity "
        "(ROCM-PIPELINE-KEY-1)"
    )


def test_an_unbuildable_config_does_not_move_the_raise() -> None:
    """`_rocm_lane_config` returns None rather than raising, and that is
    load-bearing: `_build_compiled_gemm_hsaco`'s arch guard admits the whole
    gfx11 family while only gfx1151 is promoted, so keying on an eagerly built
    config turned the gfx1100 refusal from "tessera-opt not built" into a
    `ValueError` one check earlier. A cache-key change must not move a raise.
    """
    from tessera import runtime as rt
    from tessera.compiler.rocm_pipeline import ROCMInputLevel

    spec = dict(family="matmul", input_level=ROCMInputLevel.DIRECTIVE, arch="gfx1100")
    config, identity = rt._rocm_lane_config(**spec)
    assert config is None
    assert identity[0] == "unbuildable"
    # and the refusal is still available to the lane, unchanged
    with pytest.raises(ValueError, match="no promoted family plugins for gfx1100"):
        ROCMExecutablePipeline(**spec)  # type: ignore[arg-type]

    promoted, promoted_identity = rt._rocm_lane_config(
        family="matmul", input_level=ROCMInputLevel.DIRECTIVE, arch="gfx1151"
    )
    assert promoted is not None
    assert promoted_identity == promoted.cache_key()
    assert promoted_identity != identity


def test_no_runtime_pipeline_site_respells_the_architecture() -> None:
    """`chip` selects the kernel's ISA and reaches the key through the
    config's `arch`. A site that also lists it by hand has re-created the two
    spellings, and it is the hand-written one that goes stale."""
    offenders = []
    for name, node in _functions_building_a_rocm_pipeline().items():
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Assign):
                continue
            targets = {t.id for t in inner.targets if isinstance(t, ast.Name)}
            if "key" not in targets:
                continue
            names = {n.id for n in ast.walk(inner.value) if isinstance(n, ast.Name)}
            if "chip" in names:
                offenders.append(f"{name}:{inner.lineno}")
    assert not offenders, (
        f"cache keys re-spell the architecture beside the config: {offenders}"
    )


@pytest.fixture()
def _fake_rocm_compiler(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Drive `_build_rocm_family_hsaco` with no device and no `tessera-opt`.

    A cache miss reaches the subprocess; a hit does not. Recording the
    directives that reach it is therefore a direct readout of what the key
    separated.
    """
    from tessera import runtime as rt

    compiled: list[str] = []

    def fake_run(_cmd, **kwargs):  # noqa: ANN001
        compiled.append(kwargs["input"])
        return types.SimpleNamespace(returncode=0, stdout="gpu.binary", stderr="")

    monkeypatch.setattr(rt, "_tessera_opt_path", lambda: Path("/nonexistent/tessera-opt"))
    monkeypatch.setattr(rt, "_rocm_serializer_env", dict)
    monkeypatch.setattr(rt, "_extract_hsaco_blob", lambda _out: b"\x7fELF-fake")
    monkeypatch.setattr(_subprocess, "run", fake_run)
    return compiled


def test_family_helper_separates_architectures_the_caller_key_shares(
    monkeypatch: pytest.MonkeyPatch, _fake_rocm_compiler: list[str]
) -> None:
    """The regression the helper's own keying exists to prevent.

    `key` here deliberately omits `chip`, as a caller that forgot it would.
    Without the config's identity in the key, the gfx1201 request is served
    the gfx1151 kernel -- a kernel for another ISA, from cache, silently.
    """
    from tessera import runtime as rt

    cache: dict = {}
    directive = 'module {\n  "tessera_rocm.softmax"() {name = "sm"} : () -> ()\n}\n'

    monkeypatch.setenv("TESSERA_ROCM_CHIP", "gfx1151")
    luna = rt._build_rocm_family_hsaco("softmax", directive, cache, ("dtype:f32",))
    monkeypatch.setenv("TESSERA_ROCM_CHIP", "gfx1201")
    tajasarus = rt._build_rocm_family_hsaco("softmax", directive, cache, ("dtype:f32",))

    assert len(_fake_rocm_compiler) == 2, (
        "the gfx1201 request was served from the gfx1151 entry: proof never "
        "transfers between the two AMD parts, and neither may a kernel"
    )
    assert len(cache) == 2
    assert luna == tajasarus  # the fake compiler returns one blob; the KEYS differ


def test_family_helper_separates_directives_the_caller_key_shares(
    _fake_rocm_compiler: list[str],
) -> None:
    """Same net, other axis: the directive is the whole compiler input, so two
    different directives under one caller key must not share an entry."""
    from tessera import runtime as rt

    cache: dict = {}
    for dtype in ("f32", "f16"):
        rt._build_rocm_family_hsaco(
            "softmax",
            f'module {{\n  "tessera_rocm.softmax"() {{dtype = "{dtype}"}} : () -> ()\n}}\n',
            cache,
            ("same-key-for-both",),
        )
    assert len(_fake_rocm_compiler) == 2
    assert len(cache) == 2


def test_family_helper_still_caches_identical_requests(
    _fake_rocm_compiler: list[str],
) -> None:
    """The net must not cost the cache: identical inputs compile once."""
    from tessera import runtime as rt

    cache: dict = {}
    directive = 'module {\n  "tessera_rocm.softmax"() {dtype = "f32"} : () -> ()\n}\n'
    for _ in range(3):
        rt._build_rocm_family_hsaco("softmax", directive, cache, ("k",))
    assert len(_fake_rocm_compiler) == 1
    assert len(cache) == 1
