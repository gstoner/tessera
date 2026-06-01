"""Buffer-pool migration regression tests.

Locks the invariant established by the 2026-05-18 sweep:

  1. Every dispatcher in ``apple_gpu_runtime.mm`` acquires its buffers
     through the ``TS_METAL_BUF_ACQUIRE`` / ``TS_METAL_BUF_ACQUIRE_WITH_BYTES``
     macros (RAII guards).  Raw ``[ctx.device newBufferWith{Bytes,Length}:]``
     calls are forbidden outside the pool primitives.
  2. Explicit ``metal_buffer_release(ctx, ...)`` calls are forbidden —
     the RAII guards handle release at scope exit on every path
     (success, early ``return false;``, exception).

These contracts together mean every early-return path is
release-safe by construction — a regression would silently leak
buffers into the pool's shadow.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

RUNTIME_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "src" / "compiler" / "codegen" / "Tessera_Apple_Backend"
    / "runtime" / "apple_gpu_runtime.mm"
)


@pytest.fixture(scope="module")
def runtime_src() -> str:
    return RUNTIME_PATH.read_text()


def _strip_comments(text: str) -> str:
    """Remove ``//`` line comments and ``/* */`` block comments so
    the lexical checks don't match documentation."""
    no_line = re.sub(r"//[^\n]*", "", text)
    no_block = re.sub(r"/\*.*?\*/", "", no_line, flags=re.DOTALL)
    return no_block


def test_no_raw_newbuffer_in_dispatchers(runtime_src: str) -> None:
    """Raw ``[ctx.device newBufferWith*]`` calls survive only inside
    the two pool primitives (``metal_buffer_acquire`` and the
    overflow branch).  Anything else means a dispatcher slipped past
    the migration and would leak on early ``return false``."""
    src = _strip_comments(runtime_src)
    # Find raw newBuffer* calls.
    raw = list(re.finditer(
        r"\[ctx\.device\s+newBufferWith(?:Bytes|Length)", src,
    ))

    def _fn_range(signature_re: str) -> tuple[int, int]:
        m = re.search(signature_re, src)
        assert m is not None, f"function not found: {signature_re}"
        brace = src.find("{", m.end())
        depth = 1
        pos = brace + 1
        while pos < len(src) and depth > 0:
            if src[pos] == "{":
                depth += 1
            elif src[pos] == "}":
                depth -= 1
            pos += 1
        return m.start(), pos

    # Raw calls are allowed inside the pool primitive (per-call buffers) AND
    # inside ts_dev_alloc — the R0 persistent device-tensor allocator, whose
    # buffer is long-lived and freed explicitly via ts_dev_free, so the
    # "leak on early return" rationale doesn't apply.
    # Apple-sample Pattern 4 (2026-05-31) also allows the tiny 64-byte
    # diagnostic buffer in ``tessera_apple_gpu_commit_and_wait_timeout_probe``
    # — the probe is testing infrastructure, not a workload; using the
    # buffer pool for a 64-byte one-shot would obscure rather than help.
    allowed = [
        _fn_range(r"static\s+id<MTLBuffer>\s+metal_buffer_acquire\s*\("),
        _fn_range(r"TsDeviceTensor\s*\*\s*ts_dev_alloc\s*\("),
        _fn_range(r"tessera_apple_gpu_commit_and_wait_timeout_probe\s*\("),
        # Phase 3b (2026-06-01) — bf16 MSL encode helpers allocate
        # fp32 scratch buffers for the on-GPU bf16→fp32→bf16 cast
        # chain. The encoded MPSGraph cast nodes + MSL kernel retain
        # these via the cb until commit; using the buffer pool would
        # risk aliasing because the pool would consider the buffer
        # "free" the moment the function returns, but the GPU is
        # still reading from it inside the deferred encoded chain.
        # Direct allocation gives the cb-retained lifetime that the
        # encode-session semantics need.
        _fn_range(r"tessera_apple_gpu_rope_dev_bf16_enc\s*\("),
        _fn_range(r"tessera_apple_gpu_flash_attn_dev_bf16_enc\s*\("),
    ]
    offending = [
        m for m in raw
        if not any(lo <= m.start() <= hi for lo, hi in allowed)
    ]
    assert not offending, (
        f"{len(offending)} raw [ctx.device newBufferWith*] call(s) "
        f"outside metal_buffer_acquire: first at offset {offending[0].start()}"
    )


def test_no_explicit_release_calls(runtime_src: str) -> None:
    """Explicit ``metal_buffer_release(ctx, ...)`` calls are forbidden
    in dispatchers — RAII guards (``MetalBufferGuard``) handle
    release at scope exit.  The only legitimate use is the destructor
    of the guard itself."""
    src = _strip_comments(runtime_src)
    # Allowed sites: the MetalBufferGuard destructor body (which lives
    # inside `struct MetalBufferGuard { ... ~MetalBufferGuard() { ... } };`).
    # Carve out that struct, then check the rest.
    guard_re = re.compile(r"struct\s+MetalBufferGuard\s*\{")
    gm = guard_re.search(src)
    assert gm is not None, "MetalBufferGuard struct missing"
    brace = src.find("{", gm.end() - 1)
    depth = 1; pos = brace + 1
    while pos < len(src) and depth > 0:
        if src[pos] == "{": depth += 1
        elif src[pos] == "}": depth -= 1
        pos += 1
    guard_end = pos
    before = src[:gm.start()]
    after = src[guard_end:]
    rest = before + after
    # Match only call sites (``metal_buffer_release(ctx, ...)``), not
    # the function definition line (``static void metal_buffer_release(
    # MetalDeviceContext &ctx, ...``).
    leaks = list(re.finditer(r"metal_buffer_release\s*\(\s*ctx\b", rest))
    assert not leaks, (
        f"{len(leaks)} explicit metal_buffer_release call(s) outside the "
        f"MetalBufferGuard destructor: first match at offset {leaks[0].start()}"
    )


def test_raii_macros_are_defined(runtime_src: str) -> None:
    """The two RAII macros must exist — they are the contract that
    dispatchers depend on."""
    assert "#define TS_METAL_BUF_ACQUIRE(" in runtime_src
    assert "#define TS_METAL_BUF_ACQUIRE_WITH_BYTES(" in runtime_src


def test_metal_buffer_guard_is_release_safe(runtime_src: str) -> None:
    """Guard must call ``metal_buffer_release`` in its destructor
    when ``buf != nil``."""
    # Pull the MetalBufferGuard struct text.
    m = re.search(
        r"struct\s+MetalBufferGuard\s*\{(.*?)\n\};",
        runtime_src, re.DOTALL,
    )
    assert m is not None
    body = m.group(1)
    assert "~MetalBufferGuard()" in body
    assert "metal_buffer_release" in body
    assert "if (buf)" in body, (
        "Guard must guard against nil-buf in its destructor — "
        "metal_buffer_release(ctx, nil, ...) is a programmer error"
    )


def test_native_ebm_ops_promoted_in_primitive_coverage() -> None:
    """Every EBM op with a fused entry in ``_EBM_APPLE_GPU_FUSED`` must
    have a matching registry row at ``status="partial"`` (Decision #25:
    native kernel exists → at least partial, never planned).

    This guard catches the inverse drift the user flagged: registry
    saying ``planned`` while the manifest already ships a fused MSL
    kernel.  The two tables are normative for different audiences
    (manifest = runtime acceptance; registry = compiler contract),
    but their status fields must not contradict each other.
    """
    from tessera.compiler import primitive_coverage as pc
    from tessera.compiler import backend_manifest as bm
    # Manifest → registry name mapping.  Most match 1:1; the two
    # geometric Langevin entries use a ``_step`` suffix in the
    # registry but the same fused kernel in the manifest.
    manifest_to_registry = {
        "ebm_inner_step":         "ebm_inner_step",
        "ebm_langevin_step":      "ebm_langevin_step",
        "ebm_decode_init":        "ebm_decode_init",
        "ebm_self_verify":        "ebm_self_verify",
        "ebm_energy":             "ebm_energy",
        "ebm_partition_exact":    "ebm_partition_exact",
        "ebm_bivector_langevin":  "ebm_bivector_langevin_step",
        "ebm_sphere_langevin":    "ebm_sphere_langevin_step",
    }
    stale: list[str] = []
    for manifest_name, registry_name in manifest_to_registry.items():
        assert manifest_name in bm._EBM_APPLE_GPU_FUSED, (
            f"manifest does not list {manifest_name!r}"
        )
        cov = pc.coverage_for(registry_name)
        assert cov is not None, (
            f"registry has no entry for {registry_name!r}"
        )
        if cov.status == "planned":
            stale.append(
                f"{registry_name}: registry status='planned' but manifest "
                f"has fused entry {manifest_name!r}"
            )
    assert not stale, "stale registry status:\n  " + "\n  ".join(stale)


def test_raii_macros_used_in_dispatchers(runtime_src: str) -> None:
    """Sanity: at least 50 dispatcher call sites use the macros.  If
    this count crashes to zero a regression must have reverted the
    sweep."""
    count = len(re.findall(r"\bTS_METAL_BUF_ACQUIRE(?:_WITH_BYTES)?\s*\(",
                           runtime_src))
    assert count >= 50, (
        f"Only {count} TS_METAL_BUF_ACQUIRE* call sites — the buffer "
        f"pool sweep regressed.  Expected ≥ 50."
    )
