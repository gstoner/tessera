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
    # The only allowed raw calls are inside ``metal_buffer_acquire``
    # itself — find the function's brace range and exclude calls
    # inside it.
    primitive_re = re.compile(
        r"static\s+id<MTLBuffer>\s+metal_buffer_acquire\s*\(",
    )
    pm = primitive_re.search(src)
    assert pm is not None, "metal_buffer_acquire primitive missing"
    brace = src.find("{", pm.end())
    depth = 1; pos = brace + 1
    while pos < len(src) and depth > 0:
        if src[pos] == "{": depth += 1
        elif src[pos] == "}": depth -= 1
        pos += 1
    primitive_end = pos
    offending = [
        m for m in raw
        if not (pm.start() <= m.start() <= primitive_end)
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
