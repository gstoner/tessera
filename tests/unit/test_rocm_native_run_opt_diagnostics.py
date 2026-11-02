"""Host-free guard: ROCm native packaging refuses a compile that reports errors.

LLVM codegen printed "error: no libcall available for ftanh" and still exited
0, so every gfx1151 soft-cap attention kernel shipped with tanh computing 0.
"""
from __future__ import annotations

import stat
from pathlib import Path

import pytest

from tessera.compiler import rocm_native


def _fake_tool(tmp_path: Path, stderr: str, code: int = 0) -> Path:
    tool = tmp_path / "fake-opt"
    tool.write_text(f"#!/bin/sh\ncat >/dev/null\nprintf 'module {{}}\\n'\nprintf '{stderr}' >&2\nexit {code}\n")
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    return tool


def test_run_opt_refuses_error_diagnostic_on_exit_zero(tmp_path: Path) -> None:
    tool = _fake_tool(tmp_path, "error: no libcall available for ftanh\\n")
    with pytest.raises(RuntimeError, match="no libcall available for ftanh"):
        rocm_native._run_opt(tool, "module {}", "builtin.module()")


def test_run_opt_accepts_clean_compile_and_warnings(tmp_path: Path) -> None:
    tool = _fake_tool(tmp_path, "warning: something benign\\n")
    assert rocm_native._run_opt(tool, "module {}", "builtin.module()").strip() == "module {}"


def test_run_opt_still_refuses_nonzero_exit(tmp_path: Path) -> None:
    tool = _fake_tool(tmp_path, "boom\\n", code=1)
    with pytest.raises(RuntimeError, match="boom"):
        rocm_native._run_opt(tool, "module {}", "builtin.module()")
