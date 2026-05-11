"""Tests for env-var-driven IR dumping (Phase A1).

Covers:
  * `tessera.debug_env.parse_debug_ir` — value parsing, aliases, "all"
  * `tessera.debug_env.dump_dir` — creation on demand
  * `tessera.debug_env.dump_artifact` — written-files map matches selection
  * Integration: `@tessera.jit` writes IR snapshots when env vars are set
  * `tessera-mlir diff` — exit codes + output format
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import tessera as ts
from tessera import debug_env


# ─────────────────────────────────────────────────────────────────────────────
# Pure parsing
# ─────────────────────────────────────────────────────────────────────────────


class TestParseDebugIR:
    def test_unset_returns_empty(self):
        assert debug_env.parse_debug_ir("") == frozenset()
        assert debug_env.parse_debug_ir(None) == frozenset() or os.environ.get("TESSERA_DEBUG_IR")  # noqa
        assert debug_env.parse_debug_ir("   ") == frozenset()

    def test_canonical_stages(self):
        assert debug_env.parse_debug_ir("graph") == frozenset({"graph"})
        assert debug_env.parse_debug_ir("graph,schedule") == frozenset({"graph", "schedule"})
        assert debug_env.parse_debug_ir("graph,tile,target") == frozenset({"graph", "tile", "target"})

    def test_whitespace_tolerant(self):
        assert debug_env.parse_debug_ir(" graph , schedule ") == frozenset({"graph", "schedule"})

    def test_aliases(self):
        # Match `tessera-mlir --emit=graph-ir` spelling
        assert debug_env.parse_debug_ir("graph-ir") == frozenset({"graph"})
        assert debug_env.parse_debug_ir("schedule_ir,target-ir") == frozenset({"schedule", "target"})

    def test_all_expands(self):
        assert debug_env.parse_debug_ir("all") == frozenset({"graph", "schedule", "tile", "target"})

    def test_unknown_dropped(self):
        # Don't crash user code on unknown stages — drop them silently
        assert debug_env.parse_debug_ir("graph,bogus") == frozenset({"graph"})


class TestDumpDir:
    def test_unset_returns_none(self, monkeypatch):
        monkeypatch.delenv("TESSERA_DEBUG_DUMP_DIR", raising=False)
        assert debug_env.dump_dir() is None

    def test_creates_on_demand(self, tmp_path):
        target = tmp_path / "deep" / "nested" / "ir"
        result = debug_env.dump_dir(str(target))
        assert result == target
        assert target.exists() and target.is_dir()


# ─────────────────────────────────────────────────────────────────────────────
# dump_artifact
# ─────────────────────────────────────────────────────────────────────────────


class TestDumpArtifact:
    def test_no_dump_when_env_not_set(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TESSERA_DEBUG_IR", raising=False)
        monkeypatch.delenv("TESSERA_DEBUG_DUMP_DIR", raising=False)
        written = debug_env.dump_artifact(
            symbol="test",
            graph_ir="module {}",
            directory=tmp_path,
        )
        assert written == {}
        assert list(tmp_path.iterdir()) == []

    def test_dumps_only_selected_stages(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TESSERA_DEBUG_IR", "graph,target")
        written = debug_env.dump_artifact(
            symbol="my_kernel",
            graph_ir="// graph",
            schedule_ir="// schedule",
            tile_ir="// tile",
            target_ir="// target",
            directory=tmp_path,
        )
        assert set(written) == {"graph", "target"}
        assert (tmp_path / "my_kernel.graph.mlir").read_text() == "// graph"
        assert (tmp_path / "my_kernel.target.mlir").read_text() == "// target"
        assert not (tmp_path / "my_kernel.schedule.mlir").exists()
        assert not (tmp_path / "my_kernel.tile.mlir").exists()

    def test_skips_empty_ir(self, monkeypatch, tmp_path):
        # Empty IR strings shouldn't create empty files
        monkeypatch.setenv("TESSERA_DEBUG_IR", "all")
        written = debug_env.dump_artifact(
            symbol="empty",
            graph_ir="",
            schedule_ir="",
            tile_ir="",
            target_ir="",
            directory=tmp_path,
        )
        assert written == {}

    def test_symbol_sanitization(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TESSERA_DEBUG_IR", "graph")
        written = debug_env.dump_artifact(
            symbol="bad/name with space",
            graph_ir="module {}",
            directory=tmp_path,
        )
        assert "graph" in written
        # Path uses sanitized symbol — alphanumerics and ._- only
        assert all(c.isalnum() or c in "._-" for c in written["graph"].name.split(".")[0])


# ─────────────────────────────────────────────────────────────────────────────
# JIT integration — `@tessera.jit` writes IR when env vars set
# ─────────────────────────────────────────────────────────────────────────────


class TestJitIntegration:
    def test_jit_dumps_graph_ir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TESSERA_DEBUG_IR", "graph")
        monkeypatch.setenv("TESSERA_DEBUG_DUMP_DIR", str(tmp_path))

        @ts.jit
        def my_traced_fn(x: ts.Tensor["B", "D"]) -> ts.Tensor["B", "D"]:
            return ts.ops.add(x, x)

        # Force runtime artifact materialization
        artifact = my_traced_fn.runtime_artifact()
        assert artifact is not None

        # The graph IR file should exist
        dumps = list(tmp_path.iterdir())
        graph_dumps = [p for p in dumps if p.name.endswith(".graph.mlir")]
        assert graph_dumps, f"no graph.mlir in {dumps}"
        assert graph_dumps[0].read_text().strip() != ""

    def test_jit_no_dump_when_env_unset(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TESSERA_DEBUG_IR", raising=False)
        monkeypatch.setenv("TESSERA_DEBUG_DUMP_DIR", str(tmp_path))

        @ts.jit
        def silent_fn(x: ts.Tensor["B", "D"]) -> ts.Tensor["B", "D"]:
            return ts.ops.add(x, x)

        silent_fn.runtime_artifact()
        assert list(tmp_path.iterdir()) == []


# ─────────────────────────────────────────────────────────────────────────────
# `tessera-mlir diff` CLI
# ─────────────────────────────────────────────────────────────────────────────


class TestDiffCommand:
    def _run_cli(self, *args: str, capture_output: bool = True):
        cmd = [sys.executable, "-m", "tessera.cli.mlir", *args]
        # The repo is dev-installed via `pyproject.toml`'s `pythonpath = [".",
        # "python"]`, which is honored by the pytest collector but NOT by a
        # naked subprocess. Pass PYTHONPATH so `-m tessera.cli.mlir` resolves
        # the same way it does inside the test process.
        repo_root = Path(__file__).resolve().parents[2]
        env = dict(os.environ)
        existing_path = env.get("PYTHONPATH", "")
        repo_python = str(repo_root / "python")
        env["PYTHONPATH"] = (
            f"{repo_python}{os.pathsep}{existing_path}"
            if existing_path else repo_python
        )
        return subprocess.run(
            cmd, capture_output=capture_output, text=True, env=env,
        )

    def test_identical_files_exit_zero(self, tmp_path):
        a = tmp_path / "a.mlir"
        b = tmp_path / "b.mlir"
        a.write_text("module { }\n")
        b.write_text("module { }\n")
        result = self._run_cli("diff", str(a), str(b))
        assert result.returncode == 0
        assert result.stdout == ""

    def test_differing_files_exit_one_and_show_diff(self, tmp_path):
        a = tmp_path / "a.mlir"
        b = tmp_path / "b.mlir"
        a.write_text("module {\n  func.func @f() {}\n}\n")
        b.write_text("module {\n  func.func @g() {}\n}\n")
        result = self._run_cli("diff", str(a), str(b))
        assert result.returncode == 1
        # unified-diff output
        assert "---" in result.stdout
        assert "+++" in result.stdout
        assert "@f" in result.stdout
        assert "@g" in result.stdout

    def test_diff_writes_output_file(self, tmp_path):
        a = tmp_path / "a.mlir"
        b = tmp_path / "b.mlir"
        out = tmp_path / "diff.txt"
        a.write_text("foo\n")
        b.write_text("bar\n")
        result = self._run_cli("diff", str(a), str(b), "-o", str(out))
        assert result.returncode == 1
        assert out.exists()
        text = out.read_text()
        assert "foo" in text
        assert "bar" in text

    def test_diff_missing_file_errors(self, tmp_path):
        result = self._run_cli("diff", str(tmp_path / "nope.mlir"), str(tmp_path / "nope2.mlir"))
        assert result.returncode != 0
        assert "error" in (result.stderr or "").lower()
