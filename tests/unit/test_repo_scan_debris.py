"""Filesystem debris must be visible to a sweep, because git hides it.

`._*` AppleDouble resource forks are gitignored, so a working tree holding 49 of
them reports a clean `git status` while every suffix-glob scanner reads them as
source. That is how a Super-Bear bisect (2026-09-17) compared a debris-laden
tree against a clean worktree and called the difference a code change.
"""
from __future__ import annotations


def test_apple_double_forks_are_found_and_only_them(tmp_path):
    """The finder reports every `._name` outside non-source trees and nothing
    else — including the twin the shared iterator refuses to yield."""
    from tests._support.repo_scan import find_apple_double_forks, iter_repo_files

    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "._a.py").write_bytes(b"\x00\x05\x16\x07Mac OS X\xa3")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "._b.mlir").write_bytes(b"\x00\x05\x16\x07")
    (tmp_path / "build" ).mkdir()
    (tmp_path / "build" / "._ignored.py").write_bytes(b"\x00")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "._ignored").write_bytes(b"\x00")
    forks = [f.relative_to(tmp_path).as_posix() for f in find_apple_double_forks(tmp_path)]
    assert forks == ["._a.py", "sub/._b.mlir"]
    yielded = {f.relative_to(tmp_path).as_posix() for f in iter_repo_files(tmp_path, suffixes={".py", ".mlir"})}
    assert yielded == {"a.py"}
