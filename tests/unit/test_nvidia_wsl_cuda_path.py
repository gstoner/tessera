"""WSL CUDA path discovery must not rely on an interactive shell profile."""
from __future__ import annotations

from pathlib import Path

from tests._support.environment import ensure_cuda_bin_on_path, nvidia_cuda_tool


def test_nvidia_cuda_tool_discovers_canonical_wsl_toolkit(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    tool = nvidia_cuda_tool("nvcc")
    if Path("/usr/local/cuda/bin/nvcc").is_file():
        assert tool == Path("/usr/local/cuda/bin/nvcc")
        assert str(ensure_cuda_bin_on_path()) in __import__("os").environ["PATH"]
    else:
        assert tool is None
