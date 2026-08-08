from __future__ import annotations

from pathlib import Path

from tests._support import build_artifacts
from tests._support import compiler_tool


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _fake_build(tmp_path: Path, name: str = "chosen") -> tuple[Path, Path]:
    root = tmp_path / name
    tool = root / "tools/tessera-opt/tessera-opt"
    tool.parent.mkdir(parents=True)
    tool.write_text("tool", encoding="utf-8")
    (root / "CMakeCache.txt").write_text("cache", encoding="utf-8")
    return root, tool


def test_explicit_build_directory_is_authoritative(monkeypatch, tmp_path):
    root, tool = _fake_build(tmp_path)
    monkeypatch.setenv("TESSERA_BUILD_DIR", str(root))
    monkeypatch.setenv("TESSERA_OPT", str(tmp_path / "different/tessera-opt"))

    assert build_artifacts.build_roots(defaults=("build",)) == (root,)
    assert build_artifacts.built_artifact("tools/tessera-opt/tessera-opt") == tool


def test_bad_explicit_build_directory_fails_closed(monkeypatch, tmp_path):
    fallback, _ = _fake_build(tmp_path, "fallback")
    monkeypatch.setattr(build_artifacts, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("TESSERA_BUILD_DIR", str(tmp_path / "missing"))

    assert build_artifacts.built_artifact(
        "tools/tessera-opt/tessera-opt", defaults=(fallback.name,)
    ) is None


def test_selected_tessera_opt_infers_its_cmake_root(monkeypatch, tmp_path):
    root, tool = _fake_build(tmp_path)
    sibling = root / "src/backend/lib.so"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("library", encoding="utf-8")
    monkeypatch.delenv("TESSERA_BUILD_DIR", raising=False)
    monkeypatch.setenv("TESSERA_OPT", str(tool))

    assert build_artifacts.build_roots(defaults=("unused",))[0] == root
    assert build_artifacts.built_artifact(
        "src/backend/lib.so", defaults=("unused",)
    ) == sibling


def test_artifact_override_is_authoritative(monkeypatch, tmp_path):
    root, _ = _fake_build(tmp_path)
    built = root / "lib/libx.so"
    built.parent.mkdir()
    built.write_text("built", encoding="utf-8")
    monkeypatch.setenv("TESSERA_BUILD_DIR", str(root))
    monkeypatch.setenv("TESSERA_TEST_LIB", str(tmp_path / "missing.so"))

    assert build_artifacts.built_artifact(
        "lib/libx.so", env=("TESSERA_TEST_LIB",)
    ) is None


def test_compiler_tool_honors_common_build_directory(monkeypatch, tmp_path):
    root, tool = _fake_build(tmp_path)
    monkeypatch.delenv("TESSERA_OPT", raising=False)
    monkeypatch.delenv("TESSERA_OPT_BIN", raising=False)
    monkeypatch.setenv("TESSERA_BUILD_DIR", str(root))

    assert compiler_tool.tessera_opt_candidates() == (tool,)


def test_runtime_honors_common_build_directory(monkeypatch, tmp_path):
    from tessera import runtime

    root, tool = _fake_build(tmp_path)
    gemm = root / (
        "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/"
        "libtessera_rocm_gemm.so"
    )
    gemm.parent.mkdir(parents=True)
    gemm.write_text("library", encoding="utf-8")
    monkeypatch.delenv("TESSERA_OPT", raising=False)
    monkeypatch.delenv("TESSERA_OPT_BIN", raising=False)
    monkeypatch.delenv("TESSERA_ROCM_GEMM_LIB", raising=False)
    monkeypatch.setenv("TESSERA_BUILD_DIR", str(root))

    assert runtime._tessera_opt_path() == tool
    assert runtime._rocm_gemm_lib_path() == gemm


def test_hip_compiler_build_closes_over_spectral_runtime_image():
    cmake = (_REPO_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "add_dependencies(tessera-opt TesseraSpectralHIP)" in cmake
    assert "add_dependencies(tessera-rocm-opt TesseraSpectralHIP)" in cmake
    assert "list(APPEND TESSERA_COMPILER_FOUNDATION_DEPS TesseraSpectralHIP)" in cmake
