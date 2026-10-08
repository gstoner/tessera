"""Actual ELF dependency changes must invalidate compiler metadata and images."""
import hashlib
import shutil
import subprocess
from pathlib import Path
import pytest
from tessera.compiler import rocm_native as native


@pytest.fixture
def elf_tool(tmp_path, monkeypatch):
    if not all(shutil.which(name) for name in ("cc", "ldd", "readelf")):
        pytest.skip("ELF identity proof requires host compiler and loader tools")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    low = tmp_path / "low"
    high = tmp_path / "high"
    low.mkdir()
    high.mkdir()
    source = tmp_path / "layout.c"
    def build(directory, value):
        source.write_text(f"int layout_value(void) {{ return {value}; }}")
        out = directory / "liblayout.so"
        subprocess.run(["cc", "-shared", "-fPIC", str(source), "-o", str(out)],
                       check=True, capture_output=True)
        return out
    library = build(low, 1)
    main = tmp_path / "main.c"
    main.write_text('#include <stdio.h>\nextern int layout_value(void);\n'
                    'int main(void) { printf("version1:%d\\n", layout_value()); }\n')
    tool = tmp_path / "compiler"
    subprocess.run(["cc", str(main), "-L" + str(low), "-llayout",
                    "-Wl,-rpath," + str(high) + ":" + str(low), "-o", str(tool)],
                   check=True, capture_output=True)
    native._LINKED_TOOL_FILES.clear()
    native._VERSION_FINGERPRINTS.clear()
    native._TOOL_DIGESTS.clear()
    yield tool, library, high, build
    native._LINKED_TOOL_FILES.clear()
    native._VERSION_FINGERPRINTS.clear()
    native._TOOL_DIGESTS.clear()


@pytest.mark.parametrize("change", ["replace", "symlink", "higher_priority"])
def test_library_change_without_executable_change(elf_tool, change):
    tool, library, high, build = elf_tool
    executable_hash = hashlib.sha256(tool.read_bytes()).hexdigest()
    version = native._version_fingerprint(tool)
    identity = native._tool_digest(tool)
    if change == "replace":
        replacement = build(high, 2)
        replacement.replace(library)
    elif change == "symlink":
        replacement = build(high, 2)
        library.unlink()
        library.symlink_to(replacement)
    else:
        build(high, 2)
    assert hashlib.sha256(tool.read_bytes()).hexdigest() == executable_hash
    assert native._version_fingerprint(tool) != version
    assert native._tool_digest(tool) != identity


def test_warm_elf_identity_requires_no_subprocess(elf_tool, monkeypatch):
    tool, *_ = elf_tool
    identity = native._tool_digest(tool)
    version = native._version_fingerprint(tool)
    def refused(*args, **kwargs):
        raise AssertionError("warm metadata must not invoke a process")
    monkeypatch.setattr(native.subprocess, "run", refused)
    assert native._tool_digest(tool) == identity
    assert native._version_fingerprint(tool) == version


def test_static_elf_retains_content_identity(tmp_path, monkeypatch):
    source = tmp_path / "static.c"
    source.write_text("int main(void) { return 0; }")
    tool = tmp_path / "static-compiler"
    result = subprocess.run(["cc", "-static", str(source), "-o", str(tool)],
                            capture_output=True, text=True)
    if result.returncode:
        pytest.skip("host static C runtime unavailable")
    native._LINKED_TOOL_FILES.clear()
    expected = hashlib.sha256(tool.read_bytes()).hexdigest()
    assert native._tool_digest(tool) == expected
    def refused(*args, **kwargs):
        raise AssertionError("static warm identity must not launch subprocess")
    monkeypatch.setattr(native.subprocess, "run", refused)
    assert native._tool_digest(tool) == expected
