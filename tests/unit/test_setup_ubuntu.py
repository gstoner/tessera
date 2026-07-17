"""Contracts for the Ubuntu bootstrap ordering."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SETUP = ROOT / "scripts/setup_ubuntu.sh"


def test_llvm_repo_prerequisites_are_installed_before_repo_probe() -> None:
    text = SETUP.read_text(encoding="utf-8")
    invocation = text.index("  install_llvm_repo_prerequisites\n")
    repo_probe = text.index("  configure_llvm_repo\n", invocation)
    assert invocation < repo_probe

    function = text[text.index("install_llvm_repo_prerequisites() {"):invocation]
    for package in ("ca-certificates", "wget", "gnupg"):
        assert package in function
    assert '$SUDO rm -f "$list"' in function


def test_no_apt_llvm_setup_requires_existing_probe_tools() -> None:
    text = SETUP.read_text(encoding="utf-8")
    function = text[
        text.index("install_llvm_repo_prerequisites() {"):
        text.index("# Repair/configure the LLVM source")
    ]
    assert "--no-apt requires wget" in function
    assert "--no-apt requires gpg" in function
