"""Every Apple backend pass is reachable on its own, not only inside a pipeline.

Each Apple pass declares a `getArgument()` — the name `tessera-opt --<arg>`
would run it by. That string is only real if the pass is also handed to
`registerPass`. It was not: 21 of 28 declared an argument that no binary
accepted, so `--tile-to-apple_gpu` did not exist and a lit fixture could not
isolate one pass from the pipeline around it.

Two guards. The structural one reads the sources and needs no build, so it runs
everywhere. The behavioural one asks a real binary and skips without one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests._support.compiler_tool import registered_passes, tessera_opt_path


APPLE = (Path(__file__).resolve().parents[2]
         / "src/compiler/codegen/Tessera_Apple_Backend")
PASSES_H = APPLE / "include/Tessera/Target/Apple/Passes.h"
PASSES_CPP = APPLE / "lib/Target/Apple/Passes.cpp"
LOWERING = APPLE / "lib/Target/Apple"


def _declared_factories() -> set[str]:
    """`create...Pass` factories Passes.h publishes."""
    return set(re.findall(r"\b(create\w*Pass)\s*\(", PASSES_H.read_text()))


def _registered_factories() -> set[str]:
    """Factories `registerTesseraAppleBackendPipelines` hands to registerPass."""
    body = PASSES_CPP.read_text()
    return set(re.findall(r"registerPass\(\[\]\(\)\s*\{\s*return\s+(create\w*Pass)\(",
                          body))


def _declared_arguments() -> dict[str, str]:
    """`getArgument()` string -> the file that declares it."""
    arguments: dict[str, str] = {}
    for source in sorted(LOWERING.rglob("*.cpp")):
        text = source.read_text()
        for match in re.finditer(
            r"getArgument\(\)\s*const\s*(?:final|override)\s*\{\s*return\s+\"([\w.-]+)\"",
            text,
        ):
            arguments[match.group(1)] = source.name
    return arguments


def test_every_declared_apple_pass_factory_is_registered() -> None:
    declared = _declared_factories()
    assert declared, "Passes.h published no create*Pass factories — check the regex"
    missing = sorted(declared - _registered_factories())
    assert not missing, (
        "Apple pass factories declared in Passes.h but never registered, so "
        "`tessera-opt --<their argument>` does not exist: " + ", ".join(missing)
    )


def test_every_apple_pass_declares_a_distinct_argument() -> None:
    """A duplicate argument silently shadows one pass with another."""
    arguments: dict[str, list[str]] = {}
    for source in sorted(LOWERING.rglob("*.cpp")):
        for match in re.finditer(
            r"getArgument\(\)\s*const\s*(?:final|override)\s*\{\s*return\s+\"([\w.-]+)\"",
            source.read_text(),
        ):
            arguments.setdefault(match.group(1), []).append(source.name)
    duplicates = {arg: files for arg, files in arguments.items() if len(files) > 1}
    assert not duplicates, f"Apple passes share a --argument: {duplicates}"


@pytest.mark.compiler_tool
@pytest.mark.compiler_apple
def test_a_built_driver_accepts_every_apple_pass_argument() -> None:
    """The structural guard proves intent; this proves the binary agrees."""
    tool = tessera_opt_path()
    if tool is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    available = registered_passes(tool)
    if "tessera-lower-to-apple_gpu" not in available:
        pytest.skip(f"{tool} was built without the Apple backend")
    missing = sorted(arg for arg in _declared_arguments() if arg not in available)
    assert not missing, (
        f"{tool} does not accept these Apple pass arguments even though the "
        f"passes declare them: {', '.join(missing)}"
    )
