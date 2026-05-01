"""Compiler test harness helpers for executable artifact workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from tessera.compiler.jit import JitFn, jit
from tessera.runtime import RuntimeArtifact, launch


@dataclass(frozen=True)
class CompilerHarnessResult:
    compiled: JitFn
    artifact: RuntimeArtifact
    launch_result: dict[str, Any] | None
    diagnostics: tuple[str, ...]


def compile_and_maybe_launch(
    fn: Callable[..., Any],
    *args: Any,
    jit_kwargs: dict[str, Any] | None = None,
    launch_args: Any | None = None,
    run: bool = True,
) -> CompilerHarnessResult:
    """Compile a function, capture its runtime artifact, and optionally launch it."""

    compiled = jit(fn, **(jit_kwargs or {}))
    artifact = compiled.runtime_artifact()
    launch_payload = launch_args if launch_args is not None else args
    launch_result = launch(artifact, launch_payload) if run else None
    return CompilerHarnessResult(
        compiled=compiled,
        artifact=artifact,
        launch_result=launch_result,
        diagnostics=tuple(d.format() for d in compiled.lowering_diagnostics),
    )
