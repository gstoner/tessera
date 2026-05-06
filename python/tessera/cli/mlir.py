"""Static IR dump command for Tessera Python sources.

The command is intentionally conservative: it does not execute the input
program. It extracts a lightweight module view from Python syntax and emits a
stable textual debug artifact that mirrors the future compiler entry point.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


EMIT_LEVELS = ("graph-ir", "schedule-ir", "tile-ir", "target-ir", "metadata", "diagnostics", "trace", "graphviz", "all")
IR_LEVELS = ("graph-ir", "schedule-ir", "tile-ir", "target-ir")


@dataclass(frozen=True)
class SourceSymbol:
    kind: str
    name: str
    lineno: int
    decorators: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "name": self.name,
            "lineno": self.lineno,
            "decorators": list(self.decorators),
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    source = Path(args.model)
    try:
        text = source.read_text()
        symbols = inspect_source(text)
        artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else None
        if args.mode == "compile_artifact" and args.symbol:
            rendered = render_compile_artifact_output(
                source=source,
                symbol=args.symbol,
                emit=args.emit,
                target=args.target,
                artifacts_dir=artifacts_dir,
            )
        else:
            rendered = render_output(
                source=source,
                symbols=symbols,
                emit=args.emit,
                debug=args.debug,
                mode=args.mode,
                target=args.target,
                artifacts_dir=artifacts_dir,
            )
    except OSError as exc:
        parser.error(str(exc))
    except SyntaxError as exc:
        parser.error(f"{source}: syntax error at line {exc.lineno}: {exc.msg}")
    except ValueError as exc:
        parser.error(str(exc))

    if args.output:
        Path(args.output).write_text(rendered + "\n")
    else:
        sys.stdout.write(rendered + "\n")
    return 0


def inspect_source(text: str) -> list[SourceSymbol]:
    """Return top-level functions/classes without executing user code."""

    module = ast.parse(text)
    symbols: list[SourceSymbol] = []
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "func"
        elif isinstance(node, ast.ClassDef):
            kind = "class"
        else:
            continue
        symbols.append(
            SourceSymbol(
                kind=kind,
                name=node.name,
                lineno=node.lineno,
                decorators=tuple(_decorator_name(d) for d in getattr(node, "decorator_list", ())),
            )
        )
    return symbols


def render_ir(
    *,
    source: Path,
    symbols: Iterable[SourceSymbol],
    emit: str,
    debug: bool,
    mode: str = "source_inspection",
    target: str = "cpu",
) -> str:
    header = [
        f"// tessera-mlir emit={emit}",
        f"// source={source}",
        f"// mode={mode}",
        f"// target={target}",
    ]
    if debug:
        header.append("// debug=true")

    body = list(symbols)
    if emit == "graph-ir":
        lines = header + ['module @tessera_debug {']
        lines.extend(_render_graph_symbols(body, debug=debug))
        lines.append("}")
        return "\n".join(lines)
    if emit == "schedule-ir":
        lines = header + [f'"tessera.schedule.module"() ({{']
        for symbol in body:
            lines.append(f'  "tessera.schedule.entry"() {{sym_name = "{symbol.name}", source_line = {symbol.lineno}, target = "{target}"}}')
        lines.append('}) : () -> ()')
        return "\n".join(lines)
    if emit == "tile-ir":
        lines = header + ['"tessera.tile.module"() ({']
        for symbol in body:
            lines.append(f'  "tessera.tile.entry_stub"() {{sym_name = "{symbol.name}", source_line = {symbol.lineno}}}')
        lines.append('}) : () -> ()')
        return "\n".join(lines)

    lines = header + [f'"tessera.target.module"() ({{']
    for symbol in body:
        lines.append(f'  "tessera.target.entry_stub"() {{sym_name = "{symbol.name}", source_line = {symbol.lineno}, target = "{target}"}}')
    lines.append('}) : () -> ()')
    return "\n".join(lines)


def render_output(
    *,
    source: Path,
    symbols: Iterable[SourceSymbol],
    emit: str,
    debug: bool,
    mode: str = "source_inspection",
    target: str = "cpu",
    artifacts_dir: Path | None = None,
) -> str:
    body = list(symbols)
    if emit in IR_LEVELS:
        return render_ir(source=source, symbols=body, emit=emit, debug=debug, mode=mode, target=target)
    if emit == "metadata":
        return json.dumps(_metadata_payload(source=source, symbols=body, mode=mode, target=target), indent=2, sort_keys=True)
    if emit == "diagnostics":
        return json.dumps(_diagnostics_payload(source=source, symbols=body, mode=mode, target=target), indent=2, sort_keys=True)
    if emit == "trace":
        return json.dumps(_trace_payload(source=source, symbols=body, mode=mode, target=target), indent=2, sort_keys=True)
    if emit == "graphviz":
        return _graphviz_payload(symbols=body)
    if emit == "all":
        outputs = {
            level: render_ir(source=source, symbols=body, emit=level, debug=debug, mode=mode, target=target)
            for level in IR_LEVELS
        }
        outputs["metadata"] = json.dumps(_metadata_payload(source=source, symbols=body, mode=mode, target=target), indent=2, sort_keys=True)
        outputs["diagnostics"] = json.dumps(_diagnostics_payload(source=source, symbols=body, mode=mode, target=target), indent=2, sort_keys=True)
        outputs["trace"] = json.dumps(_trace_payload(source=source, symbols=body, mode=mode, target=target), indent=2, sort_keys=True)
        outputs["graphviz"] = _graphviz_payload(symbols=body)
        if artifacts_dir is not None:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            for name, text in outputs.items():
                suffix = ".mlir" if name in IR_LEVELS else (".dot" if name == "graphviz" else ".json")
                (artifacts_dir / f"{name}{suffix}").write_text(text + "\n", encoding="utf-8")
        return json.dumps(
            {
                "schema": "tessera.mlir.debug_bundle.v1",
                "mode": mode,
                "target": target,
                "source": str(source),
                "artifacts": sorted(outputs),
                "artifacts_dir": str(artifacts_dir) if artifacts_dir is not None else None,
            },
            indent=2,
            sort_keys=True,
        )
    raise ValueError(f"unsupported emit level {emit!r}")


def render_compile_artifact_output(
    *,
    source: Path,
    symbol: str,
    emit: str,
    target: str = "cpu",
    artifacts_dir: Path | None = None,
) -> str:
    obj = _load_symbol(source, symbol)
    if not hasattr(obj, "runtime_artifact") or not callable(obj.runtime_artifact):
        raise ValueError(f"{symbol!r} is not a Tessera JIT symbol with runtime_artifact()")
    artifact = obj.runtime_artifact()
    outputs = _compile_artifact_outputs(artifact, obj=obj, source=source, symbol=symbol, target=target)
    if emit == "graph-ir":
        return outputs["graph-ir"]
    if emit == "schedule-ir":
        return outputs["schedule-ir"]
    if emit == "tile-ir":
        return outputs["tile-ir"]
    if emit == "target-ir":
        return outputs["target-ir"]
    if emit == "metadata":
        return outputs["metadata"]
    if emit == "diagnostics":
        return outputs["diagnostics"]
    if emit == "trace":
        return outputs["trace"]
    if emit == "graphviz":
        return outputs["graphviz"]
    if emit == "all":
        if artifacts_dir is not None:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            for name, text in outputs.items():
                suffix = ".mlir" if name in IR_LEVELS else (".dot" if name == "graphviz" else ".json")
                (artifacts_dir / f"{name}{suffix}").write_text(text + "\n", encoding="utf-8")
        return json.dumps(
            {
                "schema": "tessera.mlir.debug_bundle.v1",
                "mode": "compile_artifact",
                "target": target,
                "source": str(source),
                "symbol": symbol,
                "artifacts": sorted(outputs),
                "artifacts_dir": str(artifacts_dir) if artifacts_dir is not None else None,
            },
            indent=2,
            sort_keys=True,
        )
    raise ValueError(f"unsupported emit level {emit!r}")


def _compile_artifact_outputs(artifact, *, obj, source: Path, symbol: str, target: str) -> dict[str, str]:
    from tessera import debug as tessera_debug

    metadata = artifact.metadata or {}
    diagnostics = metadata.get("diagnostics", [])
    trace = list(obj.lowering_trace()) if hasattr(obj, "lowering_trace") and callable(obj.lowering_trace) else []
    return {
        "graph-ir": artifact.graph_ir,
        "schedule-ir": artifact.schedule_ir,
        "tile-ir": artifact.tile_ir,
        "target-ir": artifact.target_ir,
        "metadata": json.dumps({
            "schema": "tessera.mlir.metadata.v1",
            "mode": "compile_artifact",
            "target": metadata.get("target", target),
            "source": str(source),
            "symbol": symbol,
            "artifact_hash": artifact.artifact_hash,
            "artifact_metadata": metadata,
        }, indent=2, sort_keys=True),
        "diagnostics": json.dumps({
            "schema": "tessera.mlir.diagnostics.v1",
            "mode": "compile_artifact",
            "target": metadata.get("target", target),
            "diagnostics": diagnostics,
        }, indent=2, sort_keys=True),
        "trace": json.dumps({"traceEvents": [
            {
                "name": event.get("pass_name", "compiler.pass"),
                "cat": "tessera.compiler",
                "ph": "X",
                "pid": 1,
                "tid": index,
                "ts": 0,
                "dur": event.get("elapsed_ms", 0),
                "args": event,
            }
            for index, event in enumerate(trace)
        ]}, indent=2, sort_keys=True),
        "graphviz": tessera_debug.export_graphviz(artifact.graph_ir),
    }


def _load_symbol(source: Path, symbol: str):
    module_name = f"_tessera_mlir_debug_{source.stem}_{abs(hash(source))}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, symbol):
        raise ValueError(f"{source} has no symbol {symbol!r}")
    return getattr(module, symbol)


def _render_graph_symbols(symbols: Iterable[SourceSymbol], *, debug: bool) -> list[str]:
    lines: list[str] = []
    for symbol in symbols:
        attr_parts = [f"source_line = {symbol.lineno}"]
        if symbol.decorators:
            quoted = ", ".join(f'"{d}"' for d in symbol.decorators)
            attr_parts.append(f"decorators = [{quoted}]")
        attrs = ", ".join(attr_parts)
        if symbol.kind == "class":
            lines.append(f'  "tessera.graph.module"() {{sym_name = "{symbol.name}", {attrs}}} : () -> ()')
        else:
            lines.append(f'  func.func @{symbol.name}() attributes {{{attrs}}}')
        if debug:
            lines.append(f'  // debug: discovered {symbol.kind} {symbol.name} at line {symbol.lineno}')
    if not lines:
        lines.append('  // no top-level Tessera symbols discovered')
    return lines


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _decorator_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return ast.unparse(node) if hasattr(ast, "unparse") else type(node).__name__


def _metadata_payload(*, source: Path, symbols: Sequence[SourceSymbol], mode: str, target: str) -> dict[str, object]:
    return {
        "schema": "tessera.mlir.metadata.v1",
        "mode": mode,
        "target": target,
        "source": str(source),
        "symbol_count": len(symbols),
        "symbols": [symbol.to_dict() for symbol in symbols],
        "capabilities": {
            "source_inspection": True,
            "compile_artifact": mode == "compile_artifact",
            "executes_model": False,
        },
    }


def _diagnostics_payload(*, source: Path, symbols: Sequence[SourceSymbol], mode: str, target: str) -> dict[str, object]:
    diagnostics = []
    if mode == "compile_artifact":
        diagnostics.append({
            "severity": "warning",
            "code": "W_COMPILE_ARTIFACT_STATIC_ONLY",
            "message": "compile_artifact mode is reserved for verified compiler bundles; this command did not execute the model source",
            "source": str(source),
        })
    if not symbols:
        diagnostics.append({
            "severity": "info",
            "code": "I_NO_TOP_LEVEL_SYMBOLS",
            "message": "no top-level functions or classes discovered",
            "source": str(source),
        })
    return {
        "schema": "tessera.mlir.diagnostics.v1",
        "mode": mode,
        "target": target,
        "diagnostics": diagnostics,
    }


def _trace_payload(*, source: Path, symbols: Sequence[SourceSymbol], mode: str, target: str) -> dict[str, object]:
    return {
        "traceEvents": [
            {
                "name": "source.inspect",
                "cat": "tessera.mlir",
                "ph": "X",
                "pid": 1,
                "tid": 0,
                "ts": 0,
                "dur": 0,
                "args": {
                    "schema": "tessera.mlir.trace.v1",
                    "mode": mode,
                    "target": target,
                    "source": str(source),
                    "symbol_count": len(symbols),
                },
            }
        ]
    }


def _graphviz_payload(*, symbols: Sequence[SourceSymbol]) -> str:
    lines = ["digraph tessera_source {"]
    if not symbols:
        lines.append('  empty [label="no top-level Tessera symbols discovered"];')
    for idx, symbol in enumerate(symbols):
        label = f"{symbol.kind} {symbol.name}\\nline {symbol.lineno}"
        lines.append(f'  n{idx} [label="{label}"];')
    lines.append("}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tessera-mlir",
        description="Dump Tessera debug IR from a Python model source without executing it.",
    )
    parser.add_argument("model", help="Python model source to inspect")
    parser.add_argument("--emit", choices=EMIT_LEVELS, default="graph-ir", help="IR level to emit")
    parser.add_argument("--mode", choices=("source_inspection", "compile_artifact"), default="source_inspection", help="Debug mode; source_inspection never executes the model")
    parser.add_argument("--symbol", help="Top-level @tessera.jit symbol for --mode=compile_artifact")
    parser.add_argument("--target", default="cpu", help="Target label to attach to debug artifacts")
    parser.add_argument("--debug", action="store_true", help="Include source locations and debug comments")
    parser.add_argument("--artifacts-dir", help="Directory for --emit=all artifact files")
    parser.add_argument("-o", "--output", help="Write IR to a file instead of stdout")
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
