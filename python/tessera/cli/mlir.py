"""Static IR dump command for Tessera Python sources.

The command is intentionally conservative: it does not execute the input
program. It extracts a lightweight module view from Python syntax and emits a
stable textual debug artifact that mirrors the future compiler entry point.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, TextIO


EMIT_LEVELS = ("graph-ir", "schedule-ir", "tile-ir", "target-ir")


@dataclass(frozen=True)
class SourceSymbol:
    kind: str
    name: str
    lineno: int
    decorators: tuple[str, ...] = ()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    source = Path(args.model)
    try:
        text = source.read_text()
        symbols = inspect_source(text)
        rendered = render_ir(
            source=source,
            symbols=symbols,
            emit=args.emit,
            debug=args.debug,
        )
    except OSError as exc:
        parser.error(str(exc))
    except SyntaxError as exc:
        parser.error(f"{source}: syntax error at line {exc.lineno}: {exc.msg}")

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
) -> str:
    header = [
        f"// tessera-mlir emit={emit}",
        f"// source={source}",
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
        lines = header + ['"tessera.schedule.module"() ({']
        for symbol in body:
            lines.append(f'  "tessera.schedule.entry"() {{sym_name = "{symbol.name}", source_line = {symbol.lineno}}}')
        lines.append('}) : () -> ()')
        return "\n".join(lines)
    if emit == "tile-ir":
        lines = header + ['"tessera.tile.module"() ({']
        for symbol in body:
            lines.append(f'  "tessera.tile.entry_stub"() {{sym_name = "{symbol.name}", source_line = {symbol.lineno}}}')
        lines.append('}) : () -> ()')
        return "\n".join(lines)

    lines = header + ['"tessera.target.module"() ({']
    for symbol in body:
        lines.append(f'  "tessera.target.entry_stub"() {{sym_name = "{symbol.name}", source_line = {symbol.lineno}}}')
    lines.append('}) : () -> ()')
    return "\n".join(lines)


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tessera-mlir",
        description="Dump Tessera debug IR from a Python model source without executing it.",
    )
    parser.add_argument("model", help="Python model source to inspect")
    parser.add_argument("--emit", choices=EMIT_LEVELS, default="graph-ir", help="IR level to emit")
    parser.add_argument("--debug", action="store_true", help="Include source locations and debug comments")
    parser.add_argument("-o", "--output", help="Write IR to a file instead of stdout")
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
