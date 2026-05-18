"""Normative textual Tessera DSL parser and Graph IR lowerer.

The parser accepts the BNF in ``docs/spec/LANGUAGE_AND_IR_SPEC.md`` and lowers
the executable subset into the Python Graph IR object model. Constructs that are
not native Graph IR ops, such as control-flow regions and schedule statements,
are preserved as explicit marker ops so downstream tooling can inspect the full
source shape instead of losing syntax at the frontend boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..graph_ir import (
    GraphIRConstant,
    GraphIRFunction,
    GraphIRMesh,
    GraphIRModule,
    GraphIRTypeAlias,
    IRArg,
    IROp,
    SourceSpan,
    TENSOR_OPAQUE,
    tensor_ir_type,
)
from ..op_catalog import get_op_spec


class FrontendSyntaxError(SyntaxError):
    def __init__(self, message: str, span: Optional[SourceSpan] = None) -> None:
        self.span = span
        suffix = f" at {span.format()}" if span else ""
        super().__init__(f"{message}{suffix}")


class FrontendSemanticError(Exception):
    def __init__(self, message: str, span: Optional[SourceSpan] = None) -> None:
        self.span = span
        suffix = f" at {span.format()}" if span else ""
        super().__init__(f"{message}{suffix}")


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    line: int
    col: int


TOKEN_SPEC = [
    ("WS", r"[ \t\r]+"),
    ("COMMENT", r"//[^\n]*|/\*.*?\*/"),
    ("NEWLINE", r"\n"),
    ("ARROW", r"->"),
    ("NUMBER", r"-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"),
    ("STRING", r'"(?:\\.|[^"\\])*"'),
    ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("SYMBOL", r"[{}()\[\]<>,;:=.@x?+\-*/|!]"),
]
MASTER = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in TOKEN_SPEC), re.DOTALL)


def _lex(source: str) -> list[Token]:
    tokens = []
    line = 1
    col = 1
    pos = 0
    while pos < len(source):
        match = MASTER.match(source, pos)
        if match is None:
            raise FrontendSyntaxError(f"unexpected character {source[pos]!r}", SourceSpan(line, col))
        kind = match.lastgroup or ""
        text = match.group()
        if kind == "NEWLINE":
            line += 1
            col = 1
        elif kind in {"WS", "COMMENT"}:
            if "\n" in text:
                line += text.count("\n")
                col = len(text.rsplit("\n", 1)[-1]) + 1
            else:
                col += len(text)
        else:
            tokens.append(Token(kind, text, line, col))
            col += len(text)
        pos = match.end()
    return tokens


@dataclass
class Program:
    modules: list["Module"]


@dataclass
class Module:
    name: str
    funcs: list["Function"]
    meshes: list["Mesh"] = field(default_factory=list)
    type_decls: list["TypeDecl"] = field(default_factory=list)
    const_decls: list["ConstDecl"] = field(default_factory=list)


@dataclass
class Param:
    name: str
    type_expr: Optional[str] = None
    default: Optional["Expr"] = None
    span: Optional[SourceSpan] = None


@dataclass
class Function:
    name: str
    params: list[Param]
    return_type: Optional[str]
    body: list["Stmt"]
    attrs: dict[str, Any] = field(default_factory=dict)
    kind: str = "func"
    span: Optional[SourceSpan] = None


@dataclass
class Mesh:
    name: str
    attrs: dict[str, Any]
    span: Optional[SourceSpan] = None


@dataclass
class TypeDecl:
    name: str
    type_expr: str
    span: Optional[SourceSpan] = None


@dataclass
class ConstDecl:
    name: str
    type_expr: str
    expr: "Expr"
    span: Optional[SourceSpan] = None


@dataclass
class LValue:
    name: str
    slices: list[Optional["Expr"]] = field(default_factory=list)
    span: Optional[SourceSpan] = None


class Stmt:
    span: Optional[SourceSpan] = None


@dataclass
class Let(Stmt):
    name: str
    type_expr: Optional[str]
    expr: Optional["Expr"] = None
    span: Optional[SourceSpan] = None


@dataclass
class Assign(Stmt):
    target: LValue
    expr: "Expr"
    span: Optional[SourceSpan] = None


@dataclass
class Return(Stmt):
    values: list["Expr"]
    span: Optional[SourceSpan] = None


@dataclass
class OpStmt(Stmt):
    expr: "Expr"
    span: Optional[SourceSpan] = None


@dataclass
class Control(Stmt):
    kind: str
    body: list[Stmt]
    else_body: list[Stmt] = field(default_factory=list)
    span: Optional[SourceSpan] = None


class Expr:
    span: Optional[SourceSpan] = None


@dataclass
class Name(Expr):
    value: str
    span: Optional[SourceSpan] = None


@dataclass
class Literal(Expr):
    value: Any
    span: Optional[SourceSpan] = None


@dataclass
class Call(Expr):
    name: str
    args: list[Expr]
    attrs: dict[str, Any] = field(default_factory=dict)
    span: Optional[SourceSpan] = None


@dataclass
class Binary(Expr):
    lhs: Expr
    op: str
    rhs: Expr
    span: Optional[SourceSpan] = None


@dataclass
class TensorExpr(Expr):
    shape: tuple[str, ...]
    dtype: str
    layout: Optional[str] = None
    span: Optional[SourceSpan] = None


class _Parser:
    def __init__(self, source: str):
        self.tokens = _lex(source)
        self.i = 0

    def peek(self, offset: int = 0) -> Optional[Token]:
        index = self.i + offset
        return self.tokens[index] if index < len(self.tokens) else None

    def accept(self, text: str) -> Optional[Token]:
        token = self.peek()
        if token and token.text == text:
            self.i += 1
            return token
        return None

    def eat_text(self, text: str) -> Token:
        token = self.peek()
        if not token or token.text != text:
            got = token.text if token else "EOF"
            raise FrontendSyntaxError(f"expected {text!r}, got {got!r}", self._span(token))
        self.i += 1
        return token

    def eat_ident(self) -> str:
        token = self.peek()
        if not token or token.kind != "IDENT":
            got = token.text if token else "EOF"
            raise FrontendSyntaxError(f"expected identifier, got {got!r}", self._span(token))
        self.i += 1
        return token.text

    def _span(self, token: Optional[Token]) -> Optional[SourceSpan]:
        if token is None:
            return None
        return SourceSpan(token.line, token.col)

    def parse(self) -> Program:
        modules = []
        while self.peek() is not None:
            modules.append(self.parse_module())
        if not modules:
            raise FrontendSyntaxError("expected at least one module")
        return Program(modules)

    def parse_module(self) -> Module:
        self.eat_keyword("module")
        name = self.eat_ident()
        self.eat_text("{")
        funcs = []
        meshes = []
        type_decls = []
        const_decls = []
        while self.peek() and self.peek().text != "}":
            token = self.peek()
            if token and token.text == "mesh":
                meshes.append(self.parse_mesh())
            elif token and token.text in {"func", "kernel"}:
                funcs.append(self.parse_func())
            elif token and token.text == "type":
                type_decls.append(self.parse_type_decl())
            elif token and token.text == "let":
                const_decls.append(self.parse_const_decl())
            else:
                raise FrontendSyntaxError(f"unsupported module declaration {token.text!r}", self._span(token))
        self.eat_text("}")
        return Module(name=name, funcs=funcs, meshes=meshes, type_decls=type_decls, const_decls=const_decls)

    def parse_func(self) -> Function:
        token = self.peek()
        kind = token.text if token else "func"
        if kind not in {"func", "kernel"}:
            raise FrontendSyntaxError("expected function or kernel declaration", self._span(token))
        self.i += 1
        name = self.eat_ident()
        self.eat_text("(")
        params = []
        if self.peek() and self.peek().text != ")":
            params.append(self.parse_param())
            while self.accept(","):
                params.append(self.parse_param())
        self.eat_text(")")
        return_type = None
        if self.peek() and self.peek().kind == "ARROW":
            self.i += 1
            return_type = self.parse_type_until({"{", "@"})
        attrs = self.parse_attr_block() if self.peek() and self.peek().text == "@" else {}
        self.eat_text("{")
        body = []
        while self.peek() and self.peek().text != "}":
            body.append(self.parse_stmt())
        self.eat_text("}")
        return Function(name=name, params=params, return_type=return_type, body=body, attrs=attrs, kind=kind, span=self._span(token))

    def parse_mesh(self) -> Mesh:
        token = self.peek()
        self.eat_keyword("mesh")
        name = self.eat_ident()
        self.eat_text("=")
        self.eat_keyword("mesh")
        self.eat_text("<")
        attrs = self.parse_mesh_attrs()
        self.eat_text(">")
        self.eat_text(";")
        return Mesh(name, attrs, self._span(token))

    def parse_mesh_attrs(self) -> dict[str, Any]:
        attrs: dict[str, Any] = {}
        while self.peek() and self.peek().text != ">":
            key = self.eat_ident()
            self.eat_text("=")
            attrs[key] = self.parse_attr_value()
            self.accept(",")
        return attrs

    def parse_type_decl(self) -> TypeDecl:
        token = self.peek()
        self.eat_keyword("type")
        name = self.eat_ident()
        self.eat_text("=")
        type_expr = self.parse_type_until({";"})
        self.eat_text(";")
        return TypeDecl(name, type_expr, self._span(token))

    def parse_const_decl(self) -> ConstDecl:
        token = self.peek()
        self.eat_keyword("let")
        name = self.eat_ident()
        self.eat_text(":")
        type_expr = self.parse_type_until({"="})
        self.eat_text("=")
        expr = self.parse_expr()
        self.eat_text(";")
        return ConstDecl(name, type_expr, expr, self._span(token))

    def parse_param(self) -> Param:
        token = self.peek()
        name = self.eat_ident()
        type_expr = None
        if self.accept(":"):
            type_expr = self.parse_type_until({",", ")", "="})
        default = None
        if self.accept("="):
            default = self.parse_expr()
        return Param(name=name, type_expr=type_expr, default=default, span=self._span(token))

    def parse_stmt(self) -> Stmt:
        token = self.peek()
        if token is None:
            raise FrontendSyntaxError("unexpected EOF in statement")
        if token.text == "let":
            self.i += 1
            name = self.eat_ident()
            type_expr = None
            if self.accept(":"):
                type_expr = self.parse_type_until({"=", ";"})
            expr = None
            if self.accept("="):
                expr = self.parse_expr()
            self.eat_text(";")
            return Let(name, type_expr, expr, self._span(token))
        if token.text == "return":
            self.i += 1
            values = []
            if self.peek() and self.peek().text != ";":
                values = [self.parse_expr()]
                while self.accept(","):
                    values.append(self.parse_expr())
            self.eat_text(";")
            return Return(values, self._span(token))
        if token.text in {"if", "for", "while"}:
            return self.parse_control()
        if token.text in {"schedule", "dist", "barrier", "assert"}:
            expr = self.parse_expr()
            self.eat_text(";")
            return OpStmt(expr, self._span(token))
        target = self.parse_lvalue()
        self.eat_text("=")
        expr = self.parse_expr()
        self.eat_text(";")
        return Assign(target, expr, self._span(token))

    def parse_lvalue(self) -> LValue:
        token = self.peek()
        name = self.eat_ident()
        slices: list[Optional[Expr]] = []
        if self.accept("["):
            if self.peek() and self.peek().text != "]":
                slices.append(self.parse_optional_slice_expr())
                while self.accept(","):
                    slices.append(self.parse_optional_slice_expr())
            self.eat_text("]")
        return LValue(name, slices, self._span(token))

    def parse_optional_slice_expr(self) -> Optional[Expr]:
        if self.peek() and self.peek().text == ":":
            self.i += 1
            return None
        expr = self.parse_expr()
        if self.peek() and self.peek().text == ":":
            # The current Graph IR object model only needs to know that this is
            # a sliced lvalue. Consume full range syntax to satisfy the grammar.
            while self.peek() and self.peek().text not in {",", "]"}:
                self.i += 1
        return expr

    def parse_control(self) -> Control:
        token = self.peek()
        kind = token.text if token else ""
        self.i += 1
        if kind in {"if", "while"}:
            self.eat_text("(")
            self.parse_expr()
            self.eat_text(")")
        elif kind == "for":
            self.eat_text("(")
            self.eat_ident()
            if self.accept(","):
                self.eat_ident()
            self.eat_keyword("in")
            self.parse_range_expr()
            self.eat_text(")")
        body = self.parse_block()
        else_body = []
        if kind == "if" and self.peek() and self.peek().text == "else":
            self.i += 1
            else_body = self.parse_block()
        return Control(kind, body, else_body, self._span(token))

    def parse_range_expr(self) -> None:
        self.parse_expr()
        self.eat_text(":")
        self.parse_expr()
        if self.accept(":"):
            self.parse_expr()

    def parse_block(self) -> list[Stmt]:
        self.eat_text("{")
        body = []
        while self.peek() and self.peek().text != "}":
            body.append(self.parse_stmt())
        self.eat_text("}")
        return body

    def parse_expr(self) -> Expr:
        expr = self.parse_primary()
        while self.peek() and self.peek().text in {"+", "-", "*", "/"}:
            op = self.peek()
            self.i += 1
            rhs = self.parse_primary()
            expr = Binary(expr, op.text, rhs, getattr(expr, "span", self._span(op)))
        return expr

    def parse_primary(self) -> Expr:
        token = self.peek()
        if token is None:
            raise FrontendSyntaxError("unexpected EOF in expression")
        if token.kind == "NUMBER":
            self.i += 1
            return Literal(float(token.text) if "." in token.text else int(token.text), self._span(token))
        if token.kind == "STRING":
            self.i += 1
            return Literal(bytes(token.text[1:-1], "utf-8").decode("unicode_escape"), self._span(token))
        if token.text in {"true", "false"}:
            self.i += 1
            return Literal(token.text == "true", self._span(token))
        if token.text == "none":
            self.i += 1
            return Literal(None, self._span(token))
        if token.text == "(":
            self.i += 1
            expr = self.parse_expr()
            self.eat_text(")")
            return expr
        if token.text == "tensor" and self.peek(1) and self.peek(1).text == "<":
            return self.parse_tensor_expr()
        name = self.parse_qualified_name()
        if self.accept("("):
            args = []
            if self.peek() and self.peek().text != ")":
                args.append(self.parse_expr())
                while self.accept(","):
                    args.append(self.parse_expr())
            self.eat_text(")")
            attrs = self.parse_attr_block() if self.peek() and self.peek().text == "@" else {}
            return Call(name, args, attrs, self._span(token))
        return Name(name, self._span(token))

    def parse_tensor_expr(self) -> TensorExpr:
        token = self.peek()
        self.eat_keyword("tensor")
        self.eat_text("<")
        shape: list[str] = []
        while self.peek() and self.peek().text not in {";", ">"}:
            dim = self.peek()
            if dim.text == "x":
                self.i += 1
                continue
            shape.append(dim.text)
            self.i += 1
        if not shape:
            raise FrontendSyntaxError("tensor expression requires shape and dtype", self._span(token))
        dtype = shape.pop()
        layout = None
        if self.accept(";"):
            self.eat_keyword("layout")
            self.eat_text("=")
            layout = self.parse_layout_spec()
        self.eat_text(">")
        return TensorExpr(tuple(shape), dtype, layout, self._span(token))

    def parse_layout_spec(self) -> str:
        name = self.eat_ident()
        if self.accept("("):
            depth = 1
            parts = [name, "("]
            while self.peek() and depth:
                token = self.peek()
                self.i += 1
                parts.append(token.text)
                if token.text == "(":
                    depth += 1
                elif token.text == ")":
                    depth -= 1
            return "".join(parts)
        return name

    def parse_qualified_name(self) -> str:
        parts = [self.eat_ident()]
        while self.accept("."):
            parts.append(self.eat_ident())
        return ".".join(parts)

    def parse_attr_block(self) -> dict[str, Any]:
        self.eat_text("@")
        self.eat_text("{")
        attrs = {}
        while self.peek() and self.peek().text != "}":
            key = self.eat_ident()
            self.eat_text("=")
            attrs[key] = self.parse_attr_value()
            self.accept(",")
        self.eat_text("}")
        return attrs

    def parse_attr_value(self) -> Any:
        token = self.peek()
        if token is None:
            raise FrontendSyntaxError("unexpected EOF in attribute value")
        if token.text == "[":
            self.i += 1
            values = []
            while self.peek() and self.peek().text != "]":
                values.append(self.parse_attr_value())
                self.accept(",")
            self.eat_text("]")
            return values
        if token.kind == "NUMBER":
            self.i += 1
            return float(token.text) if "." in token.text else int(token.text)
        if token.kind == "STRING":
            self.i += 1
            return bytes(token.text[1:-1], "utf-8").decode("unicode_escape")
        if token.text in {"true", "false"}:
            self.i += 1
            return token.text == "true"
        if token.text == "none":
            self.i += 1
            return None
        if token.kind == "IDENT":
            return self.eat_ident()
        raise FrontendSyntaxError(f"expected attribute value, got {token.text!r}", self._span(token))

    def parse_type_until(self, stops: set[str]) -> str:
        parts = []
        depth = 0
        while self.peek() is not None:
            token = self.peek()
            if token is None:
                break
            if depth == 0 and token.text in stops:
                break
            if token.text in {"<", "(", "["}:
                depth += 1
            elif token.text in {">", ")", "]"}:
                depth -= 1
            parts.append(token.text)
            self.i += 1
        if not parts:
            raise FrontendSyntaxError("expected type expression")
        return "".join(parts)

    def eat_keyword(self, keyword: str) -> None:
        token = self.peek()
        if not token or token.text != keyword:
            got = token.text if token else "EOF"
            raise FrontendSyntaxError(f"expected keyword {keyword!r}, got {got!r}", self._span(token))
        self.i += 1


def parse_text(source: str) -> Program:
    return _Parser(source).parse()


def lower_text_to_graph_ir(source: str) -> GraphIRModule:
    program = parse_text(source)
    module = GraphIRModule()
    for source_module in program.modules:
        module.meshes.extend(
            GraphIRMesh.from_attrs(mesh.name, mesh.attrs, source_span=mesh.span)
            for mesh in source_module.meshes
        )
        module.type_aliases.extend(
            GraphIRTypeAlias(decl.name, decl.type_expr, source_span=decl.span)
            for decl in source_module.type_decls
        )
        module.constants.extend(
            GraphIRConstant(decl.name, decl.type_expr, _expr_to_metadata(decl.expr), source_span=decl.span)
            for decl in source_module.const_decls
        )
        for fn in source_module.funcs:
            module.functions.append(_lower_function(fn))
    return module


def compile_report_for_text(
    source: str,
    *,
    program_id: str = "<textual>",
    target: str = "cpu",
):
    """Build a :class:`CompileReport` from a textual-frontend source.

    Step 4 of the 2026-05-18 post-reassessment plan: the textual
    frontend joins ``@tessera.jit`` and ``@clifford_jit`` in
    emitting a uniform report.  Calling this function in a
    :func:`compile_report.capture_compile_reports` scope appends the
    result to the active sink.
    """
    from .. import compile_report as _cr
    module = lower_text_to_graph_ir(source)
    ir_text = module.to_mlir()
    report = _cr.CompileReport(
        program_id=program_id,
        source="textual_frontend",
        frontend=_cr.FRONTEND_TEXTUAL,
        value_kind=_cr.VALUE_KIND_TENSOR,
        target=target,
        ir_hashes={"graph_ir": _cr.hash_ir_text(ir_text)},
        target_decision={target: "textual frontend → GraphIRModule"},
    )
    if _cr.active_sink_is_capturing():
        _cr.emit_compile_report(report)
    return report


def _lower_function(fn: Function) -> GraphIRFunction:
    defined = set()
    value_types: dict[str, Any] = {}
    args = []
    for param in fn.params:
        if param.name in defined:
            raise FrontendSemanticError(f"duplicate symbol {param.name!r} in function {fn.name!r}")
        defined.add(param.name)
        ir_type, dim_names, layout = _type_expr_to_ir_type(param.type_expr)
        value_types[param.name] = ir_type
        args.append(IRArg(param.name, ir_type, dim_names=dim_names, layout=layout))
        if param.default is not None:
            value_types[f"{param.name}.__default__"] = _expr_to_metadata(param.default)

    ops: list[IROp] = []
    return_values: list[str] = []
    returned = False
    for index, stmt in enumerate(fn.body):
        if returned:
            raise FrontendSemanticError(f"function {fn.name!r} has statements after return", getattr(stmt, "span", None))
        if isinstance(stmt, Let):
            if stmt.name in defined:
                raise FrontendSemanticError(f"duplicate symbol {stmt.name!r} in function {fn.name!r}", stmt.span)
            declared_type, _, _ = _type_expr_to_ir_type(stmt.type_expr)
            if stmt.expr is None:
                defined.add(stmt.name)
                value_types[stmt.name] = declared_type
                continue
            result = _lower_expr(stmt.expr, defined, ops, value_types, result_name=stmt.name, declared_type=declared_type)
            if result is None:
                raise FrontendSemanticError(f"let {stmt.name!r} must be initialized from a call or value", stmt.span)
            defined.add(stmt.name)
            value_types.setdefault(stmt.name, declared_type)
            value_types[f"{stmt.name}.__ssa__"] = result
        elif isinstance(stmt, Assign):
            result_name = _assignment_result_name(stmt.target.name, value_types)
            result = _lower_expr(stmt.expr, defined, ops, value_types, result_name=result_name)
            if result is None:
                raise FrontendSemanticError(f"assignment {stmt.target.name!r} must use a call or value", stmt.span)
            defined.add(stmt.target.name)
            if stmt.target.slices and ops:
                ops[-1].kwargs.setdefault("tessera.lvalue_slice", True)
            if stmt.target.name not in value_types and ops:
                value_types[stmt.target.name] = _parse_type_from_string(ops[-1].result_type or str(TENSOR_OPAQUE))
            value_types[f"{stmt.target.name}.__ssa__"] = result
        elif isinstance(stmt, OpStmt):
            _lower_op_stmt(stmt, defined, ops, value_types)
        elif isinstance(stmt, Control):
            _lower_control(stmt, defined, ops, value_types)
        elif isinstance(stmt, Return):
            if index != len(fn.body) - 1:
                raise FrontendSemanticError(f"return must be the final statement in function {fn.name!r}", stmt.span)
            if not stmt.values:
                returned = True
                continue
            if len(stmt.values) == 1:
                value = stmt.values[0]
                if isinstance(value, Name):
                    _require_defined(value, defined)
                    return_values.append(str(value_types.get(f"{value.value}.__ssa__", f"%{value.value}")))
                else:
                    result = _lower_expr(value, defined, ops, value_types, result_name="return_value")
                    if result is None:
                        raise FrontendSemanticError("return expression did not lower to Graph IR", stmt.span)
                    return_values.append(result)
            else:
                operands = []
                operand_types = []
                for value in stmt.values:
                    operand = _lower_expr(value, defined, ops, value_types)
                    if operand is None:
                        raise FrontendSemanticError("return expression did not lower to Graph IR", stmt.span)
                    operands.append(operand)
                    operand_types.append(str(value_types.get(operand.removeprefix("%"), TENSOR_OPAQUE)))
                return_values.extend(operands)
                ops.append(IROp(
                    result="return_values",
                    op_name="tessera.return_pack",
                    operands=operands,
                    operand_types=operand_types,
                    result_type="tensor<*x?>",
                    source_span=stmt.span,
                ))
            returned = True
    if not returned:
        raise FrontendSemanticError(f"function {fn.name!r} is missing a return", fn.span)
    fn_attrs = {"tessera.frontend.kind": f'"{fn.kind}"'}
    result_types = [_type_expr_to_ir_type(type_expr)[0] for type_expr in _return_type_exprs(fn.return_type)]
    return GraphIRFunction(
        name=fn.name,
        args=args,
        result_types=result_types,
        body=ops,
        fn_attrs=fn_attrs,
        return_values=return_values,
    )


def _lower_expr(
    expr: Expr,
    defined: set[str],
    ops: list[IROp],
    value_types: dict[str, Any],
    *,
    result_name: Optional[str] = None,
    declared_type: Optional[Any] = None,
) -> Optional[str]:
    if isinstance(expr, Name):
        if expr.value not in defined:
            raise FrontendSemanticError(f"use of undefined symbol {expr.value!r}", expr.span)
        return str(value_types.get(f"{expr.value}.__ssa__", f"%{expr.value}"))
    if isinstance(expr, Literal):
        result = result_name or f"c{len(ops)}"
        result_type = declared_type or TENSOR_OPAQUE
        ops.append(IROp(
            result=result,
            op_name="tessera.constant",
            operands=[],
            operand_types=[],
            result_type=str(result_type),
            kwargs={"value": expr.value},
            source_span=expr.span,
            inferred_type=result_type,
        ))
        value_types[result] = result_type
        return f"%{result}"
    if isinstance(expr, TensorExpr):
        result = result_name or f"tensor{len(ops)}"
        result_type = tensor_ir_type(expr.shape, expr.dtype, layout=expr.layout)
        ops.append(IROp(
            result=result,
            op_name="tessera.tensor.literal",
            operands=[],
            operand_types=[],
            result_type=str(result_type),
            kwargs={"shape": list(expr.shape), "dtype": expr.dtype, "layout": expr.layout},
            source_span=expr.span,
            inferred_type=result_type,
        ))
        value_types[result] = result_type
        return f"%{result}"
    if isinstance(expr, Binary):
        lhs = _lower_expr(expr.lhs, defined, ops, value_types)
        rhs = _lower_expr(expr.rhs, defined, ops, value_types)
        if lhs is None or rhs is None:
            raise FrontendSemanticError("binary expression operand did not lower to Graph IR", expr.span)
        result = result_name or f"expr{len(ops)}"
        lhs_type = value_types.get(lhs.removeprefix("%"), TENSOR_OPAQUE)
        rhs_type = value_types.get(rhs.removeprefix("%"), TENSOR_OPAQUE)
        result_type = declared_type or lhs_type or rhs_type
        op_names = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
        ops.append(IROp(
            result=result,
            op_name=f"tessera.expr.{op_names.get(expr.op, 'binop')}",
            operands=[lhs, rhs],
            operand_types=[str(lhs_type), str(rhs_type)],
            result_type=str(result_type),
            source_span=expr.span,
            inferred_type=result_type,
        ))
        value_types[result] = result_type
        return f"%{result}"
    if isinstance(expr, Call):
        spec = get_op_spec(expr.name)
        op_name = spec.graph_name if spec is not None else _generic_call_op_name(expr.name)
        if op_name is None:
            raise FrontendSemanticError(f"unknown Tessera op {expr.name!r}", expr.span)
        if spec is not None and not spec.valid_arity(len(expr.args)):
            raise FrontendSemanticError(
                f"op {spec.public_name!r} expects {spec.min_arity}-{spec.max_arity} operands, got {len(expr.args)}",
                expr.span,
            )
        operands = []
        operand_types = []
        for arg in expr.args:
            if isinstance(arg, Literal):
                operand = _lower_expr(arg, defined, ops, value_types)
                operands.append(operand if operand is not None else "%?")
                name = (operand or "%?").removeprefix("%")
                operand_types.append(str(value_types.get(name, TENSOR_OPAQUE)))
                continue
            operand = _lower_expr(arg, defined, ops, value_types)
            operands.append(operand if operand is not None else "%?")
            name = (operand or "%?").removeprefix("%")
            operand_types.append(str(value_types.get(name, TENSOR_OPAQUE)))
        result = result_name or f"v{len(ops)}"
        result_type = declared_type or _infer_textual_result_type(op_name, operands, value_types)
        ops.append(IROp(
            result=result,
            op_name=op_name,
            operands=operands,
            operand_types=operand_types,
            result_type=str(result_type),
            kwargs=dict(expr.attrs),
            source_span=expr.span,
            inferred_type=result_type,
        ))
        value_types[result] = result_type
        return f"%{result}"
    raise FrontendSemanticError(f"unsupported expression {expr!r}", getattr(expr, "span", None))


def _require_defined(expr: Expr, defined: set[str]) -> None:
    if isinstance(expr, Name):
        if expr.value not in defined:
            raise FrontendSemanticError(f"return uses undefined symbol {expr.value!r}", expr.span)
        return
    if isinstance(expr, Call):
        raise FrontendSemanticError("return call expressions must be assigned before return in textual frontend v1", expr.span)
    if isinstance(expr, Literal):
        raise FrontendSemanticError("return literals are not supported in textual frontend v1", expr.span)
    raise FrontendSemanticError("return expression is not supported in textual frontend v1", getattr(expr, "span", None))


def _lower_op_stmt(stmt: OpStmt, defined: set[str], ops: list[IROp], value_types: dict[str, Any]) -> None:
    if not isinstance(stmt.expr, Call):
        raise FrontendSemanticError("standalone operation statement must be a call", stmt.span)
    name = stmt.expr.name
    if name.startswith("schedule."):
        op_name = "tessera." + name
    elif name.startswith("dist."):
        op_name = "tessera." + name
    elif name == "barrier":
        op_name = "tessera.barrier"
    elif name == "assert":
        op_name = "tessera.assert"
    else:
        lowered = _lower_expr(stmt.expr, defined, ops, value_types)
        if lowered is None:
            raise FrontendSemanticError("operation statement did not lower to Graph IR", stmt.span)
        return
    operands = []
    operand_types = []
    for arg in stmt.expr.args:
        if isinstance(arg, Literal):
            continue
        operand = _lower_expr(arg, defined, ops, value_types)
        operands.append(operand if operand is not None else "%?")
        operand_types.append(str(value_types.get((operand or "%?").removeprefix("%"), TENSOR_OPAQUE)))
    ops.append(IROp(
        result=None,
        op_name=op_name,
        operands=operands,
        operand_types=operand_types,
        kwargs=dict(stmt.expr.attrs),
        source_span=stmt.expr.span,
    ))


def _assignment_result_name(name: str, value_types: dict[str, Any]) -> str:
    if f"{name}.__ssa__" not in value_types:
        return name
    version_key = f"{name}.__version__"
    version = int(value_types.get(version_key, 0)) + 1
    value_types[version_key] = version
    return f"{name}_{version}"


def _lower_control(stmt: Control, defined: set[str], ops: list[IROp], value_types: dict[str, Any]) -> None:
    begin_name = f"tessera.scf.{stmt.kind}.begin"
    end_name = f"tessera.scf.{stmt.kind}.end"
    ops.append(IROp(
        result=None,
        op_name=begin_name,
        operands=[],
        operand_types=[],
        kwargs={"region": stmt.kind},
        source_span=stmt.span,
    ))
    for child in stmt.body:
        _lower_stmt_in_region(child, defined, ops, value_types)
    if stmt.else_body:
        ops.append(IROp(
            result=None,
            op_name="tessera.scf.else",
            operands=[],
            operand_types=[],
            kwargs={"region": stmt.kind},
            source_span=stmt.span,
        ))
        for child in stmt.else_body:
            _lower_stmt_in_region(child, defined, ops, value_types)
    ops.append(IROp(
        result=None,
        op_name=end_name,
        operands=[],
        operand_types=[],
        kwargs={"region": stmt.kind},
        source_span=stmt.span,
    ))


def _lower_stmt_in_region(stmt: Stmt, defined: set[str], ops: list[IROp], value_types: dict[str, Any]) -> None:
    if isinstance(stmt, Let):
        declared_type, _, _ = _type_expr_to_ir_type(stmt.type_expr)
        if stmt.expr is None:
            defined.add(stmt.name)
            value_types[stmt.name] = declared_type
            return
        _lower_expr(stmt.expr, defined, ops, value_types, result_name=stmt.name, declared_type=declared_type)
        defined.add(stmt.name)
        value_types.setdefault(stmt.name, declared_type)
        value_types[f"{stmt.name}.__ssa__"] = f"%{stmt.name}"
        return
    if isinstance(stmt, Assign):
        result_name = _assignment_result_name(stmt.target.name, value_types)
        result = _lower_expr(stmt.expr, defined, ops, value_types, result_name=result_name)
        defined.add(stmt.target.name)
        if ops and stmt.target.name not in value_types:
            value_types[stmt.target.name] = _parse_type_from_string(ops[-1].result_type or str(TENSOR_OPAQUE))
        value_types[f"{stmt.target.name}.__ssa__"] = result or f"%{result_name}"
        return
    if isinstance(stmt, OpStmt):
        _lower_op_stmt(stmt, defined, ops, value_types)
        return
    if isinstance(stmt, Control):
        _lower_control(stmt, defined, ops, value_types)
        return
    if isinstance(stmt, Return):
        ops.append(IROp(
            result=None,
            op_name="tessera.scf.yield",
            operands=[],
            operand_types=[],
            source_span=stmt.span,
        ))


def _generic_call_op_name(name: str) -> Optional[str]:
    if name.startswith("op.arch."):
        return "tessera.graph." + name.removeprefix("op.")
    if name.startswith("op.kv_cache."):
        return "tessera.graph." + name.removeprefix("op.")
    if name.startswith("dist."):
        return "tessera.dist." + name.removeprefix("dist.")
    if name.startswith("schedule."):
        return "tessera.schedule." + name.removeprefix("schedule.")
    if name.startswith("tessera."):
        return name
    return None


def _expr_to_metadata(expr: Expr) -> Any:
    if isinstance(expr, Literal):
        return expr.value
    if isinstance(expr, Name):
        return {"name": expr.value}
    if isinstance(expr, TensorExpr):
        return {"tensor": {"shape": list(expr.shape), "dtype": expr.dtype, "layout": expr.layout}}
    if isinstance(expr, Binary):
        return {"binary": {"op": expr.op, "lhs": _expr_to_metadata(expr.lhs), "rhs": _expr_to_metadata(expr.rhs)}}
    if isinstance(expr, Call):
        return {
            "call": expr.name,
            "args": [_expr_to_metadata(arg) for arg in expr.args],
            "attrs": dict(expr.attrs),
        }
    return repr(expr)


def _return_type_exprs(type_expr: Optional[str]) -> list[str]:
    if not type_expr:
        return []
    text = type_expr.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char in "<([":
            depth += 1
        elif char in ">)]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _type_expr_to_ir_type(type_expr: Optional[str]):
    if not type_expr:
        return TENSOR_OPAQUE, (), None
    if not type_expr.startswith("tensor<") or not type_expr.endswith(">"):
        return TENSOR_OPAQUE, (), None
    inner = type_expr[len("tensor<"):-1]
    layout = None
    if ";layout=" in inner:
        inner, layout = inner.split(";layout=", 1)
    elif "; layout=" in inner:
        inner, layout = inner.split("; layout=", 1)
    parts = inner.split("x")
    if len(parts) < 2:
        return TENSOR_OPAQUE, (), layout
    dtype = parts[-1]
    shape = tuple(parts[:-1])
    dim_names = tuple(dim for dim in shape if dim and dim != "?" and not dim.isdigit())
    normalized_shape = tuple("?" if not dim or not dim.isdigit() else dim for dim in shape)
    return tensor_ir_type(normalized_shape, dtype, layout=layout), dim_names, layout


def _parse_type_from_string(type_expr: str):
    if not type_expr.startswith("tensor<") or not type_expr.endswith(">"):
        return TENSOR_OPAQUE
    ir_type, _, _ = _type_expr_to_ir_type(type_expr)
    return ir_type


def _infer_textual_result_type(op_name: str, operands: list[str], value_types: dict[str, Any]):
    operand_types = [value_types.get(operand.removeprefix("%"), TENSOR_OPAQUE) for operand in operands]
    if op_name == "tessera.matmul" and len(operand_types) >= 2:
        lhs, rhs = operand_types[0], operand_types[1]
        if lhs.rank == 2 and rhs.rank == 2:
            return tensor_ir_type((lhs.shape[0], rhs.shape[1]), lhs.dtype or rhs.dtype, layout=lhs.layout)
        return tensor_ir_type(("*",), lhs.dtype or rhs.dtype, layout=lhs.layout)
    return operand_types[0] if operand_types else TENSOR_OPAQUE
