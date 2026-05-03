"""Textual Tessera Graph DSL parser and Graph IR lowerer.

V1 covers straight-line Graph IR construction: modules, funcs, typed params,
let/assignment, calls, literals, attr blocks, and returns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, TENSOR_OPAQUE
from ..op_catalog import get_op_spec


class FrontendSyntaxError(SyntaxError):
    pass


class FrontendSemanticError(Exception):
    pass


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    line: int
    col: int


TOKEN_SPEC = [
    ("WS", r"[ \t\r]+"),
    ("COMMENT", r"//[^\n]*"),
    ("NEWLINE", r"\n"),
    ("ARROW", r"->"),
    ("NUMBER", r"-?[0-9]+(?:\.[0-9]+)?"),
    ("STRING", r'"(?:\\.|[^"\\])*"'),
    ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("SYMBOL", r"[{}()\[\]<>,;:=.@x?]"),
]
MASTER = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in TOKEN_SPEC))


def _lex(source: str) -> list[Token]:
    tokens = []
    line = 1
    col = 1
    pos = 0
    while pos < len(source):
        match = MASTER.match(source, pos)
        if match is None:
            raise FrontendSyntaxError(f"unexpected character {source[pos]!r} at {line}:{col}")
        kind = match.lastgroup or ""
        text = match.group()
        if kind == "NEWLINE":
            line += 1
            col = 1
        elif kind in {"WS", "COMMENT"}:
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


@dataclass
class Param:
    name: str
    type_expr: Optional[str] = None


@dataclass
class Function:
    name: str
    params: list[Param]
    return_type: Optional[str]
    body: list["Stmt"]
    attrs: dict[str, Any] = field(default_factory=dict)


class Stmt:
    pass


@dataclass
class Let(Stmt):
    name: str
    type_expr: Optional[str]
    expr: "Expr"


@dataclass
class Assign(Stmt):
    name: str
    expr: "Expr"


@dataclass
class Return(Stmt):
    values: list["Expr"]


class Expr:
    pass


@dataclass
class Name(Expr):
    value: str


@dataclass
class Literal(Expr):
    value: Any


@dataclass
class Call(Expr):
    name: str
    args: list[Expr]
    attrs: dict[str, Any] = field(default_factory=dict)


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
            raise FrontendSyntaxError(f"expected {text!r}, got {got!r} at token {self.i}")
        self.i += 1
        return token

    def eat_ident(self) -> str:
        token = self.peek()
        if not token or token.kind != "IDENT":
            got = token.text if token else "EOF"
            raise FrontendSyntaxError(f"expected identifier, got {got!r} at token {self.i}")
        self.i += 1
        return token.text

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
        while self.peek() and self.peek().text != "}":
            funcs.append(self.parse_func())
        self.eat_text("}")
        return Module(name=name, funcs=funcs)

    def parse_func(self) -> Function:
        self.eat_keyword("func")
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
        return Function(name=name, params=params, return_type=return_type, body=body, attrs=attrs)

    def parse_param(self) -> Param:
        name = self.eat_ident()
        type_expr = None
        if self.accept(":"):
            type_expr = self.parse_type_until({",", ")"})
        return Param(name=name, type_expr=type_expr)

    def parse_stmt(self) -> Stmt:
        token = self.peek()
        if token is None:
            raise FrontendSyntaxError("unexpected EOF in statement")
        if token.text == "let":
            self.i += 1
            name = self.eat_ident()
            type_expr = None
            if self.accept(":"):
                type_expr = self.parse_type_until({"="})
            self.eat_text("=")
            expr = self.parse_expr()
            self.eat_text(";")
            return Let(name, type_expr, expr)
        if token.text == "return":
            self.i += 1
            values = [self.parse_expr()]
            while self.accept(","):
                values.append(self.parse_expr())
            self.eat_text(";")
            return Return(values)
        name = self.eat_ident()
        self.eat_text("=")
        expr = self.parse_expr()
        self.eat_text(";")
        return Assign(name, expr)

    def parse_expr(self) -> Expr:
        token = self.peek()
        if token is None:
            raise FrontendSyntaxError("unexpected EOF in expression")
        if token.kind == "NUMBER":
            self.i += 1
            return Literal(float(token.text) if "." in token.text else int(token.text))
        if token.kind == "STRING":
            self.i += 1
            return Literal(bytes(token.text[1:-1], "utf-8").decode("unicode_escape"))
        if token.text in {"true", "false"}:
            self.i += 1
            return Literal(token.text == "true")
        if token.text == "none":
            self.i += 1
            return Literal(None)
        name = self.parse_qualified_name()
        if self.accept("("):
            args = []
            if self.peek() and self.peek().text != ")":
                args.append(self.parse_expr())
                while self.accept(","):
                    args.append(self.parse_expr())
            self.eat_text(")")
            attrs = self.parse_attr_block() if self.peek() and self.peek().text == "@" else {}
            return Call(name, args, attrs)
        return Name(name)

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
            value = self.parse_expr()
            if not isinstance(value, Literal):
                raise FrontendSyntaxError("attribute values must be literals")
            attrs[key] = value.value
            self.accept(",")
        self.eat_text("}")
        return attrs

    def parse_type_until(self, stops: set[str]) -> str:
        parts = []
        depth = 0
        while self.peek() is not None:
            token = self.peek()
            if token is None:
                break
            if depth == 0 and token.text in stops:
                break
            if token.text == "<":
                depth += 1
            elif token.text == ">":
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
            raise FrontendSyntaxError(f"expected keyword {keyword!r}, got {got!r}")
        self.i += 1


def parse_text(source: str) -> Program:
    return _Parser(source).parse()


def lower_text_to_graph_ir(source: str) -> GraphIRModule:
    program = parse_text(source)
    module = GraphIRModule()
    for source_module in program.modules:
        for fn in source_module.funcs:
            module.functions.append(_lower_function(fn))
    return module


def _lower_function(fn: Function) -> GraphIRFunction:
    defined = set()
    args = []
    for param in fn.params:
        if param.name in defined:
            raise FrontendSemanticError(f"duplicate symbol {param.name!r} in function {fn.name!r}")
        defined.add(param.name)
        args.append(IRArg(param.name, TENSOR_OPAQUE))

    ops: list[IROp] = []
    returned = False
    for index, stmt in enumerate(fn.body):
        if returned:
            raise FrontendSemanticError(f"function {fn.name!r} has statements after return")
        if isinstance(stmt, Let):
            if stmt.name in defined:
                raise FrontendSemanticError(f"duplicate symbol {stmt.name!r} in function {fn.name!r}")
            result = _lower_expr(stmt.expr, defined, ops, result_name=stmt.name)
            if result is None:
                raise FrontendSemanticError(f"let {stmt.name!r} must be initialized from a call or value")
            defined.add(stmt.name)
        elif isinstance(stmt, Assign):
            result = _lower_expr(stmt.expr, defined, ops, result_name=stmt.name)
            if result is None:
                raise FrontendSemanticError(f"assignment {stmt.name!r} must use a call or value")
            defined.add(stmt.name)
        elif isinstance(stmt, Return):
            if index != len(fn.body) - 1:
                raise FrontendSemanticError(f"return must be the final statement in function {fn.name!r}")
            for value in stmt.values:
                _require_defined(value, defined)
            if len(stmt.values) != 1:
                raise FrontendSemanticError("textual frontend v1 supports one return value")
            if ops and isinstance(stmt.values[0], Name) and stmt.values[0].value != ops[-1].result:
                raise FrontendSemanticError("textual frontend v1 return value must be the final assigned op")
            returned = True
    if not returned:
        raise FrontendSemanticError(f"function {fn.name!r} is missing a return")
    return GraphIRFunction(name=fn.name, args=args, body=ops)


def _lower_expr(expr: Expr, defined: set[str], ops: list[IROp], *, result_name: Optional[str] = None) -> Optional[str]:
    if isinstance(expr, Name):
        if expr.value not in defined:
            raise FrontendSemanticError(f"use of undefined symbol {expr.value!r}")
        return f"%{expr.value}"
    if isinstance(expr, Literal):
        return None
    if isinstance(expr, Call):
        spec = get_op_spec(expr.name)
        if spec is None:
            raise FrontendSemanticError(f"unknown Tessera op {expr.name!r}")
        if not spec.valid_arity(len(expr.args)):
            raise FrontendSemanticError(
                f"op {spec.public_name!r} expects {spec.min_arity}-{spec.max_arity} operands, got {len(expr.args)}"
            )
        operands = []
        for arg in expr.args:
            if isinstance(arg, Literal):
                raise FrontendSemanticError(f"literal positional operands are not supported for op {spec.public_name!r}; use attrs")
            operand = _lower_expr(arg, defined, ops)
            operands.append(operand if operand is not None else "%?")
        result = result_name or f"v{len(ops)}"
        ops.append(IROp(
            result=result,
            op_name=spec.graph_name,
            operands=operands,
            operand_types=["tensor<*x?>"] * len(operands),
            result_type="tensor<*x?>",
            kwargs=dict(expr.attrs),
        ))
        return f"%{result}"
    raise FrontendSemanticError(f"unsupported expression {expr!r}")


def _require_defined(expr: Expr, defined: set[str]) -> None:
    if isinstance(expr, Name):
        if expr.value not in defined:
            raise FrontendSemanticError(f"return uses undefined symbol {expr.value!r}")
        return
    if isinstance(expr, Call):
        raise FrontendSemanticError("return call expressions must be assigned before return in textual frontend v1")
    if isinstance(expr, Literal):
        raise FrontendSemanticError("return literals are not supported in textual frontend v1")
