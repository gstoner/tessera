from typing import List, Optional
from .lexer import lex, Tok
from . import ast

class Parser:
    def __init__(self, src: str):
        self.toks = list(lex(src))
        self.i = 0

    def peek(self, k: int = 0):
        if self.i + k < len(self.toks):
            return self.toks[self.i + k]
        return None

    def eat(self, kind: str):
        t = self.peek()
        if not t or t.kind != kind:
            raise SyntaxError(f"Expected {kind}, got {t.kind if t else 'EOF'} at token index {self.i}")
        self.i += 1
        return t

    def accept(self, kind: str):
        t = self.peek()
        if t and t.kind == kind:
            self.i += 1
            return t
        return None

    def parse(self) -> ast.Module:
        funcs: List[ast.Function] = []
        while self.peek() is not None:
            funcs.append(self.parse_function())
        return ast.Module(funcs=funcs)

    def parse_ident_list(self) -> List[str]:
        names = []
        t = self.eat("IDENT")
        names.append(t.text)
        while self.accept("COMMA"):
            t = self.eat("IDENT")
            names.append(t.text)
        return names

    def parse_function(self) -> ast.Function:
        self.eat("FUNC")
        name = self.eat("IDENT").text
        self.eat("LPAREN")
        params: List[str] = []
        if self.peek() and self.peek().kind == "IDENT":
            params = self.parse_ident_list()
        self.eat("RPAREN")
        dims: List[str] = []
        if self.accept("LBRACK"):
            dims = self.parse_ident_list()
            self.eat("RBRACK")
        self.eat("LBRACE")
        body: List[ast.Stmt] = []
        while self.peek() and self.peek().kind != "RBRACE":
            body.append(self.parse_stmt())
        self.eat("RBRACE")
        return ast.Function(name=name, params=params, dims=dims, body=body)

    def parse_stmt(self) -> ast.Stmt:
        t = self.peek()
        if t.kind == "RETURN":
            self.eat("RETURN")
            val = self.eat("IDENT").text
            self.eat("SEMI")
            return ast.Return(value=val)
        target = self.eat("IDENT").text
        self.eat("EQ")
        opkw = self.peek()
        if opkw.kind == "MATMUL":
            self.eat("MATMUL")
            self.eat("LPAREN")
            lhs = self.eat("IDENT").text
            self.eat("COMMA")
            rhs = self.eat("IDENT").text
            self.eat("RPAREN")
            self.eat("SEMI")
            return ast.Assign(target=target, op=ast.MatMul(lhs=lhs, rhs=rhs))
        elif opkw.kind == "ADD":
            self.eat("ADD")
            self.eat("LPAREN")
            lhs = self.eat("IDENT").text
            self.eat("COMMA")
            rhs = self.eat("IDENT").text
            self.eat("RPAREN")
            self.eat("SEMI")
            return ast.Assign(target=target, op=ast.Add(lhs=lhs, rhs=rhs))
        else:
            raise SyntaxError(f"Unknown op at token {self.i}: {opkw.kind}")
