import re
from typing import NamedTuple, Iterator

TOKEN_SPEC = [
    ("WS",       r"[ \t]+"),
    ("COMMENT",  r"//[^\n]*"),
    ("NEWLINE",  r"\n"),
    ("FUNC",     r"\bfunc\b"),
    ("RETURN",   r"\breturn\b"),
    ("MATMUL",   r"\bmatmul\b"),
    ("ADD",      r"\badd\b"),
    ("IDENT",    r"[A-Za-z_][A-Za-z0-9_]*"),
    ("LPAREN",   r"\("),
    ("RPAREN",   r"\)"),
    ("LBRACE",   r"\{"),
    ("RBRACE",   r"\}"),
    ("LBRACK",   r"\["),
    ("RBRACK",   r"\]"),
    ("COMMA",    r","),
    ("EQ",       r"="),
    ("SEMI",     r";"),
]

MASTER = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in TOKEN_SPEC))

class Tok(NamedTuple):
    kind: str
    text: str
    line: int
    col: int

def lex(src: str) -> Iterator[Tok]:
    line = 1
    col = 1
    pos = 0
    while pos < len(src):
        m = MASTER.match(src, pos)
        if not m:
            raise SyntaxError(f"Unexpected character at line {line}, col {col}: {src[pos]!r}")
        kind = m.lastgroup
        text = m.group()
        if kind == "NEWLINE":
            line += 1
            col = 1
        elif kind in ("WS", "COMMENT"):
            col += len(text)
        else:
            yield Tok(kind, text, line, col)
            col += len(text)
        pos = m.end()
