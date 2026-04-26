from dataclasses import dataclass
from typing import List

@dataclass
class Module:
    funcs: List["Function"]

@dataclass
class Function:
    name: str
    params: List[str]
    dims: List[str]
    body: List["Stmt"]

class Stmt: pass

@dataclass
class Assign(Stmt):
    target: str
    op: "Op"

@dataclass
class Return(Stmt):
    value: str

class Op: pass

@dataclass
class MatMul(Op):
    lhs: str
    rhs: str

@dataclass
class Add(Op):
    lhs: str
    rhs: str
