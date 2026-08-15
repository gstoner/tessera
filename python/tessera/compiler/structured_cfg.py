"""Tracer-owned multi-block control-flow carrier.

The tracer records structured regions as nested Graph operation lists.  This
module recovers an explicit basic-block graph from that canonical trace without
re-executing the Python program and without consulting the legacy AST marker
frontend.  Every block carries the inherited typed Presburger-system digest so
region consumers cannot silently lose shape constraints at a branch boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Sequence

from .graph_ir import IROp
from .presburger import PresburgerSystem


STRUCTURED_CFG_SCHEMA = "tessera.structured_cfg.v1"


@dataclass(frozen=True)
class StructuredOperation:
    operation_id: str
    op_name: str
    operands: tuple[str, ...]
    results: tuple[str, ...]


@dataclass(frozen=True)
class StructuredBlock:
    block_id: str
    arguments: tuple[str, ...]
    operations: tuple[StructuredOperation, ...]
    terminator: str
    presburger_digest: str | None = None


@dataclass(frozen=True)
class StructuredEdge:
    source: str
    target: str
    kind: str
    values: tuple[str, ...] = ()
    condition: str | None = None


@dataclass(frozen=True)
class StructuredCFG:
    entry_block: str
    exit_block: str
    blocks: tuple[StructuredBlock, ...]
    edges: tuple[StructuredEdge, ...]
    presburger_system: PresburgerSystem | None = None
    schema: str = STRUCTURED_CFG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STRUCTURED_CFG_SCHEMA:
            raise ValueError("unsupported structured-CFG schema")
        block_ids = tuple(block.block_id for block in self.blocks)
        if not block_ids or len(set(block_ids)) != len(block_ids):
            raise ValueError("structured CFG block identities must be unique")
        known = set(block_ids)
        if self.entry_block not in known or self.exit_block not in known:
            raise ValueError("structured CFG entry/exit blocks must exist")
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError("structured CFG edge references an unknown block")
        reachable = {self.entry_block}
        changed = True
        while changed:
            changed = False
            for edge in self.edges:
                if edge.source in reachable and edge.target not in reachable:
                    reachable.add(edge.target)
                    changed = True
        if reachable != known:
            raise ValueError(
                "structured CFG contains unreachable blocks: "
                + ", ".join(sorted(known - reachable))
            )
        expected_digest = (
            self.presburger_system.digest if self.presburger_system else None
        )
        if any(block.presburger_digest != expected_digest for block in self.blocks):
            raise ValueError("structured CFG lost its inherited Presburger system")

    def with_presburger(self, system: PresburgerSystem) -> "StructuredCFG":
        """Propagate one typed system through every existing region block."""

        return replace(
            self,
            blocks=tuple(
                replace(block, presburger_digest=system.digest)
                for block in self.blocks
            ),
            presburger_system=system,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "entry_block": self.entry_block,
            "exit_block": self.exit_block,
            "presburger_digest": (
                self.presburger_system.digest if self.presburger_system else None
            ),
            "blocks": [
                {
                    "block_id": block.block_id,
                    "arguments": list(block.arguments),
                    "operations": [operation.__dict__ for operation in block.operations],
                    "terminator": block.terminator,
                    "presburger_digest": block.presburger_digest,
                }
                for block in self.blocks
            ],
            "edges": [edge.__dict__ for edge in self.edges],
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class _CFGBuilder:
    def __init__(self, system: PresburgerSystem | None) -> None:
        self.system = system
        self._blocks: dict[str, dict[str, Any]] = {}
        self._edges: list[StructuredEdge] = []
        self._counter = 0

    def new_block(self, hint: str, arguments: Sequence[str] = ()) -> str:
        block_id = f"bb{self._counter}_{hint}"
        self._counter += 1
        self._blocks[block_id] = {
            "arguments": tuple(arguments),
            "operations": [],
            "terminator": "unreachable",
        }
        return block_id

    def append(self, block_id: str, op: IROp) -> None:
        operations: list[StructuredOperation] = self._blocks[block_id]["operations"]
        operations.append(StructuredOperation(
            operation_id=f"{block_id}.op{len(operations)}",
            op_name=op.op_name,
            operands=tuple(op.operands),
            results=tuple(f"%{name}" for name in op.result_names),
        ))

    def terminate(self, block_id: str, terminator: str) -> None:
        if self._blocks[block_id]["terminator"] != "unreachable":
            raise ValueError(f"structured block {block_id} already has a terminator")
        self._blocks[block_id]["terminator"] = terminator

    def edge(
        self,
        source: str,
        target: str,
        kind: str,
        *,
        values: Sequence[str] = (),
        condition: str | None = None,
    ) -> None:
        self._edges.append(StructuredEdge(
            source, target, kind, tuple(values), condition
        ))

    def lower_sequence(self, ops: Sequence[IROp], current: str) -> str:
        for op in ops:
            if op.op_name == "tessera.control_if":
                current = self._lower_if(op, current)
            elif op.op_name in {"tessera.control_for", "tessera.control_scan"}:
                current = self._lower_counted(op, current)
            elif op.op_name == "tessera.control_while":
                current = self._lower_while(op, current)
            else:
                self.append(current, op)
        return current

    def _lower_if(self, op: IROp, current: str) -> str:
        kwargs = op.kwargs
        then_block = self.new_block("if_then")
        else_block = self.new_block("if_else")
        merge = self.new_block("if_merge", op.result_names)
        predicate = f"%{kwargs['_flag_ssa']}"
        self.terminate(current, "cond_br")
        self.edge(current, then_block, "true", condition=predicate)
        self.edge(current, else_block, "false", condition=predicate)
        then_exit = self.lower_sequence(tuple(kwargs["_then_body"]), then_block)
        else_exit = self.lower_sequence(tuple(kwargs["_else_body"]), else_block)
        self.terminate(then_exit, "br")
        self.terminate(else_exit, "br")
        self.edge(then_exit, merge, "yield", values=(f"%{kwargs['_then_ssa']}",))
        self.edge(else_exit, merge, "yield", values=(f"%{kwargs['_else_ssa']}",))
        return merge

    def _lower_counted(self, op: IROp, current: str) -> str:
        kwargs = op.kwargs
        is_scan = op.op_name == "tessera.control_scan"
        header_args = [f"%{kwargs['_carry_ssa']}"]
        if is_scan:
            header_args.append(f"%{kwargs['_ys_ssa']}")
        header = self.new_block("scan_header" if is_scan else "for_header", header_args)
        body = self.new_block("scan_body" if is_scan else "for_body", header_args)
        exit_block = self.new_block(
            "scan_exit" if is_scan else "for_exit", op.result_names
        )
        self.terminate(current, "br")
        self.edge(current, header, "enter", values=op.operands)
        self.terminate(header, "cond_br")
        trip = str(kwargs.get("_trip", "unknown"))
        self.edge(header, body, "loop_body", condition=f"iv < {trip}")
        self.edge(header, exit_block, "loop_exit", condition=f"iv >= {trip}")
        body_exit = self.lower_sequence(tuple(kwargs["_body"]), body)
        self.terminate(body_exit, "br")
        yielded = [f"%{kwargs['_next_ssa']}"]
        if is_scan:
            yielded.append(f"%{kwargs['_y_ssa']}")
        self.edge(body_exit, header, "backedge", values=yielded)
        return exit_block

    def _lower_while(self, op: IROp, current: str) -> str:
        kwargs = op.kwargs
        carry = f"%{kwargs['_carry_ssa']}"
        condition = self.new_block("while_condition", (carry,))
        body = self.new_block("while_body", (carry,))
        exit_block = self.new_block("while_exit", op.result_names)
        self.terminate(current, "br")
        self.edge(current, condition, "enter", values=op.operands)
        condition_exit = self.lower_sequence(tuple(kwargs["_cond"]), condition)
        predicate = f"%{kwargs['_pred_ssa']}"
        self.terminate(condition_exit, "cond_br")
        self.edge(condition_exit, body, "true", condition=predicate)
        self.edge(condition_exit, exit_block, "false", condition=predicate)
        body_exit = self.lower_sequence(tuple(kwargs["_body"]), body)
        self.terminate(body_exit, "br")
        self.edge(
            body_exit,
            condition,
            "backedge",
            values=(f"%{kwargs['_next_ssa']}",),
        )
        return exit_block

    def finish(self, body: Sequence[IROp]) -> StructuredCFG:
        entry = self.new_block("entry")
        current = self.lower_sequence(body, entry)
        exit_block = self.new_block("exit")
        self.terminate(current, "return")
        self.edge(current, exit_block, "return")
        self.terminate(exit_block, "function_exit")
        digest = self.system.digest if self.system else None
        blocks = tuple(
            StructuredBlock(
                block_id=block_id,
                arguments=data["arguments"],
                operations=tuple(data["operations"]),
                terminator=data["terminator"],
                presburger_digest=digest,
            )
            for block_id, data in self._blocks.items()
        )
        return StructuredCFG(
            entry_block=entry,
            exit_block=exit_block,
            blocks=blocks,
            edges=tuple(self._edges),
            presburger_system=self.system,
        )


def recover_structured_cfg(
    body: Sequence[IROp],
    *,
    presburger_system: PresburgerSystem | None = None,
) -> StructuredCFG:
    """Recover an explicit CFG from tracer-owned nested control operations."""

    return _CFGBuilder(presburger_system).finish(body)


__all__ = [
    "STRUCTURED_CFG_SCHEMA",
    "StructuredBlock",
    "StructuredCFG",
    "StructuredEdge",
    "StructuredOperation",
    "recover_structured_cfg",
]
