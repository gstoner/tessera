"""Opt-in pytest plugin: census of what still builds the AST Graph IR module.

E2E-REAL-6F retires the AST front end (``graph_ir._OpExtractor``). Its one
production entry is ``GraphIRBuilder.lower``; this plugin records every call
to it, whether it took the AST route or the abstract trace, why the trace was
not used, and which reader asked for the module. It also records every
``JitFn`` decoration and whether it deferred the AST emission, so a run
answers "how many decorations still emit at decoration, and why".

Usage (any host; the census is host-independent but says which host ran it)::

    TESSERA_AST_CENSUS_DIR=/path/to/out \\
        python3 -m pytest -p scripts.ast_lowering_census tests/unit -m "not slow" -n 8
    python3 scripts/ast_lowering_census.py summarize /path/to/out

Each process appends JSON lines to ``<dir>/census-<pid>.jsonl``; the
directory must exist and should be empty before a run. Without
``TESSERA_AST_CENSUS_DIR`` the plugin refuses to load rather than record
nowhere. It wraps the functions it measures and never changes their result
or their exceptions.
"""
from __future__ import annotations

import collections
import functools
import importlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_ENV = "TESSERA_AST_CENSUS_DIR"
_STATE = threading.local()
_LOCK = threading.Lock()
_OUT: Optional[Path] = None
_TESSERA_PKG = os.sep + os.path.join("python", "tessera") + os.sep
_JIT_FILE = os.path.join("tessera", "compiler", "jit.py")
_GRAPH_IR_FILE = os.path.join("tessera", "compiler", "graph_ir.py")
# Frames that belong to the emission machinery itself; the reader is the
# first frame above them.
_MACHINERY = {
    (_JIT_FILE, "get"), (_JIT_FILE, "_materialize_ast_emission"),
    (_JIT_FILE, "_emit"), (_JIT_FILE, "_jit_emit_graph_ir"),
    (_JIT_FILE, "_ensure_legacy_graph_ir"), (_JIT_FILE, "_needs_trace"),
    (_JIT_FILE, "recognized_package"),
    (_JIT_FILE, "_recover_frontend_diagnostics"),
    (_GRAPH_IR_FILE, "lower"),
    ("ast_lowering_census.py", "_materialize_ast_emission"),
    ("ast_lowering_census.py", "__init__"),
}


def _write(record: Dict[str, Any]) -> None:
    if _OUT is None:
        return
    test = os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0]
    record["test"] = test
    line = json.dumps(record, sort_keys=True, default=str) + "\n"
    with _LOCK:
        with (_OUT / f"census-{os.getpid()}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(line)


def _short(filename: str) -> str:
    index = filename.rfind(_TESSERA_PKG)
    if index >= 0:
        return filename[index + len(os.sep + "python" + os.sep):]
    return os.path.basename(filename)


def _target_label(target: Any) -> Optional[str]:
    if target is None:
        return None
    if isinstance(target, str):
        return target
    for attr in ("arch", "name", "isa"):
        value = getattr(target, attr, None)
        if value is not None:
            return f"{type(target).__name__}:{value}"
    return type(target).__name__


def _stack_facts(skip: int = 3) -> Dict[str, Any]:
    """Classify the current stack: phase, emitter, field, reader, target."""
    from tessera.compiler.jit import JitFn

    phase = "other"
    emitter = None
    field = None
    reader = None
    target = None
    decorate_locals = None
    # Default skips _stack_facts, _record_lower and the lower wrapper.
    frame = sys._getframe(skip)
    while frame is not None:
        code = frame.f_code
        short = _short(code.co_filename)
        key = (short, code.co_name)
        if key == (_JIT_FILE, "get") and field is None:
            field = frame.f_locals.get("name")
        elif key in {(_JIT_FILE, "_needs_trace"), (_JIT_FILE, "recognized_package")} \
                and field is None:
            field = code.co_name
        if emitter is None and short == _JIT_FILE and code.co_name in {
                "_jit_emit_graph_ir", "_ensure_legacy_graph_ir",
                "_recover_frontend_diagnostics"}:
            emitter = code.co_name
            if code.co_name == "_jit_emit_graph_ir":
                target = frame.f_locals.get("target_kind")
        if reader is None and key not in _MACHINERY and key != (_GRAPH_IR_FILE, "lower"):
            if short != _GRAPH_IR_FILE or code.co_name != "lower":
                reader = f"{short}:{code.co_name}"
        self_obj = frame.f_locals.get("self")
        if isinstance(self_obj, JitFn):
            if target is None:
                target = _target_label(getattr(self_obj, "target", None)) or "none"
            if code.co_name == "__call__" and phase == "other":
                phase = "call"
        if short == _JIT_FILE and code.co_name == "_decorate":
            if phase == "other":
                phase = "decoration"
            decorate_locals = frame.f_locals
        frame = frame.f_back
    if emitter is None:
        emitter = "direct"
    return {"phase": phase, "emitter": emitter, "field": field,
            "reader": reader, "target": target, "_decorate": decorate_locals}


def _eager_reasons(scope: Optional[Dict[str, Any]]) -> List[str]:
    """Why a decoration that emitted eagerly did not defer (decorator locals)."""
    if not scope:
        return []
    # ``tessera.compiler.jit`` the attribute is the decorator, not the module.
    jit_mod = importlib.import_module("tessera.compiler.jit")
    from tessera.compiler.effect_scan import scan_call_effects
    from tessera.compiler.op_catalog import get_op_spec

    reasons: List[str] = []
    fn = scope.get("fn")
    source_text = scope.get("source_text")
    target_kind = scope.get("target_kind")
    if scope.get("autodiff") is not None or scope.get("wrt") is not None:
        reasons.append("differentiation_request")
    if scope.get("shape_bounds") is not None:
        reasons.append("shape_bounds")
    if source_text is None:
        reasons.append("source_unavailable")
        return reasons
    try:
        if target_kind == "apple_gpu":
            if scope.get("_auto_batch"):
                reasons.append("apple_gpu:auto_batch")
            if scope.get("emit_package"):
                reasons.append("apple_gpu:emit_package")
            if jit_mod._resolve_dispatch_via_package(
                    scope.get("dispatch_via_package"), target_kind):
                reasons.append("apple_gpu:dispatch_via_package")
            if any(get_op_spec(name) is None
                   for name, _ in scan_call_effects(fn, source_text).ops):
                reasons.append("apple_gpu:uncatalogued_op")
            if jit_mod._body_may_need_tracer(fn, source_text):
                reasons.append("apple_gpu:body_may_need_tracer")
        if target_kind != "cpu" and not jit_mod._scan_admits_target_capability(
                fn, target_kind, source_text):
            reasons.append("capability_scan_refused")
    except Exception as exc:  # the census must never change the run
        reasons.append(f"reason_probe_failed:{type(exc).__name__}:{exc}")
    return reasons or ["unexplained"]


def _wrap_lower(original):
    @functools.wraps(original)
    def lower(self, fn, effect_tag=None, target_attr=None, source_text=None,
              prefer_abstract_trace=False, source_origin=None):
        if getattr(_STATE, "depth", 0):
            return original(self, fn, effect_tag, target_attr, source_text,
                            prefer_abstract_trace, source_origin)
        _STATE.depth = 1
        _STATE.ast_ran = False
        _STATE.trace_error = None
        try:
            return original(self, fn, effect_tag, target_attr, source_text,
                            prefer_abstract_trace, source_origin)
        finally:
            _STATE.depth = 0
            try:
                _record_lower(fn, prefer_abstract_trace)
            except Exception as exc:  # never mask the lowering's own outcome
                _write({"kind": "census_error", "error": repr(exc)})
    return lower


def _record_lower(fn, prefer_abstract_trace: bool) -> None:
    from tessera.compiler.graph_ir import _abstract_trace_specs, ir_args_from_signature

    facts = _stack_facts()
    decorate = facts.pop("_decorate")
    route = "ast" if _STATE.ast_ran else "trace"
    why = None
    if route == "ast":
        if not prefer_abstract_trace:
            why = "trace_not_preferred_effectful"
        else:
            try:
                specs = _abstract_trace_specs(ir_args_from_signature(fn))
            except Exception:
                specs = None
            if specs is None:
                why = "signature_not_traceable"
            else:
                why = f"trace_failed:{_STATE.trace_error or 'unknown'}"
    record = {"kind": "lower", "route": route, "why_ast": why,
              "function": getattr(fn, "__qualname__", repr(fn)), **facts}
    del decorate  # eager reasons are recorded per decoration, cache hits included
    _write(record)


def _wrap_extract(original):
    @functools.wraps(original)
    def _extract_ops(self, *args, **kwargs):
        _STATE.ast_ran = True
        return original(self, *args, **kwargs)
    return _extract_ops


def _wrap_trace_stage(original):
    @functools.wraps(original)
    def stage(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except Exception as exc:
            if getattr(_STATE, "depth", 0):
                _STATE.trace_error = type(exc).__name__
            raise
    return stage


def _wrap_init(original):
    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        try:
            deferred = kwargs.get("deferred_emission") is not None
            facts = _stack_facts(skip=2)
            if facts["_decorate"] is None:
                # Not a decoration: an internal owner (native vmap / public
                # JVP) built from a module its constructor already holds.
                record = {"kind": "owner", "constructor": facts["reader"]}
            else:
                record = {"kind": "decoration", "deferred": deferred}
                if not deferred:
                    record["eager_reasons"] = _eager_reasons(facts["_decorate"])
            _write({**record,
                    "target": _target_label(getattr(self, "target", None)) or "none",
                    "function": getattr(getattr(self, "_fn", None), "__qualname__", None)})
        except Exception as exc:
            _write({"kind": "census_error", "error": repr(exc)})
    return __init__


def _wrap_materialize(original):
    """A deferred emission runs once per JitFn; record who triggered it.

    ``lower`` alone undercounts readers: ``graph_ir_cache`` answers a repeat
    source without lowering, so a reader that forces the module would be
    invisible on a cache hit. ``pending`` distinguishes the first, real
    materialization from a later no-op call.
    """
    @functools.wraps(original)
    def _materialize_ast_emission(self, *args, **kwargs):
        pending = self.ast_emission_pending
        try:
            return original(self, *args, **kwargs)
        finally:
            if pending:
                try:
                    facts = _stack_facts(skip=2)
                    facts.pop("_decorate")
                    _write({"kind": "materialize",
                            "function": getattr(getattr(self, "_fn", None),
                                                "__qualname__", None),
                            **facts})
                except Exception as exc:
                    _write({"kind": "census_error", "error": repr(exc)})
    return _materialize_ast_emission


def pytest_configure(config) -> None:
    global _OUT
    raw = os.environ.get(_ENV)
    if not raw:
        raise RuntimeError(
            f"scripts.ast_lowering_census needs {_ENV}=<existing directory>")
    out = Path(raw)
    if not out.is_dir():
        raise RuntimeError(f"{_ENV}={raw!r} is not an existing directory")
    _OUT = out
    from tessera.compiler import graph_ir, trace
    from tessera.compiler.jit import JitFn

    graph_ir.GraphIRBuilder.lower = _wrap_lower(graph_ir.GraphIRBuilder.lower)
    graph_ir.GraphIRBuilder._extract_ops = _wrap_extract(graph_ir.GraphIRBuilder._extract_ops)
    trace.trace = _wrap_trace_stage(trace.trace)
    trace.to_graph_ir_module = _wrap_trace_stage(trace.to_graph_ir_module)
    JitFn.__init__ = _wrap_init(JitFn.__init__)
    JitFn._materialize_ast_emission = _wrap_materialize(JitFn._materialize_ast_emission)


# ── summary ────────────────────────────────────────────────────────────────

def _records(directory: Path) -> Iterable[Dict[str, Any]]:
    for path in sorted(directory.glob("census-*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def summarize(directory: Path, top: int = 15) -> str:
    records = list(_records(directory))
    if not records:
        raise SystemExit(f"no census records under {directory}")
    lowers = [r for r in records if r["kind"] == "lower"]
    ast = [r for r in lowers if r["route"] == "ast"]
    decorations = [r for r in records if r["kind"] == "decoration"]
    owners = [r for r in records if r["kind"] == "owner"]
    errors = [r for r in records if r["kind"] == "census_error"]
    materialized = [r for r in records if r["kind"] == "materialize"]

    def table(title: str, counter: collections.Counter) -> List[str]:
        rows = [f"\n### {title}\n", "| count | key |", "|---:|---|"]
        rows += [f"| {n} | `{k}` |" for k, n in counter.most_common(top)]
        return rows

    out = ["## AST lowering census", "",
           f"- `GraphIRBuilder.lower` calls: {len(lowers)} "
           f"(AST route {len(ast)}, abstract trace {len(lowers) - len(ast)})",
           f"- `JitFn` decorations: {len(decorations)} (deferred "
           f"{sum(r['deferred'] for r in decorations)})",
           f"- internal `JitFn` owners (not decorations): {len(owners)}",
           f"- deferred emissions later materialized: {len(materialized)} "
           f"(each builds the AST module unless graph_ir_cache already holds it)",
           f"- census errors: {len(errors)}"]
    out += table("AST lowerings by phase / emitter",
                 collections.Counter(f"{r['phase']} / {r['emitter']}" for r in ast))
    out += table("AST lowerings by why the trace was not used",
                 collections.Counter(r["why_ast"] for r in ast))
    out += table("AST lowerings by target",
                 collections.Counter(str(r["target"]) for r in ast))
    out += table("Eager decorations by reason they did not defer",
                 collections.Counter(f"{r['target']} / {reason}" for r in decorations
                                     if not r["deferred"]
                                     for reason in r.get("eager_reasons", [])))
    out += table("Call-time / deferred AST lowerings by field read",
                 collections.Counter(str(r["field"]) for r in ast if r["field"] is not None))
    out += table("Call-time / deferred AST lowerings by reader",
                 collections.Counter(str(r["reader"]) for r in ast if r["field"] is not None
                                     or r["emitter"] != "_jit_emit_graph_ir"))
    out += table("Deferred emissions materialized, by phase / field",
                 collections.Counter(f"{r['phase']} / {r['field']}" for r in materialized))
    out += table("Deferred emissions materialized, by reader",
                 collections.Counter(str(r["reader"]) for r in materialized))
    out += table("Deferred emissions materialized, by target",
                 collections.Counter(str(r["target"]) for r in materialized))
    out += table("Internal owners by constructor",
                 collections.Counter(str(r["constructor"]) for r in owners))
    out += table("Eager vs deferred decorations by target",
                 collections.Counter(f"{r['target']} / {'deferred' if r['deferred'] else 'eager'}"
                                     for r in decorations))
    return "\n".join(out) + "\n"


def main(argv: List[str]) -> int:
    if len(argv) < 2 or argv[0] != "summarize":
        print(__doc__)
        return 2
    print(summarize(Path(argv[1])))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    raise SystemExit(main(sys.argv[1:]))
