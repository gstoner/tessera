"""Compiler-testing program #4 — textual-IR round-trip + parser fuzz.

Two properties over the frontend textual-IR parser
(`tessera.compiler.frontend.parser`):

1. **Round-trip** — a randomly generated valid op-chain renders to the textual
   DSL, parses via `lower_text_to_graph_ir`, and the recovered Graph-IR op names
   exactly match what was generated. (Generate → render → parse → compare; the
   "print" step is the generator, so no separate printer is needed.)
2. **Fuzz / crash-safety** — mutating a valid program (truncate / delete / inject
   garbage) must EITHER parse OR raise a *named* parser error
   (`FrontendSyntaxError` / `FrontendSemanticError`) — never an uncaught
   non-parser exception (a crash) and never a hang.

Pure stdlib `random` with deterministic seeds (mirrors `test_string_parsers_fuzz`)
— no Hypothesis dependency, runs everywhere.
"""

import random

import pytest

from tessera.compiler.frontend import parser as P

_SEEDS = [0, 1, 2, 7, 13, 42, 123, 999, 31337]
_UNARY = ["silu", "relu", "gelu", "sigmoid", "tanh", "softmax", "rmsnorm",
          "layer_norm"]
_BINARY = ["matmul", "add", "sub", "mul", "div"]
# The named parser errors a malformed program is allowed to raise.
_PARSER_ERRORS = (P.FrontendSyntaxError, P.FrontendSemanticError)


def _gen_program(rng: random.Random, n_ops: int):
    """Build a random valid op-chain; return (dsl_text, expected_graph_op_names)."""
    names = ["x", "w"]
    lines, expected = [], []
    for i in range(n_ops):
        if rng.random() < 0.5:
            op = rng.choice(_BINARY)
            a, b = rng.choice(names), rng.choice(names)
            lines.append(f"    v{i} = op.{op}({a}, {b});")
        else:
            op = rng.choice(_UNARY)
            a = rng.choice(names)
            lines.append(f"    v{i} = op.{op}({a});")
        names.append(f"v{i}")
        expected.append(f"tessera.{op}")
    body = "\n".join(lines)
    dsl = (f"module m {{\n  func f(x: tensor<?xfp32>, w: tensor<?xfp32>) "
           f"-> tensor<?xfp32> {{\n{body}\n    return v{n_ops - 1};\n  }}\n}}")
    return dsl, expected


def _mutate(rng: random.Random, s: str) -> str:
    """One random structure-breaking mutation."""
    if not s:
        return rng.choice(["{", "}", "op.", "func"])
    kind = rng.randint(0, 5)
    i = rng.randrange(len(s))
    if kind == 0:                                   # truncate
        return s[:i]
    if kind == 1:                                   # delete a char
        return s[:i] + s[i + 1:]
    if kind == 2:                                   # duplicate a char
        return s[:i] + s[i] + s[i:]
    if kind == 3:                                   # replace with punctuation
        return s[:i] + rng.choice("{}()<>;,:%@#$") + s[i + 1:]
    if kind == 4:                                   # inject a garbage token
        return s[:i] + rng.choice([" op.", " ?? ", " 99 ", " tensor"]) + s[i:]
    return s + rng.choice(["", "}", ")", "garbage"])  # append junk


# --- round-trip property ----------------------------------------------------- #
@pytest.mark.parametrize("seed", _SEEDS)
def test_generated_program_round_trips(seed):
    rng = random.Random(seed)
    for _ in range(12):
        n = rng.randint(1, 8)
        dsl, expected = _gen_program(rng, n)
        mod = P.lower_text_to_graph_ir(dsl)
        got = [op.op_name for op in mod.functions[0].body]
        assert got == expected, f"round-trip mismatch:\n{dsl}\ngot {got}"


def test_single_op_round_trips():
    dsl, expected = _gen_program(random.Random(0), 1)
    mod = P.lower_text_to_graph_ir(dsl)
    assert [op.op_name for op in mod.functions[0].body] == expected


# --- fuzz / crash-safety ----------------------------------------------------- #
@pytest.mark.parametrize("seed", _SEEDS)
def test_mutated_program_is_crash_safe(seed):
    """A mutated program parses or raises a NAMED parser error — never an
    uncaught crash, never a hang."""
    rng = random.Random(seed)
    for _ in range(40):
        n = rng.randint(1, 6)
        dsl, _ = _gen_program(rng, n)
        bad = dsl
        for _ in range(rng.randint(1, 4)):
            bad = _mutate(rng, bad)
        try:
            P.lower_text_to_graph_ir(bad)          # may legitimately parse
        except _PARSER_ERRORS:
            pass                                   # the contract: named error
        except RecursionError as e:                # crash bugs we want surfaced
            pytest.fail(f"parser recursion blow-up on:\n{bad!r}\n{e}")
        except Exception as e:                     # noqa: BLE001
            pytest.fail(
                f"parser raised a non-parser {type(e).__name__} (crash) on:\n"
                f"{bad!r}\n{e}")


@pytest.mark.parametrize("garbage", [
    "", "   ", "{", "}", "module", "module {", "func f(", "op.matmul",
    "module m { func f() -> tensor<?xfp32> { return ; } }",
    "module { { { { {", ")" * 50, "op." * 100,
])
def test_degenerate_inputs_crash_safe(garbage):
    try:
        P.lower_text_to_graph_ir(garbage)
    except _PARSER_ERRORS:
        pass
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"non-parser {type(e).__name__} on degenerate input "
                    f"{garbage!r}: {e}")


def test_every_curated_op_parses():
    """The parser accepts the full curated op vocabulary (a stub op that the
    catalog dropped would fail to parse / lower here)."""
    for op in _UNARY:
        mod = P.lower_text_to_graph_ir(
            f"module m {{ func f(x: tensor<?xfp32>) -> tensor<?xfp32> "
            f"{{ a = op.{op}(x); return a; }} }}")
        assert mod.functions[0].body[0].op_name == f"tessera.{op}"
    for op in _BINARY:
        mod = P.lower_text_to_graph_ir(
            f"module m {{ func f(x: tensor<?xfp32>, w: tensor<?xfp32>) -> "
            f"tensor<?xfp32> {{ a = op.{op}(x, w); return a; }} }}")
        assert mod.functions[0].body[0].op_name == f"tessera.{op}"
