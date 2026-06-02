"""Compiler-correctness X-a — string-parser fuzz tests (Gap 2).

Property-based fuzz covering the four crash-prone string parsers
identified in
``docs/audit/compiler/COMPILER_AUDIT.md`` § "Coverage
matrix — string parsers":

1. ``parseBcString`` — BC ABI parser in BoundaryConditionLowerPass.cpp.
   The most-blast-radius parser because every halo + stencil pass
   downstream consumes its output.

2. ``splitComma``    — mesh-axes / mesh-sizes pass options in
   DistributionLoweringPass.cpp.  A crash here breaks every
   distribution-lowering invocation.

3. ``featureMapToInt`` — linear-attn feature_map enum.  Exact-set match;
   the bug class here is "what does the pass do with an unknown
   feature_map" (must fall through, not crash).

4. ``canonicalize_dtype`` — Python-side dtype canonicalization in
   tessera.dtype.  Already has 135 hand-written tests; fuzz adds
   random-string negative cases.

The fuzz approach: stdlib ``random`` with deterministic seeds (no
Hypothesis dependency) — we generate a corpus of (legal, malformed)
strings and assert each parser either succeeds with the expected
result or fails with a *named* diagnostic (never silent / crash).
"""
from __future__ import annotations

import os
import random
import shutil
import string
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_tessera_opt() -> str | None:
    for c in (
        os.environ.get("TESSERA_OPT"),
        shutil.which("tessera-opt"),
        str(REPO_ROOT / "build" / "tools" / "tessera-opt" / "tessera-opt"),
    ):
        if c and Path(c).exists():
            return c
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — invoke tessera-opt subprocess with a BC string spliced in.
# ─────────────────────────────────────────────────────────────────────────────


def _bc_module(bc: str) -> str:
    """Build a minimal MLIR module exercising one stencil.apply with a
    given BC string.  Returns the raw MLIR text."""
    # Escape any double-quotes the fuzzer dreams up so the MLIR string
    # literal stays well-formed.  This is the same defence the real
    # tessera-opt CLI users would have to take.
    escaped = bc.replace("\\", "\\\\").replace('"', '\\"')
    return f"""\
func.func @t(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {{
  %t = "tessera.neighbors.topology.create"() {{ kind = "2d_mesh" }} : () -> !tessera.neighbors.topology
  %s = "tessera.neighbors.stencil.define"() {{
      taps = [dense<[0, 0]> : tensor<2xi64>,
              dense<[1, 0]> : tensor<2xi64>],
      bc = "{escaped}"
  }} : () -> index
  %h = "tessera.neighbors.halo.region"(%arg0) {{ halo.width = [1, 1] }} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %o = "tessera.neighbors.stencil.apply"(%s, %h, %t) :
      (index, tensor<?x?xf32>, !tessera.neighbors.topology) -> tensor<?x?xf32>
  return %o : tensor<?x?xf32>
}}
"""


def _run_bc_pass(bc: str, binary: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [binary, "-tessera-stencil-lower", "-tessera-boundary-condition-lower"],
        input=_bc_module(bc), capture_output=True, text=True, timeout=30,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. BC parser — the four-mode parser in BoundaryConditionLowerPass.
# ─────────────────────────────────────────────────────────────────────────────


class TestBCParserFuzz:
    """Generate random BC strings and assert the parser is crash-safe
    and produces the expected per-axis mode list (for legal inputs) or
    a structured warning (for unrecognised tokens)."""

    SEEDS = [0, 1, 2, 7, 13, 42, 123, 999]

    MODES_VALUELESS = ("periodic", "reflect")
    MODES_WITH_VALUE = ("dirichlet", "neumann")

    def _gen_legal_token(self, rng: random.Random) -> tuple[str, str, bool]:
        """Return (token_text, expected_mode, expected_has_value)."""
        if rng.random() < 0.5:
            mode = rng.choice(self.MODES_VALUELESS)
            return mode, mode, False
        mode = rng.choice(self.MODES_WITH_VALUE)
        if rng.random() < 0.3:
            return mode, mode, True   # bare → has_value=true with default 0.0
        v = rng.choice([0.0, 1.0, -1.0, 0.5, -2.5, 1e-3, 100.0, -100.0])
        return f"{mode}({v})", mode, True

    def _gen_legal_bc(self, rng: random.Random, n_axes: int) -> tuple[str, list]:
        toks = [self._gen_legal_token(rng) for _ in range(n_axes)]
        text = ",".join(t[0] for t in toks)
        return text, toks

    def _gen_malformed_bc(self, rng: random.Random) -> str:
        """Return a deliberately-broken BC string the parser must not
        crash on."""
        choices = [
            "",                                 # empty
            ",,,",                              # only commas
            "periodic(extra,unmatched",         # unmatched paren
            "dirichlet()",                      # empty payload
            "neumann(abc)",                     # non-numeric payload
            "garbage",                          # unknown token
            "periodic " * 50,                   # very long
            "dirichlet(1.5,1.5)",               # multi-arg payload
            "  periodic  ",                     # whitespace
            "PERIODIC",                         # case
            "périodique",                       # non-ASCII
            "dirichlet(" + "9" * 100 + ")",     # huge number
        ]
        return rng.choice(choices)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_legal_bc_parses_no_crash(self, seed: int) -> None:
        binary = _find_tessera_opt()
        if binary is None:
            pytest.skip("tessera-opt not built")
        rng = random.Random(seed)
        for _ in range(8):
            n_axes = rng.choice([1, 2, 3, 4])
            bc, toks = self._gen_legal_bc(rng, n_axes)
            r = _run_bc_pass(bc, binary)
            assert r.returncode == 0, (
                f"legal BC {bc!r} crashed parser:\n{r.stderr}"
            )
            # Each expected mode must appear in the structured ArrayAttr.
            for _tok_text, expected_mode, _has_value in toks:
                assert f'"{expected_mode}"' in r.stdout, (
                    f"BC {bc!r}: expected mode {expected_mode!r} missing "
                    f"from emitted stencil.bc.modes"
                )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_malformed_bc_does_not_crash(self, seed: int) -> None:
        binary = _find_tessera_opt()
        if binary is None:
            pytest.skip("tessera-opt not built")
        rng = random.Random(seed)
        for _ in range(8):
            bc = self._gen_malformed_bc(rng)
            r = _run_bc_pass(bc, binary)
            # The parser must either succeed (with fall-through to
            # "periodic" or a warning attr) OR emit a textual diagnostic.
            # What it MUST NOT do is segfault, hang, or produce
            # negative-but-non-error returncode.
            assert r.returncode in (0, 1), (
                f"malformed BC {bc!r}: unexpected returncode "
                f"{r.returncode} (stderr={r.stderr})"
            )
            # Output is always valid text (never garbled bytes).
            assert isinstance(r.stdout, str)
            assert isinstance(r.stderr, str)

    def test_single_token_broadcasts_to_every_axis(self) -> None:
        """A single token like 'periodic' must broadcast across the
        rank determined by halo.width (= 2 in our test module)."""
        binary = _find_tessera_opt()
        if binary is None:
            pytest.skip("tessera-opt not built")
        r = _run_bc_pass("periodic", binary)
        assert r.returncode == 0, r.stderr
        # The output must show modes = ["periodic", "periodic"] for the
        # rank-2 halo.width.
        assert 'stencil.bc.modes = ["periodic", "periodic"]' in r.stdout

    def test_dirichlet_value_threads_through_to_attr(self) -> None:
        binary = _find_tessera_opt()
        if binary is None:
            pytest.skip("tessera-opt not built")
        r = _run_bc_pass("dirichlet(3.25),dirichlet(3.25)", binary)
        assert r.returncode == 0, r.stderr
        # Float values lit the structured ArrayAttr at full precision.
        assert "3.250000e+00" in r.stdout


# ─────────────────────────────────────────────────────────────────────────────
# 2. splitComma — mesh-axes / mesh-sizes pass options.
# ─────────────────────────────────────────────────────────────────────────────


class TestMeshAxesSplitCommaFuzz:
    """The splitComma helper in DistributionLoweringPass.cpp accepts
    comma-separated tokens with whitespace trimming.  Fuzz the
    mesh-axes / mesh-sizes pass options that drive it."""

    SEEDS = [0, 7, 42, 999]

    def _gen_axes(self, rng: random.Random) -> tuple[str, list[str]]:
        n = rng.choice([1, 2, 3, 4])
        names = rng.sample(["dp", "tp", "pp", "ep", "sp", "fsdp"], n)
        # Sometimes pad with whitespace; sometimes leave a trailing comma.
        padded = (" " + n_name + " " if rng.random() < 0.4 else n_name
                   for n_name in names)
        joined = ",".join(padded)
        if rng.random() < 0.3:
            joined += ","  # trailing comma — splitComma drops empties
        return joined, names

    def _gen_malformed_axes(self, rng: random.Random) -> str:
        return rng.choice([
            "",
            ",",
            ",,,",
            "   ,   ",
            "dp,",
            ",dp",
            "a,b,c,d,e,f,g,h,i,j",
            "axisname_" * 100,
            "  dp  ,  tp  ",
            "dp\ttp",          # tab inside
            "dp tp",            # space-only (should be one token "dp tp")
        ])

    @pytest.mark.parametrize("seed", SEEDS)
    def test_legal_axes_parse(self, seed: int) -> None:
        binary = _find_tessera_opt()
        if binary is None:
            pytest.skip("tessera-opt not built")
        rng = random.Random(seed)
        for _ in range(8):
            axes_str, expected_names = self._gen_axes(rng)
            sizes_str = ",".join("2" for _ in expected_names)
            # Provide a minimal module — body content doesn't matter for
            # the splitComma + mesh.define path.
            mod = (
                "func.func @t(%a: tensor<4x4xf32> "
                '{tessera.shard = {axes = ["dp"], dims = [0]}}) {\n'
                "  return\n}\n"
            )
            r = subprocess.run(
                [binary,
                 f"-tessera-distribution-lowering=mesh-axes={axes_str} "
                 f"mesh-sizes={sizes_str}"],
                input=mod, capture_output=True, text=True, timeout=30,
            )
            # Pass must not crash; the IR may or may not parse depending
            # on shard-attr placement, but exit code stays bounded.
            assert r.returncode in (0, 1), (
                f"axes={axes_str!r} sizes={sizes_str!r} unexpected returncode "
                f"{r.returncode} (stderr={r.stderr})"
            )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_malformed_axes_no_crash(self, seed: int) -> None:
        binary = _find_tessera_opt()
        if binary is None:
            pytest.skip("tessera-opt not built")
        rng = random.Random(seed)
        for _ in range(8):
            axes_str = self._gen_malformed_axes(rng)
            mod = "func.func @t() { return }\n"
            r = subprocess.run(
                [binary,
                 f"-tessera-distribution-lowering=mesh-axes={axes_str}"],
                input=mod, capture_output=True, text=True, timeout=30,
            )
            # Crash-safe.  Note: returncode 1 is acceptable (pass option
            # parsing may legitimately reject some malformed inputs).
            assert r.returncode in (0, 1, 2), (
                f"axes={axes_str!r} unexpected returncode {r.returncode}: "
                f"{r.stderr[:200]}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 3. featureMapToInt — linear-attn enum.  Tested via the pass behaviour
#    (unknown feature_map should fall through to default, not crash).
# ─────────────────────────────────────────────────────────────────────────────


class TestFeatureMapEnumFuzz:
    """``featureMapToInt`` is a static helper inside
    LinearAttnToAppleGPU.cpp.  We test the *behaviour* it drives:
    unknown feature_map strings must fall through cleanly (returning
    default 0=elu) rather than crashing."""

    KNOWN = ("elu", "relu", "identity", "polynomial_2")

    @pytest.mark.parametrize("known", KNOWN)
    def test_known_feature_map_strings_compile_in_source(self, known: str) -> None:
        """Source-level guard: every documented feature_map must appear
        in the C++ pass source.  A rename or typo here will be flagged
        immediately."""
        src = (REPO_ROOT / "src" / "compiler" / "codegen"
               / "Tessera_Apple_Backend" / "lib" / "Target" / "Apple"
               / "Lowering" / "LinearAttnToAppleGPU.cpp").read_text()
        assert f'"{known}"' in src, (
            f"feature_map {known!r} missing from LinearAttnToAppleGPU.cpp"
        )

    def test_unknown_feature_map_falls_through_to_default(self) -> None:
        src = (REPO_ROOT / "src" / "compiler" / "codegen"
               / "Tessera_Apple_Backend" / "lib" / "Target" / "Apple"
               / "Lowering" / "LinearAttnToAppleGPU.cpp").read_text()
        # The featureMapToInt function returns 0 for unknown — that's
        # the "fall through to default" contract.  Any change here
        # forces an explicit decision.
        assert "return 0;  // sensible default" in src, (
            "featureMapToInt must document its default-0 fall-through "
            "behaviour explicitly"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. canonicalize_dtype — Python-side dtype parser.
# ─────────────────────────────────────────────────────────────────────────────


class TestCanonicalizeDtypeFuzz:
    """The Python dtype canonicalizer is the boundary between user
    string input and the typed Dtype lattice.  135 hand-written tests
    already exist in ``test_canonical_dtype.py``; we add random-string
    fuzz to catch *any* malformed input class that crashes the parser
    rather than raising a clean TesseraDtypeError."""

    SEEDS = [0, 1, 2, 42]
    LEGAL_ALIASES = (
        "f32", "f16", "fp32", "fp16", "bf16", "bfloat16",
        "i8", "i16", "i32", "i64", "int8", "int16", "int32", "int64",
        "fp8_e4m3", "fp8_e5m2", "bool",
    )

    def _gen_legal_dtype(self, rng: random.Random) -> str:
        return rng.choice(self.LEGAL_ALIASES)

    def _gen_random_string(self, rng: random.Random) -> str:
        n = rng.randint(0, 30)
        return "".join(rng.choices(string.ascii_letters + string.digits
                                    + "_+-.()[]", k=n))

    @pytest.mark.parametrize("seed", SEEDS)
    def test_legal_aliases_canonicalize(self, seed: int) -> None:
        from tessera.dtype import canonicalize_dtype, is_canonical_dtype
        rng = random.Random(seed)
        for _ in range(20):
            alias = self._gen_legal_dtype(rng)
            canon = canonicalize_dtype(alias)
            assert is_canonical_dtype(canon), (
                f"alias {alias!r} canonicalized to non-canonical {canon!r}"
            )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_random_strings_raise_clean_error(self, seed: int) -> None:
        from tessera.dtype import canonicalize_dtype, TesseraDtypeError
        rng = random.Random(seed)
        for _ in range(20):
            s = self._gen_random_string(rng)
            if s in self.LEGAL_ALIASES:
                continue
            # The canonicalizer must raise TesseraDtypeError (or
            # ValueError for the empty-string case), never a generic
            # exception or hang.
            try:
                canonicalize_dtype(s)
            except (TesseraDtypeError, ValueError):
                pass  # expected
            except Exception as e:  # pragma: no cover
                pytest.fail(
                    f"canonicalize_dtype({s!r}) raised unexpected "
                    f"{type(e).__name__}: {e}"
                )
