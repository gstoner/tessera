"""
Phase 2 test fixtures.

These tests verify that the Python graph_ir.py emitter produces MLIR text
that structurally matches the Phase 2 lowering chain expectations.

For full C++ pass testing, use the MLIR lit tests in
tests/tessera-ir/phase2/.  The Python tests here focus on:
  1. Graph IR emission format (shard attrs, effect attrs)
  2. Effect inference from @jit decorator
  3. End-to-end graph_ir text round-trip structure
"""
import pytest
import sys
import os

# Make sure the tessera package is importable from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import tessera
from tessera.compiler.graph_ir import GraphIRBuilder, GraphIRModule


@pytest.fixture
def builder():
    """A fresh GraphIRBuilder per test."""
    return GraphIRBuilder()


@pytest.fixture
def simple_gemm_ir(builder):
    """A pre-built GEMM GraphIRModule for reuse."""
    from tessera.distributed.region import Region

    def step(W: Region["read"], X: Region["read"], Y: Region["write"]):
        Y[:] = tessera.ops.gemm(X, W)

    builder.lower(step)
    return builder.module()
