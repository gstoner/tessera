"""Phase-2 faithful Block AttnRes recurrence and two-phase oracle."""

from __future__ import annotations

import numpy as np
import pytest

from tessera.stdlib import attn_res


def _program(seed: int, layers: int, width: int):
    rng = np.random.default_rng(seed)
    matrices = tuple(
        (rng.normal(size=(width, width)) / np.sqrt(width)).astype(np.float32)
        for _ in range(layers)
    )
    sublayers = tuple(
        (lambda hidden, matrix=matrix: np.tanh(hidden @ matrix))
        for matrix in matrices
    )
    norms = tuple(
        lambda hidden: hidden / np.sqrt(
            np.mean(hidden * hidden, axis=-1, keepdims=True) + 1.0e-5
        )
        for _ in range(layers)
    )
    queries = tuple(
        rng.normal(size=(width,)).astype(np.float32) * 0.25
        for _ in range(layers + 1)
    )
    return rng, sublayers, norms, queries


@pytest.mark.parametrize("block_sizes", [(2, 2, 2), (3, 3), (1, 2, 3), (6,)])
def test_two_phase_matches_direct_for_batched_token_state(block_sizes) -> None:
    rng, sublayers, norms, queries = _program(20, sum(block_sizes), 19)
    x = rng.normal(size=(2, 3, 19)).astype(np.float32)
    direct = attn_res.block_attnres_forward(
        x, sublayers, queries, block_sizes=block_sizes, norms=norms
    )
    two_phase = attn_res.block_attnres_two_phase(
        x, sublayers, queries, block_sizes=block_sizes, norms=norms
    )
    np.testing.assert_allclose(two_phase, direct, rtol=4e-6, atol=4e-6)
    assert direct.dtype == two_phase.dtype == np.float32


def test_balanced_blocks_and_zero_query_initialization() -> None:
    assert attn_res.balanced_block_sizes(10, 6) == (2, 2, 2, 2, 1, 1)
    queries = attn_res.zero_queries(4, 8)
    assert len(queries) == 5
    assert all(query.dtype == np.float32 for query in queries)
    assert all(not np.any(query) for query in queries)


def test_zero_queries_use_attention_average_not_residual_sum() -> None:
    x = np.array([1.0, 3.0], dtype=np.float32)
    # Two size-one blocks produce b1=x and b2=mean(x,b1)=x. The final zero
    # query averages (b0,b1,b2); it never sums them to 3*x.
    sublayers = (lambda hidden: hidden, lambda hidden: hidden)
    output = attn_res.block_attnres_forward(
        x, sublayers, attn_res.zero_queries(2, 2), block_sizes=(1, 1)
    )
    np.testing.assert_allclose(output, x)
    assert not np.allclose(output, 3.0 * x)


def test_first_layer_of_each_block_excludes_partial(monkeypatch) -> None:
    counts: list[int] = []
    original = attn_res.depth_attn

    def counted(query, sources, *, eps=1.0e-6):
        counts.append(np.asarray(sources).shape[0])
        return original(query, sources, eps=eps)

    monkeypatch.setattr(attn_res, "depth_attn", counted)
    attn_res.block_attnres_forward(
        np.ones((5,), np.float32),
        (lambda hidden: hidden,) * 4,
        attn_res.zero_queries(4, 5),
        block_sizes=(2, 2),
    )
    assert counts == [1, 2, 2, 3, 3]


@pytest.mark.parametrize(
    "call,match",
    [
        (lambda: attn_res.balanced_block_sizes(3, 4), "num_blocks"),
        (lambda: attn_res.zero_queries(0, 4), "positive"),
        (lambda: attn_res.block_attnres_forward(
            np.ones(4), (lambda x: x,), attn_res.zero_queries(1, 4),
            num_blocks=1, block_sizes=(1,),
        ), "exactly one"),
        (lambda: attn_res.block_attnres_forward(
            np.ones(4), (lambda x: x,), (np.zeros(3), np.zeros(3)),
            block_sizes=(1,),
        ), "shape"),
    ],
)
def test_phase_two_contracts_fail_closed(call, match) -> None:
    with pytest.raises(ValueError, match=match):
        call()
