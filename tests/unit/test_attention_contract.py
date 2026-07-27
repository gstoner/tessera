import numpy as np
import pytest

from tessera.compiler.attention_contract import (
    attention_dropout_replay,
    plan_attention_backward_workspace,
    reference_attention_backward_split_reduced,
    reference_streaming_attention,
)


def _fixture():
    rng = np.random.default_rng(20260726)
    q = rng.normal(0.0, 0.3, (2, 4, 5, 4)).astype(np.float32)
    key = rng.normal(0.0, 0.3, (2, 2, 7, 4)).astype(np.float32)
    value = rng.normal(0.0, 0.3, (2, 2, 7, 3)).astype(np.float32)
    do = rng.normal(0.0, 0.3, (2, 4, 5, 3)).astype(np.float32)
    bias = rng.normal(0.0, 0.1, (2, 4, 5, 7)).astype(np.float32)
    return do, q, key, value, bias


def _dense_forward(q, key, value, bias, *, dropout_p=0.0, dropout_seed=0):
    output = np.zeros((*q.shape[:3], value.shape[-1]), dtype=np.float32)
    replay = attention_dropout_replay(
        batch=q.shape[0],
        query_heads=q.shape[1],
        query_rows=q.shape[2],
        key_rows=key.shape[2],
        dropout_p=dropout_p,
        seed=dropout_seed,
    )
    for batch in range(q.shape[0]):
        for head in range(q.shape[1]):
            kv_head = head // (q.shape[1] // key.shape[1])
            raw = 0.5 * (q[batch, head] @ key[batch, kv_head].T)
            raw += bias[batch, head]
            scores = 2.0 * np.tanh(raw / 2.0)
            qpos = np.arange(q.shape[2])[:, None]
            kpos = np.arange(key.shape[2])[None, :]
            valid = (kpos <= qpos) & (kpos >= qpos - 3)
            scores = np.where(valid, scores, -np.inf)
            weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            weights = np.where(valid, weights, 0.0)
            weights /= np.sum(weights, axis=1, keepdims=True)
            output[batch, head] = (
                weights * replay[batch, head]
            ) @ value[batch, kv_head]
    return output


def _dense_backward(
    do, q, key, value, bias, *, dropout_p=0.0, dropout_seed=0
):
    dq = np.zeros_like(q)
    dk = np.zeros_like(key)
    dv = np.zeros_like(value)
    replay = attention_dropout_replay(
        batch=q.shape[0],
        query_heads=q.shape[1],
        query_rows=q.shape[2],
        key_rows=key.shape[2],
        dropout_p=dropout_p,
        seed=dropout_seed,
    )
    for batch in range(q.shape[0]):
        for head in range(q.shape[1]):
            kv_head = head // (q.shape[1] // key.shape[1])
            raw = 0.5 * (q[batch, head] @ key[batch, kv_head].T)
            raw += bias[batch, head]
            tanh = np.tanh(raw / 2.0)
            scores = 2.0 * tanh
            qpos = np.arange(q.shape[2])[:, None]
            kpos = np.arange(key.shape[2])[None, :]
            valid = (kpos <= qpos) & (kpos >= qpos - 3)
            scores = np.where(valid, scores, -np.inf)
            weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            weights = np.where(valid, weights, 0.0)
            weights /= np.sum(weights, axis=1, keepdims=True)
            output_weights = weights * replay[batch, head]
            output = output_weights @ value[batch, kv_head]
            dp = (
                do[batch, head] @ value[batch, kv_head].T
            ) * replay[batch, head]
            delta = np.sum(output * do[batch, head], axis=1, keepdims=True)
            ds = weights * (dp - delta) * (1.0 - tanh * tanh)
            dq[batch, head] = 0.5 * (ds @ key[batch, kv_head])
            dk[batch, kv_head] += 0.5 * (ds.T @ q[batch, head])
            dv[batch, kv_head] += output_weights.T @ do[batch, head]
    return dq, dk, dv


def test_rank4_streaming_recurrence_matches_dense_ragged_gqa_fixture() -> None:
    _, q, key, value, bias = _fixture()
    actual = reference_streaming_attention(
        q,
        key,
        value,
        block_size=3,
        scale=0.5,
        bias=bias,
        causal=True,
        window_left=3,
        window_right=0,
        softcap=2.0,
    )
    np.testing.assert_allclose(actual, _dense_forward(q, key, value, bias), atol=2e-6)


@pytest.mark.parametrize("split_count", [2, 3])
def test_split_reduced_backward_matches_dense_fixed_order_fixture(
    split_count: int,
) -> None:
    do, q, key, value, bias = _fixture()
    actual = reference_attention_backward_split_reduced(
        do,
        q,
        key,
        value,
        split_count=split_count,
        scale=0.5,
        bias=bias,
        causal=True,
        window_left=3,
        window_right=0,
        softcap=2.0,
    )
    expected = _dense_backward(do, q, key, value, bias)
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, reference, atol=2e-6)


def test_forward_and_backward_replay_identical_dropout_with_combined_modifiers() -> None:
    do, q, key, value, bias = _fixture()
    kwargs = {
        "scale": 0.5,
        "bias": bias,
        "causal": True,
        "window_left": 3,
        "window_right": 0,
        "softcap": 2.0,
        "dropout_p": 0.25,
        "dropout_seed": 37,
    }
    actual_forward = reference_streaming_attention(
        q, key, value, block_size=3, **kwargs
    )
    expected_forward = _dense_forward(
        q,
        key,
        value,
        bias,
        dropout_p=0.25,
        dropout_seed=37,
    )
    np.testing.assert_allclose(actual_forward, expected_forward, atol=2e-6)

    actual_backward = reference_attention_backward_split_reduced(
        do, q, key, value, split_count=3, **kwargs
    )
    expected_backward = _dense_backward(
        do,
        q,
        key,
        value,
        bias,
        dropout_p=0.25,
        dropout_seed=37,
    )
    for result, reference in zip(
        actual_backward, expected_backward, strict=True
    ):
        np.testing.assert_allclose(result, reference, atol=2e-6)


def test_dropout_replay_is_tile_and_split_independent() -> None:
    full = attention_dropout_replay(
        batch=2,
        query_heads=4,
        query_rows=5,
        key_rows=7,
        dropout_p=0.25,
        seed=37,
    )
    assert full.shape == (2, 4, 5, 7)
    assert set(np.unique(full)) <= {0.0, np.float32(4.0 / 3.0)}
    np.testing.assert_array_equal(
        full,
        attention_dropout_replay(
            batch=2,
            query_heads=4,
            query_rows=5,
            key_rows=7,
            dropout_p=0.25,
            seed=37,
        ),
    )


def test_backward_workspace_owns_slices_and_fixed_reduction_order() -> None:
    plan = plan_attention_backward_workspace(
        batch=1,
        query_heads=4,
        kv_heads=2,
        query_rows=17,
        key_rows=19,
        head_dim=64,
        value_dim=64,
        split_count=2,
    )

    assert plan.bytes == 37888
    assert plan.reduction_order == (0, 1)
    assert [item.name for item in plan.slices] == [
        "forward_o",
        "row_lse",
        "row_delta",
        "partial_dk",
        "partial_dv",
    ]
    assert all(item.offset % plan.alignment == 0 for item in plan.slices)
    assert plan.slices[-2].producer == "dkdv_split"
    assert plan.slices[-2].consumers == ("dkdv_fixed_order_reduce",)


def test_attention_contract_rejects_invalid_gqa_and_split_count() -> None:
    with pytest.raises(ValueError, match="divisible"):
        plan_attention_backward_workspace(
            batch=1,
            query_heads=3,
            kv_heads=2,
            query_rows=4,
            key_rows=4,
            head_dim=8,
            value_dim=8,
        )
    with pytest.raises(ValueError, match="at least two"):
        plan_attention_backward_workspace(
            batch=1,
            query_heads=2,
            kv_heads=1,
            query_rows=4,
            key_rows=4,
            head_dim=8,
            value_dim=8,
            split_count=1,
        )
