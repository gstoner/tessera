"""Tests for Phase H2 — LSTM (state-propagation primitive + Module).

Coverage:
  * `ops.lstm_cell` / `ops.lstm_state_h` / `ops.lstm_state_c` — single-step
    forward + numerical-Jacobian backward
  * `nn.LSTMCell` — Module wrapper, parameter init
  * `nn.LSTM` — multi-step unroll, BPTT through 3 timesteps
  * Phantom regression — verify LSTM is no longer a phantom
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts


# ─────────────────────────────────────────────────────────────────────────────
# ops.lstm_cell + state extractors
# ─────────────────────────────────────────────────────────────────────────────


class TestLSTMCellOp:
    def test_packed_output_shape(self):
        B, In, H = 2, 4, 3
        x = np.random.randn(B, In).astype(np.float32)
        h0 = np.zeros((B, H), dtype=np.float32)
        c0 = np.zeros((B, H), dtype=np.float32)
        W_ih = np.random.randn(4 * H, In).astype(np.float32)
        W_hh = np.random.randn(4 * H, H).astype(np.float32)
        packed = ts.ops.lstm_cell(x, h0, c0, W_ih, W_hh)
        assert packed.shape == (B, 2 * H)

    def test_state_extractors(self):
        B, H = 2, 3
        packed = np.arange(B * 2 * H, dtype=np.float32).reshape(B, 2 * H)
        h_t = ts.ops.lstm_state_h(packed)
        c_t = ts.ops.lstm_state_c(packed)
        assert h_t.shape == (B, H)
        assert c_t.shape == (B, H)
        np.testing.assert_array_equal(h_t, packed[..., :H])
        np.testing.assert_array_equal(c_t, packed[..., H:])

    def test_numerical_jacobian_one_step(self):
        np.random.seed(0)
        B, In, H = 1, 3, 2
        x = (np.random.randn(B, In) * 0.5).astype(np.float64)
        h0 = (np.random.randn(B, H) * 0.1).astype(np.float64)
        c0 = (np.random.randn(B, H) * 0.1).astype(np.float64)
        W_ih = (np.random.randn(4 * H, In) * 0.3).astype(np.float64)
        W_hh = (np.random.randn(4 * H, H) * 0.3).astype(np.float64)
        b_ih = (np.random.randn(4 * H) * 0.1).astype(np.float64)
        b_hh = (np.random.randn(4 * H) * 0.1).astype(np.float64)

        x_p = ts.nn.Parameter(x.copy())
        W_ih_p = ts.nn.Parameter(W_ih.copy())

        with ts.autodiff.tape() as t:
            packed = ts.ops.lstm_cell(x_p, h0, c0, W_ih_p, W_hh, b_ih, b_hh)
            h_t = ts.ops.lstm_state_h(packed)
            loss = ts.ops.reduce(ts.ops.mul(h_t, h_t), op="sum")
            t.backward(loss)
        analytic_dx = x_p.grad.numpy()
        analytic_dW = W_ih_p.grad.numpy()

        def loss_fn(x_v, W_v):
            packed = ts.ops.lstm_cell(x_v, h0, c0, W_v, W_hh, b_ih, b_hh)
            return float((packed[..., :H] ** 2).sum())

        eps = 1e-6
        # spot-check dx[0, 1]
        x_pos = x.copy(); x_pos[0, 1] += eps
        x_neg = x.copy(); x_neg[0, 1] -= eps
        num_dx = (loss_fn(x_pos, W_ih) - loss_fn(x_neg, W_ih)) / (2 * eps)
        assert abs(analytic_dx[0, 1] - num_dx) < 1e-6

        # spot-check dW_ih[2, 1]
        W_pos = W_ih.copy(); W_pos[2, 1] += eps
        W_neg = W_ih.copy(); W_neg[2, 1] -= eps
        num_dW = (loss_fn(x, W_pos) - loss_fn(x, W_neg)) / (2 * eps)
        assert abs(analytic_dW[2, 1] - num_dW) < 1e-6

    def test_bptt_two_steps_matches_numerical(self):
        np.random.seed(0)
        B, In, H = 1, 2, 3
        x1 = (np.random.randn(B, In) * 0.5).astype(np.float64)
        x2 = (np.random.randn(B, In) * 0.5).astype(np.float64)
        h0 = (np.random.randn(B, H) * 0.1).astype(np.float64)
        c0 = (np.random.randn(B, H) * 0.1).astype(np.float64)
        W_ih = (np.random.randn(4 * H, In) * 0.3).astype(np.float64)
        W_hh = (np.random.randn(4 * H, H) * 0.3).astype(np.float64)
        b_ih = (np.random.randn(4 * H) * 0.1).astype(np.float64)
        b_hh = (np.random.randn(4 * H) * 0.1).astype(np.float64)

        W_p = ts.nn.Parameter(W_ih.copy())
        with ts.autodiff.tape() as t:
            p1 = ts.ops.lstm_cell(x1, h0, c0, W_p, W_hh, b_ih, b_hh)
            h1 = ts.ops.lstm_state_h(p1)
            c1 = ts.ops.lstm_state_c(p1)
            p2 = ts.ops.lstm_cell(x2, h1, c1, W_p, W_hh, b_ih, b_hh)
            h2 = ts.ops.lstm_state_h(p2)
            loss = ts.ops.reduce(ts.ops.mul(h2, h2), op="sum")
            t.backward(loss)
        analytic = W_p.grad.numpy()

        def loss_fn(W_val):
            p1 = ts.ops.lstm_cell(x1, h0, c0, W_val, W_hh, b_ih, b_hh)
            h1 = p1[..., :H]
            c1 = p1[..., H:]
            p2 = ts.ops.lstm_cell(x2, h1, c1, W_val, W_hh, b_ih, b_hh)
            return float((p2[..., :H] ** 2).sum())

        eps = 1e-6
        Wp = W_ih.copy(); Wp[2, 1] += eps
        Wn = W_ih.copy(); Wn[2, 1] -= eps
        num = (loss_fn(Wp) - loss_fn(Wn)) / (2 * eps)
        assert abs(analytic[2, 1] - num) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# nn.LSTMCell + nn.LSTM
# ─────────────────────────────────────────────────────────────────────────────


class TestLSTMCellModule:
    def test_construction_param_shapes(self):
        cell = ts.nn.LSTMCell(input_size=4, hidden_size=8)
        names = sorted(n for n, _ in cell.named_parameters())
        assert names == ["W_hh", "W_ih", "b_hh", "b_ih"]
        assert cell.W_ih.shape == (32, 4)
        assert cell.W_hh.shape == (32, 8)
        assert cell.b_ih.shape == (32,)

    def test_no_bias(self):
        cell = ts.nn.LSTMCell(input_size=4, hidden_size=8, bias=False)
        names = sorted(n for n, _ in cell.named_parameters())
        assert names == ["W_hh", "W_ih"]

    def test_forward_shapes(self):
        cell = ts.nn.LSTMCell(input_size=4, hidden_size=8)
        B = 2
        x = np.random.randn(B, 4).astype(np.float32)
        h = np.zeros((B, 8), dtype=np.float32)
        c = np.zeros((B, 8), dtype=np.float32)
        h_new, c_new = cell(x, (h, c))
        assert h_new.shape == (B, 8)
        assert c_new.shape == (B, 8)


class TestLSTMModule:
    def test_forward_shapes(self):
        lstm = ts.nn.LSTM(input_size=4, hidden_size=8)
        x_seq = np.random.randn(2, 5, 4).astype(np.float32)
        out, (h_n, c_n) = lstm(x_seq)
        assert out.shape == (2, 5, 8)
        assert h_n.shape == (2, 8)
        assert c_n.shape == (2, 8)

    def test_invalid_input_rank(self):
        lstm = ts.nn.LSTM(input_size=4, hidden_size=8)
        with pytest.raises(ValueError, match=r"\(B, T, input_size\)"):
            lstm(np.zeros((4, 4)))

    def test_state_dict_roundtrip(self):
        l1 = ts.nn.LSTM(input_size=4, hidden_size=8)
        l2 = ts.nn.LSTM(input_size=4, hidden_size=8)
        l2.load_state_dict(l1.state_dict())
        x = np.random.randn(1, 3, 4).astype(np.float32)
        out1, _ = l1(x)
        out2, _ = l2(x)
        np.testing.assert_allclose(out1, out2)

    def test_bptt_populates_grads(self):
        np.random.seed(0)
        lstm = ts.nn.LSTM(input_size=2, hidden_size=3)
        x_seq = np.random.randn(1, 3, 2).astype(np.float32)
        with ts.autodiff.tape() as t:
            _, (h_n, c_n) = lstm(x_seq)
            loss = ts.ops.reduce(ts.ops.mul(h_n, h_n), op="sum")
            t.backward(loss)
        for n, p in lstm.named_parameters():
            assert p.grad is not None, f"no grad for {n}"
            assert p.grad.shape == p.shape


class TestNoMorePhantoms:
    def test_lstm_is_not_phantom(self):
        # Constructing LSTM/LSTMCell should not raise NotImplementedError
        ts.nn.LSTM(input_size=2, hidden_size=4)
        ts.nn.LSTMCell(input_size=2, hidden_size=4)
