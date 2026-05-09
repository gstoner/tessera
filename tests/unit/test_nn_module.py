"""Tests for the Tier 1 stateful `tessera.nn` surface.

Covers:
  * `Parameter` construction, lifecycle, grad slot
  * `Module` attribute routing, parameter iteration, state_dict round-trip,
    train()/eval(), repr
  * `Sequential`, `ModuleList`, `ModuleDict` containers
  * Each layer (`Linear`, `RMSNorm`, `LayerNorm`, `Embedding`, `Dropout`,
    `MLP`, `MultiHeadAttention`) — construction, forward shape, deferred-mode
    semantics where relevant
  * Composition smoke test (mock transformer block)
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts


# ─────────────────────────────────────────────────────────────────────────────
# Parameter
# ─────────────────────────────────────────────────────────────────────────────


class TestParameter:
    def test_construct_from_shape(self):
        p = ts.nn.Parameter(shape=(4, 8), dtype="fp32")
        assert p.shape == (4, 8)
        assert p.dtype == "fp32"
        assert p.requires_grad is True
        assert p.grad is None
        assert p.numpy().shape == (4, 8)

    def test_construct_from_shape_tuple(self):
        p = ts.nn.Parameter((3, 5))
        assert p.shape == (3, 5)
        assert (p.numpy() == 0).all()

    def test_construct_from_numpy(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        p = ts.nn.Parameter(arr)
        assert p.shape == (3, 4)
        assert np.allclose(p.numpy(), arr)

    def test_construct_from_distributed_array(self):
        d = ts.array.from_domain(ts.domain.Rect((2, 3)), dtype="fp32", distribution=ts.dist.Replicated())
        p = ts.nn.Parameter(d)
        assert p.shape == (2, 3)
        assert p.data is d  # holds the same handle

    def test_grad_setter_accepts_numpy(self):
        p = ts.nn.Parameter(shape=(4,))
        g = np.ones((4,), dtype=np.float32) * 2.5
        p.grad = g
        assert p.grad is not None
        assert np.allclose(p.grad.numpy(), 2.5)

    def test_zero_grad_clears(self):
        p = ts.nn.Parameter(shape=(4,))
        p.grad = np.ones(4, dtype=np.float32)
        p.zero_grad()
        assert p.grad is None

    def test_array_protocol(self):
        p = ts.nn.Parameter(np.arange(6, dtype=np.float32).reshape(2, 3))
        assert np.allclose(np.asarray(p), p.numpy())

    def test_repr(self):
        p = ts.nn.Parameter(shape=(2, 3))
        s = repr(p)
        assert "Parameter" in s and "(2, 3)" in s

    def test_invalid_construction_raises(self):
        with pytest.raises(TypeError):
            ts.nn.Parameter("not a tensor")


# ─────────────────────────────────────────────────────────────────────────────
# Module
# ─────────────────────────────────────────────────────────────────────────────


class TestModule:
    def test_subclass_must_call_super_init(self):
        class Bad(ts.nn.Module):
            def __init__(self):
                # Missing super().__init__() on purpose
                pass

        b = Bad()
        with pytest.raises(RuntimeError, match=r"super\(\)\.__init__"):
            b.weight = ts.nn.Parameter(shape=(2,))

    def test_setattr_routes_parameter(self):
        m = ts.nn.Module()
        m.w = ts.nn.Parameter(shape=(4,))
        assert "w" in m._parameters
        assert "w" not in m.__dict__
        assert m.w is m._parameters["w"]

    def test_setattr_routes_submodule(self):
        m = ts.nn.Module()
        child = ts.nn.Linear(4, 8)
        m.lin = child
        assert "lin" in m._modules
        assert m.lin is child

    def test_setattr_normal_attribute(self):
        m = ts.nn.Module()
        m.layer_count = 7
        assert m.layer_count == 7
        assert "layer_count" not in m._parameters
        assert "layer_count" not in m._modules

    def test_overwrite_parameter_with_non_parameter(self):
        m = ts.nn.Module()
        m.w = ts.nn.Parameter(shape=(2,))
        m.w = 42  # Replacing a Parameter with a regular attr should remove it from _parameters
        assert "w" not in m._parameters
        assert m.w == 42

    def test_named_parameters_includes_children(self):
        m = ts.nn.Module()
        m.lin1 = ts.nn.Linear(4, 8, bias=True)
        m.lin2 = ts.nn.Linear(8, 4, bias=False)
        names = sorted(n for n, _ in m.named_parameters())
        assert names == ["lin1.bias", "lin1.weight", "lin2.weight"]

    def test_named_parameters_no_recurse(self):
        m = ts.nn.Module()
        m.lin = ts.nn.Linear(4, 8)
        m.w = ts.nn.Parameter(shape=(3,))
        own = sorted(n for n, _ in m.named_parameters(recurse=False))
        assert own == ["w"]

    def test_state_dict_roundtrip(self):
        m1 = ts.nn.Linear(4, 8)
        sd = m1.state_dict()
        m2 = ts.nn.Linear(4, 8)
        m2.load_state_dict(sd)
        x = np.random.randn(2, 4).astype(np.float32)
        assert np.allclose(m1(x), m2(x))

    def test_load_state_dict_strict_missing_key(self):
        m = ts.nn.Linear(4, 8)
        sd = m.state_dict()
        del sd["weight"]
        with pytest.raises(KeyError, match="missing"):
            m.load_state_dict(sd, strict=True)

    def test_load_state_dict_strict_extra_key(self):
        m = ts.nn.Linear(4, 8)
        sd = m.state_dict()
        sd["bogus"] = np.zeros(2)
        with pytest.raises(KeyError, match="unexpected"):
            m.load_state_dict(sd, strict=True)

    def test_load_state_dict_shape_mismatch(self):
        m = ts.nn.Linear(4, 8)
        sd = m.state_dict()
        sd["weight"] = np.zeros((9, 9), dtype=np.float32)
        with pytest.raises(ValueError, match="shape mismatch"):
            m.load_state_dict(sd)

    def test_train_eval_recurses(self):
        m = ts.nn.Module()
        m.drop = ts.nn.Dropout(p=0.1)
        m.lin = ts.nn.Linear(4, 4)
        m.eval()
        assert m.training is False
        assert m.drop.training is False
        assert m.lin.training is False
        m.train()
        assert m.training is True
        assert m.drop.training is True

    def test_zero_grad_clears_all(self):
        m = ts.nn.Linear(4, 4)
        for p in m.parameters():
            p.grad = np.ones(p.shape, dtype=np.float32)
        m.zero_grad()
        assert all(p.grad is None for p in m.parameters())

    def test_repr_shows_tree(self):
        m = ts.nn.Sequential(ts.nn.Linear(4, 8), ts.nn.RMSNorm(8))
        s = repr(m)
        assert "Sequential" in s
        assert "Linear" in s
        assert "RMSNorm" in s

    def test_modules_iter_includes_self(self):
        m = ts.nn.Module()
        m.lin = ts.nn.Linear(4, 4)
        all_mods = list(m.modules())
        assert all_mods[0] is m
        assert m.lin in all_mods

    def test_forward_not_implemented_by_default(self):
        m = ts.nn.Module()
        with pytest.raises(NotImplementedError):
            m()


# ─────────────────────────────────────────────────────────────────────────────
# Containers
# ─────────────────────────────────────────────────────────────────────────────


class TestSequential:
    def test_forward_chains(self):
        seq = ts.nn.Sequential(ts.nn.Linear(4, 8), ts.nn.Linear(8, 2))
        x = np.random.randn(3, 4).astype(np.float32)
        assert seq(x).shape == (3, 2)

    def test_indexing(self):
        seq = ts.nn.Sequential(ts.nn.Linear(4, 8), ts.nn.RMSNorm(8))
        assert isinstance(seq[0], ts.nn.Linear)
        assert isinstance(seq[-1], ts.nn.RMSNorm)
        assert len(seq) == 2

    def test_rejects_non_module(self):
        with pytest.raises(TypeError):
            ts.nn.Sequential(ts.nn.Linear(4, 8), "not a module")


class TestModuleList:
    def test_append_and_iter(self):
        ml = ts.nn.ModuleList()
        ml.append(ts.nn.Linear(4, 8))
        ml.append(ts.nn.Linear(8, 4))
        assert len(ml) == 2
        items = list(ml)
        assert items[0].in_features == 4
        assert items[1].out_features == 4

    def test_init_from_list(self):
        ml = ts.nn.ModuleList([ts.nn.Linear(4, 8), ts.nn.Linear(8, 4)])
        assert len(ml) == 2

    def test_extend(self):
        ml = ts.nn.ModuleList([ts.nn.Linear(4, 8)])
        ml.extend([ts.nn.Linear(8, 4), ts.nn.Linear(4, 2)])
        assert len(ml) == 3


class TestModuleDict:
    def test_basic_dict_ops(self):
        md = ts.nn.ModuleDict({"q": ts.nn.Linear(4, 8), "k": ts.nn.Linear(4, 8)})
        assert "q" in md
        assert len(md) == 2
        md["v"] = ts.nn.Linear(4, 8)
        assert sorted(md.keys()) == ["k", "q", "v"]
        del md["v"]
        assert "v" not in md


# ─────────────────────────────────────────────────────────────────────────────
# Layers
# ─────────────────────────────────────────────────────────────────────────────


class TestLinear:
    def test_forward_shape_with_bias(self):
        m = ts.nn.Linear(4, 8)
        x = np.random.randn(3, 4).astype(np.float32)
        y = m(x)
        assert y.shape == (3, 8)
        names = {n for n, _ in m.named_parameters()}
        assert names == {"weight", "bias"}

    def test_forward_shape_without_bias(self):
        m = ts.nn.Linear(4, 8, bias=False)
        x = np.random.randn(3, 4).astype(np.float32)
        y = m(x)
        assert y.shape == (3, 8)
        assert m.bias is None
        names = {n for n, _ in m.named_parameters()}
        assert names == {"weight"}

    def test_forward_higher_rank(self):
        m = ts.nn.Linear(8, 16)
        x = np.random.randn(2, 4, 8).astype(np.float32)
        assert m(x).shape == (2, 4, 16)


class TestRMSNorm:
    def test_forward_shape(self):
        m = ts.nn.RMSNorm(8)
        x = np.random.randn(2, 4, 8).astype(np.float32)
        assert m(x).shape == (2, 4, 8)

    def test_default_weight_is_ones(self):
        m = ts.nn.RMSNorm(8)
        assert np.allclose(m.weight.numpy(), 1.0)


class TestLayerNorm:
    def test_forward_shape_default(self):
        m = ts.nn.LayerNorm(8)
        x = np.random.randn(2, 4, 8).astype(np.float32)
        assert m(x).shape == (2, 4, 8)

    def test_no_affine(self):
        m = ts.nn.LayerNorm(8, elementwise_affine=False)
        x = np.random.randn(2, 4, 8).astype(np.float32)
        y = m(x)
        # When non-affine, output mean ~ 0, var ~ 1 over the last axis
        assert abs(y.mean()) < 1e-3
        # No parameters
        assert list(m.named_parameters()) == []

    def test_no_bias(self):
        m = ts.nn.LayerNorm(8, bias=False)
        names = {n for n, _ in m.named_parameters()}
        assert names == {"weight"}


class TestEmbedding:
    def test_lookup_shape(self):
        e = ts.nn.Embedding(100, 16)
        idx = np.array([[1, 2, 3], [4, 5, 6]])
        assert e(idx).shape == (2, 3, 16)

    def test_index_out_of_range(self):
        e = ts.nn.Embedding(10, 4)
        with pytest.raises(IndexError):
            e(np.array([10]))

    def test_negative_index_rejected(self):
        e = ts.nn.Embedding(10, 4)
        with pytest.raises(IndexError):
            e(np.array([-1]))


class TestDropout:
    def test_eval_mode_is_identity(self):
        m = ts.nn.Dropout(p=0.5).eval()
        x = np.random.randn(4, 8).astype(np.float32)
        y = m(x)
        assert np.allclose(y, x)

    def test_train_mode_zeros_some(self):
        m = ts.nn.Dropout(p=0.5, seed=42).train()
        x = np.ones((1, 1024), dtype=np.float32)
        y = m(x)
        # Roughly half of entries are zeroed under p=0.5
        zero_frac = (y == 0).mean()
        assert 0.3 < zero_frac < 0.7


class TestMLP:
    def test_forward_shape(self):
        m = ts.nn.MLP(dim=16, hidden_dim=32)
        x = np.random.randn(2, 4, 16).astype(np.float32)
        assert m(x).shape == (2, 4, 16)

    def test_parameter_names(self):
        m = ts.nn.MLP(dim=16, hidden_dim=32)
        names = sorted(n for n, _ in m.named_parameters())
        assert names == ["W_down", "W_gate", "W_up"]


class TestMultiHeadAttention:
    def test_self_attention_shape(self):
        m = ts.nn.MultiHeadAttention(embed_dim=32, num_heads=4)
        x = np.random.randn(2, 8, 32).astype(np.float32)
        assert m(x).shape == (2, 8, 32)

    def test_cross_attention_different_seq_len(self):
        m = ts.nn.MultiHeadAttention(embed_dim=32, num_heads=4)
        q = np.random.randn(2, 8, 32).astype(np.float32)
        kv = np.random.randn(2, 12, 32).astype(np.float32)
        assert m(q, kv).shape == (2, 8, 32)

    def test_invalid_head_dim_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            ts.nn.MultiHeadAttention(embed_dim=33, num_heads=4)

    def test_dropout_gated_by_training(self):
        m = ts.nn.MultiHeadAttention(embed_dim=32, num_heads=4, dropout_p=0.5)
        x = np.random.randn(2, 4, 32).astype(np.float32)
        m.eval()
        y_eval_a = m(x)
        y_eval_b = m(x)
        assert np.allclose(y_eval_a, y_eval_b)  # deterministic in eval


# ─────────────────────────────────────────────────────────────────────────────
# Composition smoke test — build a mock transformer block end-to-end.
# ─────────────────────────────────────────────────────────────────────────────


class TransformerBlock(ts.nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_hidden: int):
        super().__init__()
        self.norm1 = ts.nn.RMSNorm(dim)
        self.attn = ts.nn.MultiHeadAttention(embed_dim=dim, num_heads=num_heads)
        self.norm2 = ts.nn.RMSNorm(dim)
        self.mlp = ts.nn.MLP(dim=dim, hidden_dim=mlp_hidden)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TestComposition:
    def test_transformer_block_forward(self):
        block = TransformerBlock(dim=32, num_heads=4, mlp_hidden=64)
        x = np.random.randn(2, 8, 32).astype(np.float32)
        y = block(x)
        assert y.shape == (2, 8, 32)

    def test_transformer_block_named_parameters(self):
        block = TransformerBlock(dim=32, num_heads=4, mlp_hidden=64)
        names = sorted(n for n, _ in block.named_parameters())
        # 8 expected: 2 norms + 4 MHA linears (each w+b) + 3 MLP weights
        # = 2 + 8 + 3 = 13
        assert len(names) == 13
        assert all(n.startswith(("norm1.", "attn.", "norm2.", "mlp.")) for n in names)

    def test_transformer_block_state_dict_roundtrip(self):
        b1 = TransformerBlock(dim=16, num_heads=2, mlp_hidden=32)
        b2 = TransformerBlock(dim=16, num_heads=2, mlp_hidden=32)
        b2.load_state_dict(b1.state_dict())
        x = np.random.randn(1, 4, 16).astype(np.float32)
        assert np.allclose(b1(x), b2(x))


# ─────────────────────────────────────────────────────────────────────────────
# Phantom-stub regression tests — ensure the not-yet-implemented surface still
# fails with a clear NotImplementedError message after the Tier 1 lift.
# ─────────────────────────────────────────────────────────────────────────────


class TestRemainingPhantoms:
    """No phantoms remain after Phase H2 — all `tessera.nn.*` names ship as
    real classes. This test asserts that empty state. If a new phantom is
    introduced later, add it to a parametrized check here.
    """

    def test_no_phantoms_remain(self):
        # Iterate the public surface; every callable should be a real class.
        public = [n for n in dir(ts.nn) if not n.startswith("_")]
        for name in public:
            obj = getattr(ts.nn, name)
            if isinstance(obj, type):
                # Real Modules instantiate without raising NotImplementedError
                # (with appropriate args). We don't try to construct here —
                # just verify the class doesn't carry our phantom marker text.
                assert "Tier 1+ backlog" not in (obj.__init__.__doc__ or ""), (
                    f"phantom still in tessera.nn: {name}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# Phase B1 — Buffer protocol
# ─────────────────────────────────────────────────────────────────────────────


class TestBuffer:
    def test_construct_from_numpy(self):
        b = ts.nn.Buffer(np.arange(6, dtype=np.float32).reshape(2, 3))
        assert b.shape == (2, 3)
        assert b.persistent is True
        assert np.allclose(b.numpy(), np.arange(6).reshape(2, 3))

    def test_construct_non_persistent(self):
        b = ts.nn.Buffer(shape=(4,), persistent=False)
        assert b.persistent is False

    def test_construct_from_shape(self):
        b = ts.nn.Buffer(shape=(2, 3), dtype="fp32")
        assert b.shape == (2, 3)
        assert (b.numpy() == 0).all()

    def test_construct_from_distributed_array(self):
        d = ts.array.from_domain(
            ts.domain.Rect((4,)), dtype="fp32", distribution=ts.dist.Replicated()
        )
        b = ts.nn.Buffer(d)
        assert b.data is d

    def test_no_grad_attribute(self):
        b = ts.nn.Buffer(shape=(4,))
        # Buffers must not expose the gradient surface
        assert not hasattr(b, "grad")
        assert not hasattr(b, "requires_grad")

    def test_array_protocol(self):
        b = ts.nn.Buffer(np.arange(4, dtype=np.float32))
        assert np.allclose(np.asarray(b), b.numpy())

    def test_invalid_construction_raises(self):
        with pytest.raises(TypeError):
            ts.nn.Buffer("not a tensor")
        with pytest.raises(TypeError):
            ts.nn.Buffer(None)


class TestModuleBufferIntegration:
    def test_register_buffer_routes_to_buffers_dict(self):
        m = ts.nn.Module()
        m.register_buffer("running_mean", np.zeros(4, dtype=np.float32))
        assert "running_mean" in m._buffers
        assert "running_mean" not in m._parameters
        assert "running_mean" not in m._modules
        assert isinstance(m.running_mean, ts.nn.Buffer)

    def test_register_buffer_accepts_existing_buffer_instance(self):
        m = ts.nn.Module()
        b = ts.nn.Buffer(shape=(3,))
        m.register_buffer("b", b)
        assert m.b is b

    def test_register_buffer_persistent_override(self):
        m = ts.nn.Module()
        b = ts.nn.Buffer(shape=(3,), persistent=True)
        m.register_buffer("b", b, persistent=False)
        assert m.b.persistent is False  # explicit kwarg wins

    def test_register_buffer_rejects_none(self):
        m = ts.nn.Module()
        with pytest.raises(TypeError):
            m.register_buffer("nope", None)

    def test_setattr_buffer_routes_correctly(self):
        m = ts.nn.Module()
        m.b = ts.nn.Buffer(shape=(2,))
        assert "b" in m._buffers
        assert "b" not in m._parameters

    def test_buffer_overwrite_with_parameter(self):
        m = ts.nn.Module()
        m.x = ts.nn.Buffer(shape=(2,))
        m.x = ts.nn.Parameter(shape=(2,))
        assert "x" not in m._buffers
        assert "x" in m._parameters

    def test_named_buffers_recurse(self):
        m = ts.nn.Module()
        m.register_buffer("running_mean", np.zeros(4, dtype=np.float32))

        class Inner(ts.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("inner_buf", np.ones(2, dtype=np.float32))

        m.inner = Inner()
        names = sorted(n for n, _ in m.named_buffers())
        assert names == ["inner.inner_buf", "running_mean"]

    def test_parameters_does_not_yield_buffers(self):
        m = ts.nn.Module()
        m.w = ts.nn.Parameter(shape=(4,))
        m.register_buffer("running_mean", np.zeros(4, dtype=np.float32))
        params = list(m.parameters())
        assert len(params) == 1
        assert all(isinstance(p, ts.nn.Parameter) for p in params)

    def test_state_dict_includes_persistent_buffers(self):
        m = ts.nn.Module()
        m.w = ts.nn.Parameter(shape=(4,))
        m.register_buffer("running_mean", np.zeros(4, dtype=np.float32))
        sd = m.state_dict()
        assert set(sd.keys()) == {"w", "running_mean"}

    def test_state_dict_excludes_non_persistent_buffers(self):
        m = ts.nn.Module()
        m.w = ts.nn.Parameter(shape=(4,))
        m.register_buffer("transient", np.zeros(4, dtype=np.float32), persistent=False)
        sd = m.state_dict()
        assert "transient" not in sd

    def test_load_state_dict_restores_buffers(self):
        m1 = ts.nn.Module()
        m1.register_buffer("rm", np.array([1.0, 2.0, 3.0], dtype=np.float32))
        sd = m1.state_dict()

        m2 = ts.nn.Module()
        m2.register_buffer("rm", np.zeros(3, dtype=np.float32))
        m2.load_state_dict(sd)
        np.testing.assert_allclose(m2.rm.numpy(), [1.0, 2.0, 3.0])

    def test_load_state_dict_strict_with_buffer_keys(self):
        m = ts.nn.Module()
        m.register_buffer("rm", np.zeros(3, dtype=np.float32))
        # Strict mode rejects extra keys
        with pytest.raises(KeyError, match="unexpected"):
            m.load_state_dict({"rm": np.zeros(3), "extra": np.zeros(2)})

    def test_zero_grad_does_not_touch_buffers(self):
        m = ts.nn.Linear(4, 4)
        m.register_buffer("rm", np.ones(4, dtype=np.float32))
        for p in m.parameters():
            p.grad = np.ones(p.shape, dtype=np.float32)
        m.zero_grad()
        # All grads cleared, buffer untouched
        assert all(p.grad is None for p in m.parameters())
        np.testing.assert_allclose(m.rm.numpy(), 1.0)

    def test_repr_shows_buffers(self):
        m = ts.nn.Module()
        m.register_buffer("rm", np.zeros(4, dtype=np.float32))
        s = repr(m)
        assert "Buffer" in s
        assert "persistent=True" in s


# ─────────────────────────────────────────────────────────────────────────────
# Phase B3 — Module.to(dtype) dtype migration
# ─────────────────────────────────────────────────────────────────────────────


class TestModuleToDtype:
    def test_migrate_parameters(self):
        m = ts.nn.Linear(4, 8)
        assert m.weight.dtype == "fp32"
        m.to("fp16")
        assert m.weight.dtype == "fp16"
        assert m.weight.numpy().dtype == np.float16
        assert m.bias.dtype == "fp16"

    def test_migrate_persistent_buffers(self):
        m = ts.nn.Module()
        m.register_buffer("mask", np.ones((4, 4), dtype=np.float32))
        m.to("fp16")
        assert m.mask.dtype == "fp16"
        assert m.mask.numpy().dtype == np.float16

    def test_skips_non_persistent_buffers(self):
        m = ts.nn.Module()
        m.register_buffer("temp", np.ones(4, dtype=np.float32), persistent=False)
        m.to("fp16")
        # Non-persistent buffers are runtime scratch — leave them alone
        assert m.temp.dtype == "fp32"
        assert m.temp.numpy().dtype == np.float32

    def test_round_trip(self):
        m = ts.nn.Linear(4, 8)
        original = m.weight.numpy().copy()
        m.to("fp16").to("fp32")
        assert m.weight.dtype == "fp32"
        # Values within fp16 quantization noise of the original
        np.testing.assert_allclose(m.weight.numpy(), original, atol=1e-3)

    def test_returns_self_for_chaining(self):
        m = ts.nn.Linear(4, 8)
        result = m.to("fp16")
        assert result is m
        # Can chain further calls
        result = m.to("fp32").eval()
        assert result is m
        assert m.training is False

    def test_recurses_into_children(self):
        m = ts.nn.Sequential(ts.nn.Linear(4, 8), ts.nn.RMSNorm(8))
        m.to("fp16")
        for p in m.parameters():
            assert p.dtype == "fp16"

    def test_numpy_dtype_alias_accepted(self):
        m = ts.nn.Linear(4, 4)
        m.to("float16")
        assert m.weight.dtype == "fp16"
        m.to("float32")
        assert m.weight.dtype == "fp32"

    def test_invalid_dtype_rejected(self):
        m = ts.nn.Linear(4, 4)
        with pytest.raises(ValueError, match="Unknown dtype"):
            m.to("bogus")

    def test_device_migration_not_supported(self):
        m = ts.nn.Linear(4, 4)
        with pytest.raises(ValueError, match="Device migration"):
            m.to("cuda")

    def test_drops_stale_grads(self):
        m = ts.nn.Linear(4, 4)
        for p in m.parameters():
            p.grad = np.ones(p.shape, dtype=np.float32)
        m.to("fp16")
        # Stale fp32 grads should be cleared since they don't match the new dtype
        assert all(p.grad is None for p in m.parameters())

    def test_autodiff_still_works_after_migration(self):
        # The autodiff buffer-id registry must be updated by to(dtype) so
        # gradients still find their parameters after the buffer is replaced.
        m = ts.nn.Linear(4, 4)
        m.to("fp32")  # No-op migration but still re-registers buffers

        x = np.random.randn(2, 4).astype(np.float32)
        with ts.autodiff.tape() as t:
            y = m(x)
            loss = ts.ops.reduce(ts.ops.mul(y, y), op="sum")
            t.backward(loss)
        assert all(p.grad is not None for p in m.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# Phase A4 — activation modules + cross-attention + rotary + casted variants
# + cross-entropy loss + clip_grad_norm_
# ─────────────────────────────────────────────────────────────────────────────


class TestActivationModules:
    @pytest.mark.parametrize("cls,reference", [
        (ts.nn.SiLU, lambda x: x / (1.0 + np.exp(-x))),
        (ts.nn.Sigmoid, lambda x: 1.0 / (1.0 + np.exp(-x))),
        (ts.nn.ReLU, lambda x: np.maximum(0, x)),
        (ts.nn.Tanh, np.tanh),
    ])
    def test_activation_matches_reference(self, cls, reference):
        m = cls()
        x = np.random.randn(3, 4).astype(np.float32)
        np.testing.assert_allclose(m(x), reference(x), rtol=1e-5)

    def test_identity_passes_through(self):
        m = ts.nn.Identity()
        x = np.array([1, 2, 3])
        assert m(x) is x

    def test_activation_modules_have_no_parameters(self):
        for cls in (ts.nn.SiLU, ts.nn.Sigmoid, ts.nn.GELU, ts.nn.ReLU, ts.nn.Tanh, ts.nn.Identity):
            assert list(cls().parameters()) == []


class TestMultiHeadCrossAttention:
    def test_requires_explicit_key_value(self):
        m = ts.nn.MultiHeadCrossAttention(embed_dim=16, num_heads=4)
        x = np.random.randn(2, 4, 16).astype(np.float32)
        with pytest.raises(ValueError, match="explicit key and value"):
            m(x)

    def test_cross_attention_shape(self):
        m = ts.nn.MultiHeadCrossAttention(embed_dim=16, num_heads=4)
        q = np.random.randn(2, 4, 16).astype(np.float32)
        k = np.random.randn(2, 8, 16).astype(np.float32)
        v = np.random.randn(2, 8, 16).astype(np.float32)
        assert m(q, k, v).shape == (2, 4, 16)


class TestRotaryEmbedding:
    def test_forward_preserves_shape(self):
        rope = ts.nn.RotaryEmbedding(head_dim=8, max_position=32)
        x = np.random.randn(2, 4, 8).astype(np.float32)
        assert rope(x).shape == x.shape

    def test_position_offset(self):
        rope = ts.nn.RotaryEmbedding(head_dim=8, max_position=32)
        x = np.random.randn(2, 4, 8).astype(np.float32)
        # position=0 vs position=8 should give different outputs (rotation differs)
        y0 = rope(x, position=0)
        y8 = rope(x, position=8)
        assert not np.allclose(y0, y8)

    def test_odd_head_dim_rejected(self):
        with pytest.raises(ValueError, match="even"):
            ts.nn.RotaryEmbedding(head_dim=7)

    def test_position_overflow_rejected(self):
        rope = ts.nn.RotaryEmbedding(head_dim=8, max_position=4)
        x = np.random.randn(8, 8).astype(np.float32)  # seq_len=8 > max_position=4
        with pytest.raises(ValueError, match="exceeds max_position"):
            rope(x)


class TestCastedLinearEmbedding:
    def test_casted_linear_output_dtype(self):
        m = ts.nn.CastedLinear(8, 16, cast_dtype="fp16")
        x = np.random.randn(2, 8).astype(np.float32)
        y = m(x)
        assert y.dtype == np.float16
        assert y.shape == (2, 16)

    def test_casted_linear_inherits_parameters(self):
        m = ts.nn.CastedLinear(8, 16, cast_dtype="fp16")
        names = {n for n, _ in m.named_parameters()}
        assert names == {"weight", "bias"}

    def test_casted_embedding(self):
        m = ts.nn.CastedEmbedding(num_embeddings=10, embedding_dim=4, cast_dtype="fp16")
        out = m(np.array([0, 1, 2]))
        assert out.dtype == np.float16
        assert out.shape == (3, 4)


class TestCrossEntropyLoss:
    def test_uniform_logits_gives_log_n(self):
        # Uniform logits → softmax = 1/C → -log(1/C) = log(C)
        m = ts.nn.CrossEntropyLoss()
        logits = np.zeros((4, 10), dtype=np.float32)
        target = np.array([0, 1, 2, 3])
        loss = float(m(logits, target))
        assert abs(loss - np.log(10)) < 1e-5

    def test_perfect_prediction_low_loss(self):
        m = ts.nn.CrossEntropyLoss()
        # Strong logit on the correct class
        logits = np.full((4, 10), -100.0, dtype=np.float32)
        target = np.array([0, 1, 2, 3])
        for i, t in enumerate(target):
            logits[i, t] = 100.0
        loss = float(m(logits, target))
        assert loss < 1e-5

    @pytest.mark.parametrize("reduction,expected_shape", [
        ("none", (4,)),
        ("sum", ()),
        ("mean", ()),
    ])
    def test_reduction_modes(self, reduction, expected_shape):
        m = ts.nn.CrossEntropyLoss(reduction=reduction)
        logits = np.random.randn(4, 10).astype(np.float32)
        target = np.array([0, 1, 2, 3])
        out = m(logits, target)
        assert np.asarray(out).shape == expected_shape

    def test_invalid_reduction_rejected(self):
        with pytest.raises(ValueError, match="reduction"):
            ts.nn.CrossEntropyLoss(reduction="bogus")


class TestClipGradNorm:
    def test_no_grads_returns_zero(self):
        m = ts.nn.Linear(4, 4)
        # No backward call → all .grad are None
        norm = ts.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
        assert norm == 0.0

    def test_below_threshold_unchanged(self):
        m = ts.nn.Linear(4, 4)
        for p in m.parameters():
            p.grad = np.full(p.shape, 0.01, dtype=np.float32)
        before = [p.grad.numpy().copy() for p in m.parameters()]
        ts.nn.utils.clip_grad_norm_(m.parameters(), max_norm=100.0)
        for b, p in zip(before, m.parameters()):
            np.testing.assert_allclose(p.grad.numpy(), b)

    def test_above_threshold_scaled(self):
        m = ts.nn.Linear(4, 4)
        for p in m.parameters():
            p.grad = np.ones(p.shape, dtype=np.float32) * 5.0
        norm_before = ts.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
        # After clip, total norm should be ≤ max_norm
        scaled = [p.grad.numpy() for p in m.parameters()]
        norm_after = float(np.sqrt(sum((g * g).sum() for g in scaled)))
        assert norm_before > 1.0
        assert norm_after <= 1.0 + 1e-6

    def test_inf_norm(self):
        m = ts.nn.Linear(4, 4)
        for p in m.parameters():
            g = np.zeros(p.shape, dtype=np.float32)
            g.flat[0] = 10.0
            p.grad = g
        ts.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0, norm_type=float("inf"))
        for p in m.parameters():
            assert np.abs(p.grad.numpy()).max() <= 1.0 + 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Composition with autodiff — ensure A4 modules play nicely with tape
# ─────────────────────────────────────────────────────────────────────────────


class TestA4WithAutodiff:
    def test_silu_module_in_tape(self):
        x_p = ts.nn.Parameter(np.random.randn(3, 4).astype(np.float64))
        m = ts.nn.SiLU()
        with ts.autodiff.tape() as t:
            y = m(x_p)
            loss = ts.ops.reduce(y, op="sum")
            t.backward(loss)
        assert x_p.grad is not None
        assert x_p.grad.shape == x_p.shape

    def test_clip_grad_after_backward(self):
        # End-to-end: forward, backward, clip, then verify scaled grads
        np.random.seed(0)
        m = ts.nn.MLP(dim=4, hidden_dim=8)
        x = np.random.randn(2, 4).astype(np.float32)
        with ts.autodiff.tape() as t:
            y = m(x)
            loss = ts.ops.reduce(ts.ops.mul(y, y), op="sum")
            t.backward(loss)
        # Big loss → big grads. Clip them.
        ts.nn.utils.clip_grad_norm_(m.parameters(), max_norm=0.1)
        norm = float(np.sqrt(sum((p.grad.numpy() ** 2).sum() for p in m.parameters())))
        assert norm <= 0.1 + 1e-5
