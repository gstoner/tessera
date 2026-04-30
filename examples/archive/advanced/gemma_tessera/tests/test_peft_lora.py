"""
Tests for tessera_gemma.peft — LoRA adapters and QLoRA simulation.
"""
import pytest
import torch
import torch.nn as nn
from tessera_gemma.peft import (
    LoRAAdapter, LoRALinear, apply_lora, apply_lora_regex,
    lora_state_dict, load_lora_state_dict,
    merge_lora, unmerge_lora,
    QLinearSim, apply_qlora_sim,
    freeze_by_regex, param_groups_with_adapter_lrmult,
)
from tessera_gemma import GemmaConfig, TesseraGemmaForCausalLM


# ---------------------------------------------------------------------------
# LoRAAdapter
# ---------------------------------------------------------------------------
class TestLoRAAdapter:
    def test_zero_init(self):
        ad = LoRAAdapter(64, 128, rank=4, alpha=8.0, dropout=0.0, name="test")
        x = torch.randn(2, 8, 64)
        # lora_B is zero → output should be zero
        out = ad(x)
        assert torch.all(out == 0)

    def test_output_shape(self):
        ad = LoRAAdapter(64, 128, rank=4, alpha=8.0, dropout=0.0, name="t")
        x = torch.randn(3, 10, 64)
        assert ad(x).shape == (3, 10, 128)

    def test_scaling(self):
        ad = LoRAAdapter(4, 4, rank=2, alpha=4.0, dropout=0.0, name="s")
        assert ad.scaling == pytest.approx(2.0)   # 4.0 / 2

    def test_disabled_returns_zeros(self):
        ad = LoRAAdapter(8, 8, rank=2, alpha=4.0, dropout=0.0, name="t")
        ad.lora_B.data.fill_(1.0)  # non-zero B
        ad.enabled = False
        x = torch.randn(1, 4, 8)
        assert torch.all(ad(x) == 0)


# ---------------------------------------------------------------------------
# LoRALinear
# ---------------------------------------------------------------------------
class TestLoRALinear:
    def setup_method(self):
        base = nn.Linear(32, 64, bias=False)
        self.ll = LoRALinear(base)

    def test_no_adapter_same_as_base(self):
        x = torch.randn(2, 32)
        expected = self.ll.base(x)
        assert torch.equal(self.ll(x), expected)

    def test_add_adapter(self):
        self.ll.add_adapter("a", rank=4, alpha=8.0)
        assert "a" in self.ll.adapters

    def test_duplicate_adapter_raises(self):
        self.ll.add_adapter("b", rank=2)
        with pytest.raises(ValueError, match="already exists"):
            self.ll.add_adapter("b", rank=2)

    def test_adapter_zero_b_no_effect(self):
        self.ll.add_adapter("c", rank=4, alpha=8.0)
        x = torch.randn(2, 32)
        base_out = self.ll.base(x)
        with_adapter = self.ll(x)
        assert torch.equal(base_out, with_adapter)  # lora_B=0

    def test_disabled_adapter_no_effect(self):
        self.ll.add_adapter("d", rank=4, alpha=8.0)
        self.ll.adapters["d"].lora_B.data.fill_(0.1)
        self.ll.enable_adapter("d", enabled=False)
        x = torch.randn(2, 32)
        assert torch.equal(self.ll(x), self.ll.base(x))

    def test_merge_unmerge_roundtrip(self):
        base = nn.Linear(16, 32, bias=False)
        ll = LoRALinear(base)
        ll.add_adapter("e", rank=2, alpha=4.0)
        # Set non-trivial adapter weights
        ll.adapters["e"].lora_A.data.fill_(0.01)
        ll.adapters["e"].lora_B.data.fill_(0.01)

        w_before = ll.base.weight.data.clone()
        ll.merge()
        assert ll.merged
        ll.unmerge()
        assert not ll.merged
        assert torch.allclose(ll.base.weight.data, w_before, atol=1e-5)


# ---------------------------------------------------------------------------
# apply_lora / apply_lora_regex
# ---------------------------------------------------------------------------
class TestApplyLora:
    def _tiny_model(self):
        cfg = GemmaConfig.debug_tiny()
        return TesseraGemmaForCausalLM(cfg)

    def test_apply_lora_creates_adapters(self):
        m = self._tiny_model()
        count = apply_lora(m, patterns=["q_proj", "v_proj"], rank=4)
        assert count > 0

    def test_apply_lora_wraps_linear(self):
        m = self._tiny_model()
        apply_lora(m, patterns=["q_proj"], rank=4)
        for name, mod in m.named_modules():
            if "q_proj" in name and isinstance(mod, LoRALinear):
                return  # found at least one
        pytest.fail("No LoRALinear found for q_proj")

    def test_apply_lora_regex_pattern(self):
        m = self._tiny_model()
        rules = [{"pattern": r".*\.q_proj", "name": "qv", "rank": 8, "alpha": 16.0}]
        count = apply_lora_regex(m, rules)
        assert count == len(m.layers)   # one q_proj per layer

    def test_apply_all_attn_projections(self):
        m = self._tiny_model()
        count = apply_lora(m, patterns=["q_proj", "k_proj", "v_proj", "o_proj"], rank=4)
        assert count == 4 * len(m.layers)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
class TestLoRACheckpoint:
    def test_state_dict_keys(self):
        cfg = GemmaConfig.debug_tiny()
        m = TesseraGemmaForCausalLM(cfg)
        apply_lora(m, patterns=["q_proj"], rank=4)
        sd = lora_state_dict(m)
        assert all(".A" in k or ".B" in k for k in sd)
        assert len(sd) > 0

    def test_load_state_dict_roundtrip(self):
        cfg = GemmaConfig.debug_tiny()
        m1 = TesseraGemmaForCausalLM(cfg)
        m2 = TesseraGemmaForCausalLM(cfg)
        apply_lora(m1, patterns=["q_proj"], rank=4)
        apply_lora(m2, patterns=["q_proj"], rank=4)

        # Set random adapter weights on m1
        for mod in m1.modules():
            if isinstance(mod, LoRALinear):
                for ad in mod.adapters.values():
                    ad.lora_B.data.normal_()

        sd = lora_state_dict(m1)
        load_lora_state_dict(m2, sd)

        sd2 = lora_state_dict(m2)
        for k in sd:
            assert torch.allclose(sd[k], sd2[k])


# ---------------------------------------------------------------------------
# merge / unmerge on full model
# ---------------------------------------------------------------------------
class TestMergeUnmerge:
    def test_merge_count(self):
        cfg = GemmaConfig.debug_tiny()
        m = TesseraGemmaForCausalLM(cfg)
        apply_lora(m, patterns=["q_proj"], rank=4)
        count = merge_lora(m)
        assert count == len(m.layers)

    def test_unmerge_count(self):
        cfg = GemmaConfig.debug_tiny()
        m = TesseraGemmaForCausalLM(cfg)
        apply_lora(m, patterns=["v_proj"], rank=4)
        merge_lora(m)
        count = unmerge_lora(m)
        assert count == len(m.layers)

    def test_output_unchanged_after_merge_unmerge(self):
        cfg = GemmaConfig.debug_tiny()
        m = TesseraGemmaForCausalLM(cfg).eval()
        apply_lora(m, patterns=["q_proj"], rank=4)
        x = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            y_before = m(x).clone()
        merge_lora(m)
        unmerge_lora(m)
        with torch.no_grad():
            y_after = m(x)
        assert torch.allclose(y_before, y_after, atol=1e-4)


# ---------------------------------------------------------------------------
# QLoRA simulation
# ---------------------------------------------------------------------------
class TestQLinearSim:
    def test_output_shape(self):
        base = nn.Linear(64, 128, bias=False)
        q = QLinearSim(base)
        x = torch.randn(2, 8, 64)
        assert q(x).shape == (2, 8, 128)

    def test_disabled_uses_base(self):
        base = nn.Linear(16, 32, bias=False)
        q = QLinearSim(base)
        q.set_enabled(False) if hasattr(q, "set_enabled") else setattr(q, "enabled", False)
        x = torch.randn(1, 16)
        assert torch.allclose(q(x), base(x))

    def test_apply_qlora_sim_count(self):
        cfg = GemmaConfig.debug_tiny()
        m = TesseraGemmaForCausalLM(cfg)
        apply_lora(m, patterns=["q_proj"], rank=4)
        count = apply_qlora_sim(m, patterns=["q_proj"])
        assert count > 0


# ---------------------------------------------------------------------------
# Freeze and LR groups
# ---------------------------------------------------------------------------
class TestFreezeAndLR:
    def test_freeze_embed(self):
        cfg = GemmaConfig.debug_tiny()
        m = TesseraGemmaForCausalLM(cfg)
        count = freeze_by_regex(m, patterns=["embed_tokens"])
        assert count > 0
        for name, p in m.named_parameters():
            if "embed_tokens" in name:
                assert not p.requires_grad

    def test_param_groups_keys(self):
        cfg = GemmaConfig.debug_tiny()
        m = TesseraGemmaForCausalLM(cfg)
        apply_lora(m, patterns=["q_proj"], rank=4)
        groups = param_groups_with_adapter_lrmult(m, base_lr=1e-4,
                                                   adapter_lr_mult={"q_proj": 10.0})
        lrs = {g["lr"] for g in groups}
        # Should have adapter LR = 1e-3 and base LR = 1e-4
        assert any(abs(lr - 1e-3) < 1e-9 for lr in lrs)
        assert any(abs(lr - 1e-4) < 1e-9 for lr in lrs)
