
from dataclasses import dataclass
from typing import List, Dict
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .prompts import format_layer1_prompt, format_layer2_prompt, GUIDING_PHRASES
from .rewarders import math_reward
from .grpo import grpo_loss, GRPOConfig
from .sampling import generate_batch

@dataclass
class MGRPOConfig:
    model_name: str = "sshleifer/tiny-gpt2"
    lr: float = 1e-5
    l1_samples: int = 2
    l2_samples: int = 2
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    clip_range: float = 0.2
    kl_coeff: float = 0.01
    seed: int = 1234
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class MGRPOTrainer:
    def __init__(self, cfg: MGRPOConfig):
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        self.cfg = cfg

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.policy = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(cfg.device)
        self.policy_old = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(cfg.device)
        self.policy_ref = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(cfg.device)
        for p in self.policy_ref.parameters():
            p.requires_grad = False

        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=cfg.lr)
        self.grpo_cfg = GRPOConfig(clip_range=cfg.clip_range, kl_coeff=cfg.kl_coeff)

    def train_step(self, batch: List[Dict]):
        """
        batch: list of dicts with keys: question(str), answer(str)
        """
        cfg = self.cfg

        # ==== Layer 1: generate K initial responses ====
        l1_prompts = [format_layer1_prompt(row["question"]) for row in batch]
        l1_prompts = [p for p in l1_prompts for _ in range(cfg.l1_samples)]
        l1_texts, l1_inputs = generate_batch(
            self.policy, self.tokenizer, l1_prompts,
            max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature, top_p=cfg.top_p
        )

        # Group ids: same question index for its K samples
        import torch
        group_ids = torch.arange(len(batch), device=cfg.device).repeat_interleave(cfg.l1_samples)

        # Compute rewards for L1 generations
        rewards_l1 = []
        for i, text in enumerate(l1_texts):
            q_idx = i // cfg.l1_samples
            gold = batch[q_idx]["answer"]
            rewards_l1.append(math_reward(text, gold))
        rewards_l1 = torch.tensor(rewards_l1, dtype=torch.float32, device=cfg.device)

        # Re-score logits under pi (current), old, ref
        logits_pi_l1 = self.policy(**l1_inputs, output_hidden_states=False, output_attentions=False, use_cache=False).logits
        logits_old_l1 = self.policy_old(**l1_inputs, output_hidden_states=False, output_attentions=False, use_cache=False).logits
        logits_ref_l1 = self.policy_ref(**l1_inputs, output_hidden_states=False, output_attentions=False, use_cache=False).logits

        labels_mask_l1 = torch.ones_like(l1_inputs["input_ids"], dtype=torch.float32, device=cfg.device)
        loss_l1, stats_l1 = grpo_loss(
            logits_pi_l1, logits_old_l1, logits_ref_l1,
            l1_inputs["input_ids"], labels_mask_l1, rewards_l1, group_ids, self.grpo_cfg
        )

        # ==== Layer 2: self-correction ====
        l2_prompts = []
        for i, text in enumerate(l1_texts):
            q_idx = i // cfg.l1_samples
            phrase = random.choice(GUIDING_PHRASES)
            l2_prompts.append(format_layer2_prompt(batch[q_idx]["question"], text, phrase))

        l2_prompts = [p for p in l2_prompts for _ in range(cfg.l2_samples)]
        l2_texts, l2_inputs = generate_batch(
            self.policy, self.tokenizer, l2_prompts,
            max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature, top_p=cfg.top_p
        )

        rewards_l2 = []
        group_ids_l2 = []
        for i, text in enumerate(l2_texts):
            l1_idx = i // cfg.l2_samples
            q_idx = l1_idx // cfg.l1_samples
            gold = batch[q_idx]["answer"]
            r2 = math_reward(text, gold)
            rewards_l2.append(r2)
            group_ids_l2.append(q_idx)

        rewards_l2 = torch.tensor(rewards_l2, dtype=torch.float32, device=cfg.device)
        group_ids_l2 = torch.tensor(group_ids_l2, dtype=torch.long, device=cfg.device)

        logits_pi_l2 = self.policy(**l2_inputs, output_hidden_states=False, output_attentions=False, use_cache=False).logits
        logits_old_l2 = self.policy_old(**l2_inputs, output_hidden_states=False, output_attentions=False, use_cache=False).logits
        logits_ref_l2 = self.policy_ref(**l2_inputs, output_hidden_states=False, output_attentions=False, use_cache=False).logits

        labels_mask_l2 = torch.ones_like(l2_inputs["input_ids"], dtype=torch.float32, device=cfg.device)
        loss_l2, stats_l2 = grpo_loss(
            logits_pi_l2, logits_old_l2, logits_ref_l2,
            l2_inputs["input_ids"], labels_mask_l2, rewards_l2, group_ids_l2, self.grpo_cfg
        )

        # ==== Combine and update ====
        loss = loss_l1 + loss_l2
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        stats = {
            **{f"l1/{k}": v for k, v in stats_l1.items()},
            **{f"l2/{k}": v for k, v in stats_l2.items()},
            "loss_total": float(loss.item()),
        }
        return stats
