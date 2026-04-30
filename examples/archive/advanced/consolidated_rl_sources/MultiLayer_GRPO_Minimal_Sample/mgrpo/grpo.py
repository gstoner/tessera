
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class GRPOConfig:
    clip_range: float = 0.2
    kl_coeff: float = 0.01

def compute_logprobs(logits, input_ids, labels_mask):
    """
    Compute sequence and token logprobs for a causal LM.
    logits: (B, T, V)
    input_ids: (B, T)
    labels_mask: (B, T) float mask of which positions count toward loss
    """
    logprobs = F.log_softmax(logits, dim=-1)  # (B,T,V)
    token_lp = logprobs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)  # (B,T)
    token_lp = token_lp * labels_mask  # mask out non-label positions
    seq_logprob = token_lp.sum(dim=-1)  # (B,)
    return seq_logprob, token_lp

def sequence_kl(logits_pi, logits_ref, labels_mask):
    """
    Token-wise KL(p||q) averaged over label tokens.
    """
    p = F.log_softmax(logits_pi, dim=-1).exp()
    logp = F.log_softmax(logits_pi, dim=-1)
    logq = F.log_softmax(logits_ref, dim=-1)
    kl = (p * (logp - logq)).sum(-1)  # (B,T)
    denom = labels_mask.sum(dim=-1).clamp_min(1.0)
    return (kl * labels_mask).sum(dim=-1) / denom  # (B,)

def grpo_loss(
    logits_pi, logits_old, logits_ref, input_ids, labels_mask, rewards, group_ids, cfg: GRPOConfig
):
    """
    GRPO objective with PPO-style ratio clipping and KL penalty to a frozen reference.

    Args:
        logits_pi:  (B,T,V) logits of current policy
        logits_old: (B,T,V) logits of behavior/old policy (for importance ratios)
        logits_ref: (B,T,V) logits of frozen reference policy (for KL)
        input_ids:  (B,T)    token ids used for scoring
        labels_mask:(B,T)    float mask for which positions to include
        rewards:    (B,)     scalar reward per sequence (e.g., 0/1)
        group_ids:  (B,)     int ids mapping each sequence to its query group
        cfg:                GRPOConfig
    """
    seq_lp_pi, _ = compute_logprobs(logits_pi, input_ids, labels_mask)
    seq_lp_old, _ = compute_logprobs(logits_old, input_ids, labels_mask)

    # group baseline: mean reward within each group id
    with torch.no_grad():
        baselines = torch.zeros_like(rewards)
        unique = torch.unique(group_ids)
        for gid in unique.tolist():
            mask = (group_ids == gid)
            if mask.any():
                baselines[mask] = rewards[mask].mean()
        advantages = rewards - baselines  # (B,)

    ratios = torch.exp(seq_lp_pi - seq_lp_old)  # (B,)

    # PPO-style clipping
    unclipped = ratios * advantages
    clipped = torch.clamp(ratios, 1 - cfg.clip_range, 1 + cfg.clip_range) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()

    # KL to reference
    kl_seq = sequence_kl(logits_pi, logits_ref, labels_mask)
    kl_loss = cfg.kl_coeff * kl_seq.mean()

    loss = policy_loss + kl_loss
    stats = {
        "policy_loss": float(policy_loss.item()),
        "kl_loss": float(kl_loss.item()),
        "adv_mean": float(advantages.mean().item()),
        "ratio_mean": float(ratios.mean().item()),
        "reward_mean": float(rewards.mean().item()),
    }
    return loss, stats
