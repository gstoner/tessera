
import argparse, yaml, random, torch
from src.grpo_roc.utils import set_seed
from src.grpo_roc.dataset import gen_integer_math
from src.grpo_roc.policy import TinyLSTMPolicy
from src.grpo_roc.rollout import run_one_turn, extract_answer_and_format
from src.grpo_roc.reward import answer_only_reward
from src.grpo_roc.trainer import GRPOTrainer, Trajectory

def make_prompt(q: str) -> str:
    sys_prompt = "You are an agent. Use tool_call{...}/tool_call to compute if needed. Wrap final integer in \\boxed{} inside answeranswer ... /answeranswer."
    return sys_prompt + "\n\n" + q + "\n"

def do_rollout(policy, prompt, max_turns=3, tool_timeout_s=1.0, device='cpu'):
    out = policy.sample(prompt, max_len=256)
    text = out.text
    text2, tinfo = run_one_turn(text, tool_timeout_s=tool_timeout_s)
    ans, fmt = extract_answer_and_format(text2)
    return out, tinfo, fmt, ans

def oversample(policy, q, truth, group_size, oversample_factor, **kw):
    oversampled = []
    prompt = make_prompt(q)
    for _ in range(group_size * oversample_factor):
        out, tinfo, fmt, ans = do_rollout(policy, prompt, **kw)
        reward = answer_only_reward(ans, truth)
        traj = Trajectory(tokens=out.tokens, logprob_old=policy.logprob(out.tokens, prompt_len=len(prompt)),
                          reward=reward, tool_calls=tinfo['tool_calls'], tool_errors=tinfo['tool_errors'],
                          answer_tags=fmt['answer_tags'])
        oversampled.append(traj)
    return oversampled

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.device: cfg["device"] = args.device
    set_seed(cfg["seed"])
    device = cfg["device"]
    policy = TinyLSTMPolicy(vocab_size=cfg["vocab_size"]).to(device)
    trainer = GRPOTrainer(policy, eps_clip_low=cfg["eps_clip_low"], eps_clip_high=cfg["eps_clip_high"], lr=cfg["lr"], device=device)
    data = gen_integer_math(4)
    step = 0
    while step < args.steps:
        batch = random.sample(data, k=min(cfg["batch_prompts"], len(data)))
        all_selected = []
        for q, truth in batch:
            overs = oversample(policy, q, truth, cfg["group_size"], cfg["oversample_factor"],
                               max_turns=cfg["max_turns"], tool_timeout_s=cfg["tool_timeout_s"], device=device)
            selected = trainer.select_via_roc(overs, cfg["select_size"])
            all_selected.extend(selected)
        stats = trainer.step(all_selected)
        print(f"step {step}: loss={stats['loss']:.4f} adv_mean={stats['adv_mean']:.3f} selected={len(all_selected)}")
        step += 1

if __name__ == "__main__":
    main()
