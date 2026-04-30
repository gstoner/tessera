import argparse, numpy as np
from typing import Tuple
from morl_utils import pareto_mask, scalarize_linear, scalarize_tchebycheff, pcgrad_pairwise
from envs import DeepSeaTreasure, ResourceGathering

# Tiny linear policy/value to keep the demo self-contained (numpy only).
class TinyPolicy:
    def __init__(self, obs_dim, act_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.W = 0.01 * rng.standard_normal((obs_dim, act_dim)).astype(np.float32)

    def logits(self, obs):
        return obs @ self.W  # [B, A]

    def act(self, obs, rng):
        l = self.logits(obs)
        # softmax sample
        e = np.exp(l - np.max(l, axis=-1, keepdims=True))
        p = e / np.sum(e, axis=-1, keepdims=True)
        a = np.array([rng.choice(p.shape[1], p=p[i]) for i in range(p.shape[0])], dtype=np.int64)
        return a, p

    def grad_logp(self, obs, a):
        # grad wrt W for chosen actions (REINFORCE style)
        l = self.logits(obs)
        e = np.exp(l - np.max(l, axis=-1, keepdims=True))
        p = e / np.sum(e, axis=-1, keepdims=True)
        onehot = np.zeros_like(p)
        onehot[np.arange(p.shape[0]), a] = 1.0
        g_logits = onehot - p  # d log p(a|s) / d logits
        # d logits / d W = obs^T
        # Return [A, obs_dim] then reshape to W shape
        batch = obs.shape[0]
        gW = np.zeros_like(self.W)
        for i in range(batch):
            gW += np.outer(obs[i], g_logits[i])
        return gW / batch

    def step(self, lr, grad_W):
        self.W += lr * grad_W

class TinyValue:
    def __init__(self, obs_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.w = 0.01 * rng.standard_normal((obs_dim,)).astype(np.float32)
        self.b = 0.0

    def value(self, obs):
        return obs @ self.w + self.b

    def grad_mse(self, obs, target):
        pred = self.value(obs)
        diff = pred - target
        gw = (2.0/obs.shape[0]) * (obs.T @ diff)
        gb = (2.0/obs.shape[0]) * np.sum(diff)
        return gw, gb, pred

    def step(self, lr, gw, gb):
        self.w -= lr * gw
        self.b -= lr * gb

def make_env(name, seed):
    if name == "deep_sea": return DeepSeaTreasure(seed=seed)
    if name == "gather":   return ResourceGathering(seed=seed)
    raise ValueError("unknown env")

def rollout(env, policy, horizon, seed):
    rng = np.random.default_rng(seed)
    obs_buf, act_buf, rew_buf, done_buf = [], [], [], []
    o = env.reset()
    for t in range(horizon):
        a, _ = policy.act(o[None, :], rng)
        a = int(a[0])
        o2, r, d, _ = env.step(a)
        obs_buf.append(o); act_buf.append(a); rew_buf.append(r); done_buf.append(d)
        o = o2
        if d: o = env.reset()
    return (np.stack(obs_buf).astype(np.float32),
            np.array(act_buf, dtype=np.int64),
            np.stack(rew_buf).astype(np.float32),
            np.array(done_buf, dtype=bool))

def gae(rewards, values, gamma=0.99, lam=0.95):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0
    for t in reversed(range(T)):
        nextv = values[t+1] if t+1 < len(values) else 0.0
        delta = rewards[t] + gamma * nextv - values[t]
        lastgaelam = delta + gamma * lam * lastgaelam
        adv[t] = lastgaelam
    return adv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="deep_sea", choices=["deep_sea","gather"])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--horizon", type=int, default=128)
    ap.add_argument("--pref", type=str, default="0.5,0.5")
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scalar", choices=["linear","tcheby"], default="linear")
    args = ap.parse_args()

    w = np.array([float(x) for x in args.pref.split(",")], dtype=np.float32)
    w = w / max(1e-8, np.sum(w))

    env = make_env(args.env, args.seed)
    obs_dim = env.reset().shape[0]
    act_dim = 4  # both envs use 4 actions

    pol = TinyPolicy(obs_dim, act_dim, seed=args.seed)
    val = TinyValue(obs_dim, seed=args.seed+1)

    total_steps = 0
    while total_steps < args.steps:
        obs, act, rew, done = rollout(env, pol, args.horizon, seed=args.seed + total_steps)
        # Per-objective GAE (using a simple value fn per objective or shared approx. here use scalarized baseline)
        # Compute scalarized rewards per step
        if args.scalar == "linear":
            scal = scalarize_linear(rew, w)
        else:
            z = np.max(rew, axis=0)  # crude ref point
            scal = scalarize_tchebycheff(rew, w, z)

        vals = np.array([val.value(o) for o in obs], dtype=np.float32)
        adv  = gae(scal, vals)
        ret  = adv + vals

        # Policy gradient (REINFORCE with advantage baseline)
        # Combine per-objective grads using PCGrad on a toy per-objective gradient estimate:
        # We approximate per-objective grad by weighting rewards with one-hot on objective m.
        # (For a real system, compute grads via differentiable path per objective)
        grads = []
        for m in range(rew.shape[1]):
            wm = np.zeros_like(w); wm[m] = 1.0
            scal_m = scalarize_linear(rew, wm)
            adv_m  = gae(scal_m, np.zeros_like(scal_m))
            # crude gradient estimate scales grad_logp by normalized advantage
            gW = pol.grad_logp(obs, act) * (np.mean(adv_m) / (np.std(adv_m)+1e-5))
            grads.append(gW.flatten())
        grads = np.stack(grads, axis=0)
        g_comb = pcgrad_pairwise(grads).reshape(pol.W.shape)

        pol.step(args.lr, g_comb)

        # Value update (MSE on scalarized return)
        gw, gb, pred = val.grad_mse(obs, ret)
        val.step(lr=0.5*args.lr, gw=gw, gb=gb)

        total_steps += len(obs)

        if (total_steps // args.horizon) % 5 == 0:
            # Pareto frontier report from a batch of outcomes
            pts = rew.reshape(-1, rew.shape[-1])
            mask = pareto_mask(pts)
            nfront = int(mask.sum())
            print(f"[{total_steps:6d}] pref={w.tolist()} return≈{float(ret.mean()):.2f} pareto_front={nfront} / {len(pts)}")

if __name__ == "__main__":
    main()
