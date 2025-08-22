import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return F.relu(self.fc(x))

class GatedMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
    def forward(self, x):
        return F.relu(self.fc2(F.relu(self.fc1(x))))

class MixedOp(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.ops = nn.ModuleList([
            LinearBlock(in_dim, out_dim),
            GatedMLP(in_dim, 32, out_dim)
        ])
        self.alpha = nn.Parameter(torch.zeros(len(self.ops)))
    def forward(self, x, tau=1.0):
        weights = F.gumbel_softmax(self.alpha, tau=tau, hard=False)
        out = sum(w * op(x) for w, op in zip(weights, self.ops))
        return out, weights

def run_search(steps=200, device="cpu"):
    torch.manual_seed(42)
    in_dim, out_dim = 16, 8
    model = MixedOp(in_dim, out_dim).to(device)
    w_opt = optim.Adam([p for n, p in model.named_parameters() if n != "alpha"], lr=1e-2)
    a_opt = optim.Adam([model.alpha], lr=3e-2)
    X_train, y_train = torch.randn(64, in_dim, device=device), torch.randn(64, out_dim, device=device)
    X_val, y_val = torch.randn(64, in_dim, device=device), torch.randn(64, out_dim, device=device)
    tau = 5.0
    for step in range(steps):
        # Weight update
        model.train()
        out, _ = model(X_train, tau)
        loss = F.mse_loss(out, y_train)
        w_opt.zero_grad(); loss.backward(); w_opt.step()

        # Arch update with cost penalty
        model.eval()
        out, weights = model(X_val, tau)
        ce = F.mse_loss(out, y_val)
        cost = (weights[1] * 0.2) # pretend expert 1 is more costly
        arch_loss = ce + cost
        a_opt.zero_grad(); arch_loss.backward(); a_opt.step()

        tau = max(0.5, tau * 0.99)
        if step % 50 == 0:
            print(f"Step {step} Arch Weights: {weights.detach().cpu().numpy()}")

    print("Final alphas:", model.alpha.data)
    print("Chosen op:", torch.argmax(model.alpha).item())

if __name__ == "__main__":
    run_search()
