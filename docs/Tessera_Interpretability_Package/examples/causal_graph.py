#!/usr/bin/env python3
import torch

def causal_graph_from_features(Fmat, y, l2=1e-2, thresh=1e-2):
    N,D = Fmat.shape
    A = torch.zeros(D,D)
    for j in range(D):
        idx = [i for i in range(D) if i!=j]
        X = Fmat[:, idx]
        beta = torch.linalg.lstsq(X.T@X + l2*torch.eye(X.shape[1]), X.T@Fmat[:,j]).solution
        A[idx, j] = beta
    A[torch.abs(A) < thresh] = 0.0
    w = torch.linalg.lstsq(Fmat, y).solution
    return A, w

if __name__ == "__main__":
    torch.manual_seed(0)
    N,D = 512, 6
    W_true = torch.tensor([[0,0.8,0,0,0,0],
                           [0,0,  0,0,0,0],
                           [0,0.3,0,0.5,0,0],
                           [0,0,  0,0,0,0.6],
                           [0,0,  0,0,0,0  ],
                           [0,0,  0,0,0,0  ]], dtype=torch.float32)
    F0 = torch.randn(N, D)
    F = F0 + F0 @ W_true
    y = (F @ torch.randn(D)).tanh()
    A, w = causal_graph_from_features(F, y)
    print("Estimated edges:", int((A.abs()>0).sum().item()))
