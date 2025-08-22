#!/usr/bin/env python3
"""Toy Bayesian attention demo (prototype, tiny sizes)."""
import math, random
from tessera.pp import model, sample, observe, Normal, autoguide, svi

def matmul(A, B):  # lists
    m, n, p = len(A), len(A[0]), len(B[0])
    out = [[0.0]*p for _ in range(m)]
    for i in range(m):
        for k in range(n):
            for j in range(p):
                out[i][j] += A[i][k]*B[k][j]
    return out

def softmax_rowwise(S):
    out = []
    for row in S:
        m = max(row)
        exps = [math.exp(v-m) for v in row]
        Z = sum(exps)
        out.append([e/Z for e in exps])
    return out

def matvec(A, v):
    return [sum(a*b for a,b in zip(row, v)) for row in A]

@model
def BayesAttention(Q, K, V, Y=None):
    D = len(Q[0])
    # Uncertain key projection
    Wk = sample("Wk", Normal([[0.0]*D for _ in range(D)], [[0.05]*D for _ in range(D)]))
    KP = matmul(K, Wk)
    # Scores
    KT = list(map(list, zip(*KP)))
    S = matmul(Q, KT)
    invsqrtD = 1.0/math.sqrt(D)
    S = [[v*invsqrtD for v in row] for row in S]
    P = softmax_rowwise(S)
    # Aggregate
    Yhat = [[sum(pv*vv for pv,vv in zip(p_row, col)) for col in zip(*V)] for p_row in P]
    if Y is not None:
        for i, (yhat_row, y_row) in enumerate(zip(Yhat, Y)):
            for j, (yh, yv) in enumerate(zip(yhat_row, y_row)):
                observe(f"Y_{i}_{j}", Normal(yh, 0.05), value=yv)
    return {"Y": Yhat, "Wk": Wk}

def synth_data(b=2, t=4, d=4, seed=0):
    r = random.Random(seed)
    def mat(rows, cols): return [[r.uniform(-1,1) for _ in range(cols)] for _ in range(rows)]
    Q = mat(b, d); K = mat(b, d); V = mat(b, d)
    # generate a pseudo target Y by a fixed projection
    Wk_true = [[(1 if i==j else 0) for j in range(d)] for i in range(d)]
    KP = matmul(K, Wk_true)
    KT = list(map(list, zip(*KP)))
    S = matmul(Q, KT); invsqrtD = 1.0/math.sqrt(d)
    S = [[v*invsqrtD for v in row] for row in S]
    P = softmax_rowwise(S)
    Y = [[sum(pv*vv for pv,vv in zip(p_row, col)) for col in zip(*V)] for p_row in P]
    return Q, K, V, Y

if __name__ == "__main__":
    Q, K, V, Y = synth_data()
    guide = autoguide.mean_field(BayesAttention)
    res = svi(BayesAttention, guide, data={"Q": Q, "K": K, "V": V, "Y": Y}, steps=100, lr=0.05, seed=2)
    print("Posterior(Wk.loc)[0][:4]:", str(res["posterior"].get("Wk",{}).get("loc","N/A"))[:120])
