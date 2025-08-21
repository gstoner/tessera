# Tessera Example — PINN for 2D Navier–Stokes (Incompressible)

**Goal:** Demonstrate how Tessera expresses Physics‑Informed Neural Networks (PINNs) for a 2D incompressible Navier–Stokes system using stream‑function or pressure–velocity form, with operator adjoints for coupled PDEs.

---

## 1) PDE Setup

**Incompressible Navier–Stokes (2D):**

- Velocity: u(x,y,t) = (u, v)
- Continuity: ∂u/∂x + ∂v/∂y = 0
- Momentum: ∂u/∂t + u·∇u = −∇p + ν∇²u + f

**Stream‑Function formulation:** u = ∂ψ/∂y, v = −∂ψ/∂x ensures incompressibility.

---

## 2) Tessera Model

```python
from tessera import op, graph

# Neural field ψ(x,y,t) → stream function
psi = op.mlp(in_dim=3, hidden=[256,256,256], out_dim=1, activation="tanh")

def velocity_from_stream(xyt):
    ψ = psi(xyt)                        # [N,1]
    u = op.grad(ψ, x="y")               # ∂ψ/∂y
    v = -op.grad(ψ, x="x")              # −∂ψ/∂x
    return op.concat([u,v], axis=-1)    # [N,2]
```

---

## 3) Residual Operators

```python
def pde_residuals(xyt, ν, f):
    u = velocity_from_stream(xyt)        # [N,2]
    ux = op.grad(u[:,0], x="x"); uy = op.grad(u[:,0], x="y")
    vx = op.grad(u[:,1], x="x"); vy = op.grad(u[:,1], x="y")

    # Continuity (auto‑satisfied by stream function form)
    div = ux + vy                        # should be ~0

    # Momentum residuals (projected, pressure‑free form)
    ut = op.grad(u, x="t")
    lap_u = op.laplacian(u[:,0], ["x","y"])
    lap_v = op.laplacian(u[:,1], ["x","y"])
    adv_u = u[:,0]*ux + u[:,1]*uy
    adv_v = u[:,0]*vx + u[:,1]*vy

    res_u = ut[:,0] + adv_u - ν*lap_u - f[:,0]
    res_v = ut[:,1] + adv_v - ν*lap_v - f[:,1]
    return div, res_u, res_v
```

---

## 4) Loss Function with Boundary Conditions

```python
def pinn_loss(samples, ν):
    xyt_int, xyt_bc, u_bc, f = samples
    div, res_u, res_v = pde_residuals(xyt_int, ν, f)

    # PDE residual loss
    L_pde = op.mse(res_u, 0.0) + op.mse(res_v, 0.0) + op.mse(div, 0.0)

    # Boundary loss (Dirichlet / no‑slip)
    u_hat = velocity_from_stream(xyt_bc)
    L_bc = op.mse(u_hat, u_bc)

    return L_pde + λ_bc * L_bc
```

---

## 5) Training Step (Autodiff & Operator Adjoints)

```python
@graph.training_step(module="PINN_NavierStokes")
def step(batch):
    loss = pinn_loss(batch, ν=1e-3)
    grads = graph.backward(loss)    # operator adjoints handle coupled PDEs
    return grads, {"loss": loss}
```

---

## 6) Discretization & Sampling

- Collocation points in domain Ω × [0,T]
- Boundary sampling for Dirichlet/Neumann conditions
- Optionally add sensor data (supervision) for hybrid loss

---

## 7) Numerical Tips

- Use scaled inputs (x,y,t) → (−1,1)
- Soft constraints for boundary + PDE balance (tune λ_bc)
- Mixed precision: enable bf16/fp32 accumulation
- Use spectral regularization (e.g., low‑pass on ψ) if needed
